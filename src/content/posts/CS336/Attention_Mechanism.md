---
title: Attention Mechanism —— 从缩放点积到因果多头注意力
published: 2026-03-06
description: 系统讲解Transformer注意力机制的核心原理与完整实现：从数值稳定的Softmax出发，推导缩放点积注意力的数学公式与几何意义；深入剖析因果掩码的下三角矩阵机制及其在自回归语言模型中的关键作用；详解多头注意力的拆分与合并逻辑及并行计算优势；最后阐述RoPE旋转位置编码如何直接融入注意力计算，与因果掩码协同实现带位置感知的因果注意力。
cover: "/assets/images/posts/attention_mechanism.png"
coverInContent: false
tags: [Transformer, Attention, 缩放点积注意力, 因果掩码, MHA, RoPE, Softmax, LLM基础]
category: CS336
draft: false
---

# Attention Mechanism —— 从缩放点积到因果多头注意力

## 引言

在前两篇博客中，我们完成了**从文本到token ID的转化**（BPE分词器），以及**构建Transformer所需的所有基础算子**（线性层、嵌入层、RMSNorm、SwiGLU、RoPE）。现在，我们终于要触及Transformer最核心、最具革命性的组件——**注意力机制（Attention Mechanism）**。

如果说Transformer是一栋大厦，那么注意力机制就是它的**承重结构**。没有注意力，Transformer就退化为一个普通的**全连接网络**，失去了捕捉**长距离依赖**的能力。2017年《Attention Is All You Need》论文的发布，标志着**注意力机制**正式成为深度学习的主流范式。如今，从GPT到Llama，从BERT到T5，所有主流大语言模型都建立在注意力机制之上。

本文将深入解析注意力的每一个实现细节：
- **数值稳定的Softmax**：一个看似微小却至关重要的工程技巧
- **缩放点积注意力**：注意力最核心的数学公式
- **因果掩码**：让语言模型只能“看到过去”
- **多头机制**：如何让模型从多个角度关注输入
- **RoPE与注意力的融合**：位置编码如何直接嵌入注意力计算

---

## 一、数值稳定的Softmax

### 1.1 Softmax的本质

Softmax函数**将一个向量映射为概率分布**。对于输入向量 $z = (z_1, z_2, ..., z_n)$，其输出为：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

这个公式有两个关键特性：
1. **输出总和为1**：$\sum_i \text{softmax}(z_i) = 1$
2. **保持单调性**：$z_i$ 越大，输出概率越大

在注意力机制中，Softmax应用于**注意力分数**（Q与K的点积除以$\sqrt{d_k}$），将分数转换为“关注权重”——**权重越大，表示模型越关注对应的Value向量**。

### 1.2 数值溢出问题

上述公式在实现中存在一个严重的数值隐患：**指数函数 $e^{z_i}$ 增长极快**。当 $z_i$ 取值较大时（如大于100），$e^{100}$ 会导致**浮点数溢出**（overflow），在Python中表现为 **`inf`**；而当 $z_i$ 取值较小（如小于-100），$e^{-100}$ 会下溢（underflow）为 **`0`**。

在注意力中，**$z_i$ 是Q和K的点积除以$\sqrt{d_k}$**，其绝对值可能很大，尤其是在训练初期或模型参数尺度较大时。**数值溢出会导致梯度变为NaN，训练崩溃**。

### 1.3 防止溢出的技巧

解决溢出的方案为**在指数运算前，从每个输入中减去其最大值**。

对于任意向量 $z$，令 $z_{\max} = \max_i z_i$。我们对 $z_i - z_{\max}$ 应用Softmax：

$$\text{softmax}(z_i) = \frac{e^{z_i - z_{\max}}}{\sum_{j=1}^n e^{z_j - z_{\max}}}$$

这个变换是**数学上恒等**的，因为：

$$\frac{e^{z_i - z_{\max}}}{\sum_j e^{z_j - z_{\max}}} = \frac{e^{z_i} / e^{z_{\max}}}{\sum_j e^{z_j} / e^{z_{\max}}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

现在，所有指数函数的输入都 $\leq 0$，最大值为0。这完美地防止了溢出（最大值也不会超过$e^0=1$），同时**保留下溢问题不影响梯度**。

代码实现极为简洁：

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    x_softmax = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_softmax
```

**关键点**：**`keepdim=True`** 保留了**原始张量的维度结构**，使得**广播减法**可以正确执行（将每个元素减去对应**行的最大值**）。

### 1.4 在注意力中的应用

在缩放点积注意力中，通常直接使用`torch.softmax`，它内部已经实现了**数值稳定技巧**。我们自定义的`softmax`函数则展示了其底层原理。在注意力分数张量中，**`dim=-1`对应于`key_len`维度** —— 即**对每个query，将其对所有key的注意力分数归一化为概率分布**。

---

## 二、缩放点积注意力

### 2.1 注意力机制的直觉理解

在深入数学之前，先建立直觉：**注意力机制本质上是一个“查询-键值”检索系统**。

想象你在图书馆找一本书：
- **Query（查询）**：你要找的**书名或主题**
- **Key（键）**：每本书的**标签或摘要**
- **Value（值）**：书本身的**内容**

你通过**比对Query和所有Key的相似度**，找到**最相关**的几本书，然后根据**相似度加权组**合它们的**Value（内容）**，得到最终的信息。

在Transformer中，这个过程被形式化为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 2.2 公式的逐行实现

```python
def Scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.einsum('... q d, ... k d -> ... q k', Q, K) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.einsum('... q k, ... k v -> ... q v', attention_weights, V)
    return output
```

**第1步：计算点积分数**
```python
scores = torch.einsum('... q d, ... k d -> ... q k', Q, K) / (d_k ** 0.5)
```
`einsum`表达式 `'... q d, ... k d -> ... q k'` 表示（`...` 表示前导维度（如batch_size和num_heads）被保留）：
- 输入Q的形状为 `(..., q, d_k)`（q是query数量，d_k是维度）
- 输入K的形状为 `(..., k, d_k)`（k是key数量）
- 输出形状为 `(..., q, k)`，其中**每个元素是Q中第i个向量与K中第j个向量的点积**

**关键点**：除以 $\sqrt{d_k}$ 的原因 —— 这是缩放点积注意力中“**缩放**”二字的来源。
- 当 $d_k$ 较大时，点积 $Q \cdot K$ 的**方差**也会很大（约等于 $d_k$），导致softmax的**输入值分布在较宽的范围**，梯度进入**饱和区**（softmax输出的概率接近0或1），从而**梯度很小，训练缓慢**。
- 除以 $\sqrt{d_k}$ 将方差拉回到1左右，**使softmax的梯度保持在一个健康的范围**。这是Transformer作者的一个关键洞见。

**第2步：应用掩码**
```python
if mask is not None:
    scores = scores.masked_fill(mask == False, float('-inf'))
```
掩码是一个布尔张量，形状与scores兼容（通常为 `(batch, 1, q, k)`）。值为`False`的位置表示“**不允许关注**”，我们将对应的分数设为 `-inf`。这样在softmax中，$e^{-\infty}=0$，该位置的权重变为0。

**第3步：Softmax归一化**
```python
attention_weights = torch.softmax(scores, dim=-1)
```
**对每个query（最后一维是key维度），将其分数转换为概率分布**。

**第4步：加权求和**
```python
output = torch.einsum('... q k, ... k v -> ... q v', attention_weights, V)
```
将权重张量 `(..., q, k)` 与V `(..., k, v)` 进行**张量乘法**，得到输出 `(..., q, v)`——即**每个query聚合了所有value的加权和**。

### 2.3 注意力的几何意义

从几何视角看，注意力机制将每个**查询向量**映射到由所有**键值对**构成的**凸组合**（因为softmax输出非负且和为1）。输出向量位于由V的行向量张成的凸包中。这种“**软选择**”机制使得模型可以**根据输入动态调整信息流**，这正是注意力的强大之处。

---

## 三、因果掩码：语言模型的时间单向性

### 3.1 为什么需要因果性？

语言模型的任务是**自回归**生成：**给定已生成的token序列，预测下一个token**。这意味着在预测第 $t$ 个token时，模型只能看到位置 $1, 2, ..., t-1$ 的token，**不能看到未来位置** $t, t+1, ...$ 的信息。

如果不施加因果约束，模型在训练时可以**通过“偷看”未来位置来降低损失**，但推理时未来位置不存在，这将导致**训练与推理的不一致**（称为“信息泄露”）。因此，我们必须在注意力计算中引入**因果掩码（Causal Mask）**。

### 3.2 下三角掩码矩阵

因果掩码是一个**下三角矩阵**（Lower Triangular Matrix）：

$$\text{mask}_{ij} = \begin{cases} 1 & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}$$

其中 **$i$ 是query的位置索引，$j$ 是key的位置索引**。当 $i \geq j$ 时，表示query $i$ **可以关注**到key $j$（即位置 $j$ 是“过去”或“当前”的）；当 $i < j$ 时，表示query $i$ **不能关注**到未来的key $j$。

在代码中，我们通过`torch.tril`生成下三角矩阵：

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

`tril`（Lower Triangle）保留矩阵的**下三角部分**（包括对角线），其余置0。然后转换为**布尔型**，并扩展两个维度以**匹配注意力分数的形状**：`(1, 1, seq_len, seq_len)`，其中第一个`1`是**batch**维度（可广播），第二个`1`是**head**维度（可广播）。

### 3.3 掩码的应用

在`Scaled_dot_product_attention`中，掩码通过`masked_fill`应用：

```python
scores = scores.masked_fill(mask == False, float('-inf'))
```

**所有`False`位置（即上三角，未来位置）被设为`-inf`，softmax后权重为0。**

**注意点**：在实际实现中，`mask`的维度可能包含head维度，也可能不包含，取决于具体用法。代码中采用了**广播机制**，使得 **`(1, 1, seq_len, seq_len)`的掩码可以自动匹配`(batch_size, num_heads, seq_len, seq_len)`的分数张量**。

---

## 四、多头注意力：从单一视角到多视角

### 4.1 单头注意力的局限

**单头注意力（Single-head Attention）** 只能**从一个角度计算query和key的相似度**。但在自然语言中，一个词与上下文的关系往往是**多维度的** —— 例如，“苹果”既可以指水果，也可以指公司，不同头可以分别关注这些不同的语义方面。

**多头注意力（Multi-Head Attention）** 的设计思路是：**使用多组独立的投影矩阵，将 $d_{\text{model}}$ 维的Q、K、V投影到多个子空间，每个子空间分别计算注意力，然后将结果拼接起来。**

数学上：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

其中：

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 4.2 多头的拆分与合并

在代码中，没有为每个头单独创建投影矩阵，而是**用一个大的投影矩阵一次性投影出所有头的Q、K、V，然后用`einops.rearrange`进行维度拆分**。

```python
q = self.wq(x)  # (batch, seq_len, d_model)
k = self.wk(x)
v = self.wv(x)

q = rearrange(q, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
k = rearrange(k, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
v = rearrange(v, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
```

`rearrange` 的表达式 `'b s (n_h d_k) -> b n_h s d_k'` 的含义是：
- 左侧 `'b s (n_h d_k)'`：输入形状为 (batch, seq_len, d_model)，其中最后一维d_model 被解释为 `n_h`(n_heads) 乘以 `d_k`
- 右侧 `'b n_h s d_k'`：输出形状为 (batch, n_heads, seq_len, d_k)

这种写法比手动`view`或`reshape`更清晰、更不易出错。`einops`库让张量维度的变换变得具有**可读性**。

### 4.3 多头注意力的效率优势

多头机制的核心优势在于**并行化**。在单头注意力中，所有计算是顺序的；而在多头中，**所有头可以独立并行计算**，因为它们在`batch`和`head`维度上是**独立**的（注意力的softmax在每个头内部进行，**头与头之间没有数据依赖**）。

这使得多头注意力在GPU上可以充分发挥**并行计算能力**，同时通过**拆分维度**降低了每个头的`d_k`，从而**减少了点积计算的复杂度**（$O(h \cdot d_k^2)$ vs $O((h \cdot d_k)^2)$）。

### 4.4 输出投影

注意力计算完成后，我们得到形状为 `(batch, n_heads, seq_len, d_k)` 的张量，需要将其合并回 `(batch, seq_len, d_model)`：

```python
attn_output = rearrange(attn_output, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)
out = self.wo(attn_output)
```

**最后的 `wo` 是一个线性层，将拼接后的向量投影回 `d_model` 维，作为多头注意力的最终输出。**

---

## 五、RoPE与注意力的融合

### 5.1 位置编码的不同范式

在第二篇博客中，我们介绍了 **RoPE（旋转位置编码）**。它与传统位置编码（如正弦位置编码或可学习位置编码）有一个根本区别：

- **传统方式**：在输入层的词嵌入上**添加**位置向量（加法融合）
- **RoPE方式：在注意力计算的Q和K上施加旋转**（乘法融合）

RoPE的设计使得位置信息**直接嵌入到注意力分数中**，而不是作为**额外的特征向量**。这有两个好处：
1. **位置编码不占用模型的“表达容量”（词嵌入维度不变）**
2. **注意力分数天然地包含了相对位置信息**

### 5.2 在多头注意力中集成RoPE

在`CausalMultiHeadAttention`类中，我们通过`use_rope`参数控制是否启用RoPE：

```python
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_rope=False, max_seq_len=None, theta=10000.0):
        # ...
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len=max_seq_len)
```

在前向传播中，我们在将Q和K拆分为多头之后，对每个头分别应用RoPE：

```python
if self.use_rope:
    assert token_positions is not None
    q_rope = []
    k_rope = []
    for head in range(self.num_heads):
        q_head = q[:, head, :, :]  # (batch, seq_len, d_k)
        k_head = k[:, head, :, :]
        q_head_rope = self.rope(q_head, token_positions)
        k_head_rope = self.rope(k_head, token_positions)
        q_rope.append(q_head_rope.unsqueeze(1))
        k_rope.append(k_head_rope.unsqueeze(1))
    q = torch.cat(q_rope, dim=1)
    k = torch.cat(k_rope, dim=1)
```

- **对每个头分别应用的原因**：RoPE的`d_k`是**每个头的维度**，而整个`d_model`可能很大。每个头拥有**独立的`d_k`维子空间**，**旋转操作在每个头的子空间内独立进行**。
- **只对Q和K施加RoPE的原因：位置信息通过Q·K的点积影响注意力权重，而V只是被权重加权聚合，不需要携带位置信息**。这已被理论和实验证明是有效的。

### 5.3 token_positions参数的传入

在`forward`中，接受`token_positions`参数（形状为 `(batch, seq_len)`），它指定了**每个token在序列中的绝对位置**（从0开始）。这些位置索引用于**从预计算的cos/sin表中取值**。

在训练中，如果我们直接处理**连续的序列片段**，`token_positions`通常就是`0, 1, 2, ..., seq_len-1`。但在推理过程中，我们可能**增量生成token**，此时位置偏移需要正确设置。

### 5.4 因果掩码与RoPE的协同

在带RoPE的因果多头注意力中，因果掩码和RoPE共同工作：
- **RoPE —— 在注意力分数中隐式地编码了相对位置信息**
- **因果掩码 —— 显式地阻止未来位置的注意力**

两者相辅相成：**RoPE告诉模型位置差为d的两个词应该如何交互，而因果掩码强制这种交互只能是单向的（从过去到未来）**。

---

## 六、完整流程回顾

现在，让我们把注意力机制的整个前向传播流程串起来：

```
输入 x: (batch, seq_len, d_model)
    ↓
Q = Linear(x), K = Linear(x), V = Linear(x)  [各 (batch, seq_len, d_model)]
    ↓
rearrange: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
    ↓
（可选）对每个头分别应用RoPE旋转（只作用于Q和K）
    ↓
Scaled_dot_product_attention:
    scores = (Q @ K^T) / sqrt(d_k)     [形状 (batch, n_heads, seq_len, seq_len)]
    scores = mask_fill(mask, -inf)     [因果掩码]
    weights = softmax(scores, dim=-1)
    output = weights @ V               [形状 (batch, n_heads, seq_len, d_k)]
    ↓
rearrange: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
    ↓
out = Linear(output)                   [输出投影]
    ↓
返回 out: (batch, seq_len, d_model)
```

> [!note] 总结
> 本文深入剖析了Transformer注意力机制的每一个核心环节：
> 1. **数值稳定的Softmax**：通过减去最大值防止溢出，这是一个微小但关键的工程技巧
> 2. **缩放点积注意力**：用点积衡量相似性，用缩放控制方差，用softmax转换成权重
> 3. **因果掩码**：用下三角矩阵实现“只看过去”，是语言模型自回归属性的基础
> 4. **多头注意力**：用`einops`优雅地拆分合并，让模型从多个子空间捕捉不同的语义关系
> 5. **RoPE融合**：将位置编码直接注入注意力计算，无需额外向量加法
> 
> 这些组件共同构成了Transformer的“计算核心”。理解注意力机制，就理解了Transformer的精髓所在。