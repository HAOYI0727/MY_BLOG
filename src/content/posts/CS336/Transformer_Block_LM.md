---
title: TransformerBlock & TransformerLM —— 完整语言模型
published: 2026-03-08
description: 系统讲解如何将Transformer的各个组件组合成完整的可训练语言模型：从TransformerBlock的Pre-Norm与Post-Norm架构对比入手，剖析残差连接如何保障深层网络梯度流动；深入模块化设计中的权重加载接口；拆解TransformerLM四大核心组件（Token Embedding → N层TransformerBlock → Final RMSNorm → LM Head）及其超参数设计的权衡考量；完整跟踪从token ID到logits的前向传播数据流（形状变换全程）。
cover: "/assets/images/posts/transformer_block_lm.png"
coverInContent: false
tags: [Transformer, Pre-Norm, 残差连接, 超参数设计, 前向传播, LLM基础]
category: CS336
draft: false
---

# TransformerBlock & TransformerLM —— 完整语言模型

## 引言

经过前三篇博客的积累，我们已经完成了从分词到基础算子、再到注意力机制的全部构建工作。现在，我们可以**将所有组件精妙地组合在一起**，构建出一个**完整的、可训练的语言模型**。

如果说**注意力机制**是Transformer的“心脏”，那么**TransformerBlock**就是它的“器官”，而整个**TransformerLM**则是完整的“生命体”。本篇博客将展示这些组件如何协同工作，形成**从输入token ID到输出logits**的完整数据流。

我们将重点讨论以下几个设计维度：
- **归一化的位置**：**Pre-Norm vs Post-Norm**，现代LLM为何几乎全部转向Pre-Norm
- **模块的参数传递**：如何设计支持外部**预训练权重加载**的灵活接口
- **模型超参数**：从`vocab_size`到`d_ff`，每个数字背后的设计考量
- **权重初始化**：稳定训练的第一道防线
- **完整前向传播**：从token ID到logits的每一步数据变换

--- 

## 一、TransformerBlock：残差连接的优雅封装

### 1.1 Transformer块的内部结构

一个标准的Transformer块由两个子层组成：
1. **多头自注意力子层（MHA, Multi-Head Self-Attention）**
2. **前馈网络子层（FFN, Feed-Forward Network）**

每个子层都遵循“**残差连接 + 归一化**”的模式。在代码中，`TransformerBlock` 将这两个子层封装在一起，并提供了丰富的配置选项，支持多种**消融实验**（如是否使用RMSNorm、是否使用RoPE、是否使用SwiGLU、归一化位置等）。

### 1.2 Pre-Norm vs Post-Norm

**归一化的位置 —— Pre-Norm vs Post-Norm 是决定训练稳定性的关键选择**，是Transformer架构设计中最关键的决策之一。

**Post-Norm**（原始Transformer论文中的设计）：

**每个子层的输出先经过残差连接，再进行归一化**：
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Pre-Norm**（现代LLM的标配）：

**每个子层先进行归一化，再经过子层，最后与输入相加（残差连接）**：
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

两者在数据流上的差异看似微小，但对训练的稳定性和最终性能有着深远影响。

**Pre-Norm更优的原因**：
1. **梯度流动更顺畅**：在Pre-Norm中，**残差连接直接将输入传递到输出**，梯度可以无阻碍地通过**残差路径反向传播**。而在Post-Norm中，梯度必须经过**归一化层的导数**，这可能导致**梯度衰减**。
2. **训练初期更稳定**：Post-Norm在深层网络中容易导致**梯度爆炸或消失**，尤其当层数较多时。Pre-Norm的梯度范数在整个网络中**更加均匀**，允许使用更大的**学习率**。
3. **实证优势**：几乎所有现代大语言模型（GPT-3、Llama、PaLM等）都采用Pre-Norm。实验表明，**在相同深度下，Pre-Norm比Post-Norm达到更低的训练损失，且收敛更快**。

在代码中，`TransformerBlock`通过`norm_position`参数支持两种模式：

```python
if self.norm_position == 'pre':
    # Pre-Norm: 先归一化，再子层，最后残差
    x1 = self.norm1(in_features)
    x1 = self.causal_multi_head_attention(x1, token_positions)
    x1 = x1 + in_features
    
    x2 = self.norm2(x1)
    x2 = self.ffn(x2)
    out = x2 + x1
else:  # 'post'
    # Post-Norm: 先子层，残差，再归一化
    x1 = self.causal_multi_head_attention(in_features, token_positions)
    x1 = self.norm1(x1 + in_features)
    
    x2 = self.ffn(x1)
    out = self.norm2(x2 + x1)
```

### 1.3 残差连接的本质

**残差连接**（Residual Connection）是深度学习中的一项革命性发明，它通过**将输入直接加到输出上**，解决了深层网络中的**梯度消失**问题。

在Transformer块中，两个子层都使用了**残差连接**。数学上，Pre-Norm模式可以表示为：$$\text{out} = x + \text{FFN}(\text{LN}(x + \text{Attention}(\text{LN}(x))))$$

**残差连接允许梯度在反向传播时绕过子层的非线性变换，直接流向前层**，这使得训练非常深的网络（数十层甚至上百层）成为可能。如果没有残差连接，Transformer的训练将**极不稳定**。

### 1.4 位置信息的传递

在`TransformerBlock`的`forward`中生成了**`token_positions`张量**：

```python
token_positions = torch.arange(seq_len, device=in_features.device)
token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
```

这个张量传递给**注意力模块**（当`use_rope=True`时）。注意，**位置信息不是作为独立的嵌入添加的，而是通过RoPE直接注入到Q和K的计算中**——这正是RoPE的设计哲学。

---

## 二、模块的参数传递

### 2.1 设计动机

在大规模模型训练中，我们经常需要**从外部加载预训练权重**——无论是为了**微调**（Fine-tuning）、**断点续训**，还是为了进行**模型融合**。因此，模块必须支持**在初始化时通过参数传入预训练权重**。

代码中的`TransformerBlock`和`TransformerLM`都设计了丰富的权重参数，允许调用者**传入各个子模块的权重张量**。

### 2.2 TransformerBlock的权重加载

在`TransformerBlock`的`__init__`中接受10个**权重参数** —— `attn_q_proj_weight`, `attn_k_proj_weight`, `attn_v_proj_weight`, `attn_o_proj_weight`,`ln1_weight`, `ln2_weight`,`ffn_w1_weight`, `ffn_w2_weight`, `ffn_w3_weight`

每个权重如果非`None`，则直接赋值给对应模块的`weight.data`。例如：
```python
if attn_q_proj_weight is not None:
    self.causal_multi_head_attention.wq.weight.data = attn_q_proj_weight.data
```

这种设计使得`TransformerBlock`可以**独立于其父模块被初始化并加载权重**，实现了**模块级别的参数复用**。

### 2.3 TransformerLM的权重字典

在`TransformerLM`中，我们使用一个统一的字典`weights`来**传递所有层的预训练权重**。

键名遵循层次化的命名约定，例如：`"layers.0.attn.q_proj.weight"`表示第0层的Q投影权重；`"layers.5.ln1.weight"`表示第5层的第一个归一化层权重；`"token_embeddings.weight"`表示词嵌入矩阵；`"lm_head.weight"`表示输出投影权重；`"ln_final.weight"`表示最终归一化权重

在初始化`TransformerBlock`时，从`weights`字典中**按需提取对应层的权重**：
```python
transformer_block = TransformerBlock(
    # ...
    attn_q_proj_weight=weights.get(f"layers.{layer}.attn.q_proj.weight"),
    # ...
)
```

这种字典方式的好处是**解耦了权重存储和模型定义**——可以**从任意格式的检查点文件中加载权重**（如PyTorch的`.pt`、Hugging Face的`.bin`等），只要将其转换为**字典格式**即可。

### 2.4 权重加载的时机

代码中的权重加载发生在 **`__init__`阶段**，这意味着权重是在**模块创建**时就被赋值的。这要求外部传入的权重张量具有**正确的形状**（与模块定义匹配）。若形状不匹配，PyTorch会抛出异常。

在实际工程中，我们通常**先构建模型（随机初始化），然后从检查点文件加载权重**。代码中直接在`__init__`中加载是一种简化，旨在展示接口设计，实际使用中也可以采用`load_state_dict`的方式。

---

## 三、TransformerLM的整体架构

### 3.1 四大核心组件

完整的Transformer语言模型由四个主要部分串联而成：

1. **词嵌入层 (Token Embedding)：将token ID映射为稠密向量 `(batch, seq_len) → (batch, seq_len, d_model)`**
2. **TransformerBlock (N个)：逐层处理，每层都进行多头自注意力MHA与前馈网络FFN，保持维度不变**
3. **归一化层 (RMSNorm)：在输出前进行最后一次归一化（Pre-Norm架构中保留）**
4. **线性层 (LM Head)：将`d_model`维映射回`vocab_size`维，得到logits**

代码中的`TransformerLM`正是按此结构构建的：

```python
self.embedding_module = EmbeddingModule(vocab_size, d_model, device)

self.transformer_blocks = nn.ModuleList()
for _ in range(n_layers):
    self.transformer_blocks.append(TransformerBlock(...))

if use_rmsnorm:
    self.final_norm = RMSNorm(d_model, eps=1e-5)
else:
    self.final_norm = nn.Identity()

self.lm_head = LinearModule(d_model, vocab_size)
```

### 3.2 超参数的设计考量

每个超参数的取值都影响着模型的容量、速度和内存占用，这些超参数相互耦合，共同决定了模型的总参数量：

- **`vocab_size`（词表大小）** —— 由**分词器**决定，通常为32K～100K。**影响嵌入层和输出层的参数量：`vocab_size × d_model × 2`**；词表越大，模型可表示的token越多，但嵌入层参数量也线性增加
- **`context_length`（上下文长度）** —— 模型能处理的**最大序列长度**。**影响位置编码的预计算范围、注意力矩阵的复杂度（$O(L^2)$）**；更大的上下文能捕获更长的依赖，但计算和内存开销随长度平方增长
- **`d_model`（模型维度）** —— 所有层中向量的**基本维度**，决定模型的“宽度”，直接影响**参数量和计算量**。典型值：GPT-2 small为768，GPT-3为12288，Llama 70B为8192
- **`n_layers`（层数）** —— 模型的“**深度**”，更多层能捕捉更抽象的特征，但层数增加会线性**增加计算量和延迟**
- **`n_heads`（注意力头数）** —— 决定**多头注意力的并行子空间数**。需要满足 **`d_model % n_heads == 0`**，每个头的维度 **`d_k = d_model / n_heads`**，更多的头能让模型关注更多不同的方面，但**增加计算量**
- **`d_ff`（前馈网络隐藏维度）** —— 在SwiGLU中，通常设为 **`(8/3) * d_model`** 并取整到64的倍数，以**匹配标准FFN的参数量**。更大的`d_ff`增强了FFN的表达能力，但增加了参数和计算
- **`theta`（RoPE底数）** —— **控制旋转频率的分布**，通常取10000。较大的theta使低频旋转更慢，有利于**长距离依赖**

---

## 四、权重初始化策略

### 4.1 初始化的意义

神经网络的权重初始化直接决定了训练的起始状态。初始化不当可能导致：
- **梯度消失**：激活值过小，梯度传递不到浅层
- **梯度爆炸**：激活值过大，梯度溢出为NaN
- **对称性问题**：所有神经元学习相同特征

良好的初始化应保持**各层输出的方差一致**，使得信号可以在网络中稳定传播。

### 4.2 初始化代码

在`TransformerLM`中，我们提供了一个`_init_weights`方法，该方法遍历所有模块，**对线性层和嵌入层使用均值为0、标准差为0.02的正态分布初始化**：

```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, LinearModule):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, EmbeddingModule):
            torch.nn.init.normal_(module.embedding_matrix, mean=0.0, std=0.02)
```

**使用0.02的标准差的原因** —— 这是GPT-2论文中采用的初始化策略，经过实验验证在中等规模模型上表现良好。对于更大的模型，可能需要**更小的标准差（如0.006）配合更大的学习率**。

### 4.3 各模块的专用初始化

实际上，各个基础模块已经包含了**各自的初始化逻辑**：
- `LinearModule`使用**截断正态分布**，标准差为`sqrt(2/(din+dout))`（mishmash风格）
- `EmbeddingModule`使用标准差为1的**截断正态分布**
- `RMSNorm`初始化为**全1**
- `SwiGLU`中的线性层同样使用`LinearModule`的初始化

因此，`_init_weights`中的通用初始化**可能会覆盖**这些专用初始化，这取决于调用顺序。在实际工程中，通常选择一种统一的初始化策略，并在模块定义时设置好，避免多次初始化。

---

## 五、前向传播的数据流

**输入：token ID张量** —— 假设有一个batch大小为`B`，序列长度为`S`的输入：

```python
in_indices: torch.Tensor of shape (B, S) # 每个元素是token ID，范围 [0, vocab_size-1]
```

**Step 1: Token Embedding** —— 每个token ID被替换为其对应的 **`d_model`维嵌入向量**。这一步的参数数量为 **`vocab_size × d_model`**。
```python
x = self.embedding_module(in_indices)  # (B, S) -> (B, S, d_model)
```

**Step 2: 逐层通过TransformerBlock**

对于每一层 `l` —— `x = self.transformer_blocks[l](x)`

在每个TransformerBlock内部，数据流如下：

**1. 子层1（多头自注意力MHA）**：
```python
x_norm1 = RMSNorm(x)                       # (B, S, d_model)
x_attn = CausalMultiHeadAttention(x_norm1) # (B, S, d_model)
x = x + x_attn                            # 残差连接
```

**2. 子层2（前馈网络FFN）**：
```python
x_norm2 = RMSNorm(x)                       # (B, S, d_model)
x_ffn = SwiGLU(x_norm2)                    # (B, S, d_model)
x = x + x_ffn                             # 残差连接
```

整个过程中，序列长度`S`和模型维度`d_model`都**保持不变**。

**Step 3: 最终归一化**
```python
x = self.final_norm(x)  # (B, S, d_model)
```
在Pre-Norm架构中，最后一层输出后通常还会进行一次归一化，以**确保输入到LM Head的分布是稳定的**。

**Step 4: LM Head（输出投影）**
```python
logits = self.lm_head(x)  # (B, S, d_model) -> (B, S, vocab_size)
```
线性层将每个位置的**`d_model`维向量映射为`vocab_size`维的logits**。每个logit表示对应token的**未归一化得分**。

**输出：logits张量** —— 最终输出logits的形状为 **`(B, S, vocab_size)`**。这个张量可以直接输入到 **`CrossEntropyLoss`**（内部会应用`log_softmax`）计算训练损失，或者在推理时应用 **`softmax`** 得到**下一个token的概率分布**。

> [!note] 可视化数据流
> ```
>      Token IDs (B, S)
>              ↓
>  Embedding (B, S, d_model)
>              ↓
> ┌─── TransformerBlock 0 ───┐
> │  RMSNorm → Attention → + │
> │  RMSNorm → SwiGLU → +    │
> └──────────────────────────┘
>              ↓ (B, S, d_model)
> ┌─── TransformerBlock 1 ───┐
> │  ...                     │
> └──────────────────────────┘
>              ↓ (B, S, d_model)
>             ... (重复 N 次)
>              ↓  
>  Final RMSNorm (B, S, d_model)
>              ↓
> LM Head (Linear) (B, S, vocab_size)
>              ↓
>    Logits (B, S, vocab_size)
> ```

---

## 六、模块化设计的作用

### 6.1 依赖关系清晰

从第一篇到第四篇，我们逐步构建了一个层次化的模块体系：

```
BPE Tokenizer (外部工具，非PyTorch模块)
    ↓
EmbeddingModule, LinearModule, RMSNorm, SwiGLU, RoPE (基础算子)
    ↓
CausalMultiHeadAttention (注意力机制)
    ↓
TransformerBlock (组合注意力 + FFN + 残差 + 归一化)
    ↓
TransformerLM (组合嵌入 + 多层Block + 输出层)
```

每一层都依赖于其下层，但**不依赖于上层**，这使得我们可以独立测试和替换任何一个模块。

### 6.2 支持丰富的消融实验

代码中通过多个布尔/枚举参数支持消融实验：
- `use_rmsnorm`：对比RMSNorm vs 无归一化
- `norm_position`：对比Pre-Norm vs Post-Norm
- `use_rope`：对比RoPE vs 无位置编码
- `use_swiglu`：对比SwiGLU vs FFNSiLU

这些参数贯穿整个模型栈，从`TransformerLM`一路传递到各个基础模块，使得我们可以在不修改核心代码的情况下，轻松切换不同的设计选择。

### 6.3 与标准实现的对比

与Hugging Face的GPT2Model相比，我们的实现：
- **更简洁**：没有历史遗留的兼容性代码
- **更透明**：所有模块**从零构建**，不依赖黑盒封装
- **更灵活**：支持多种消**融实验**配置

当然，这些实现尚未包含一些工程优化（如Flash Attention、kv-cache等），但这些优化可以无缝集成到现有模块中。


> [!note] 总结
> 本篇博客是系列中的里程碑 —— 我们将前三篇的所有**组件**汇聚在一起，构建了一个**完整的、可训练的Transformer语言模型**。重点讨论了：
> 
> 1. **TransformerBlock**：**Pre-Norm vs Post-Norm**的选择，**残差连接**的妙处，以及**位置信息**的传递方式
> 2. **参数传递设计**：如何通过字典方式**灵活加载预训练权重**，实现模块级别的**参数复用**
> 3. **整体架构**：四大核心组件及其**超参数**的设计考量，每个数字背后的权衡
> 4. **权重初始化**：从专用初始化到通用策略，确保训练起始阶段的**稳定性**
> 5. **前向传播数据流**：**从token ID到logits的完整路径**，每一步的形状变换
> 
> 至此，我们已经完成了模型构建的全部工作。后续我们将进入训练环节——如何准备数据、如何计算损失、如何优化参数，最终让这个模型真正“学会”语言。