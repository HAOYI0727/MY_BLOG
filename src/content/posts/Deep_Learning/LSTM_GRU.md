---
title: LSTM & GRU —— 门控循环神经网络
published: 2025-08-20
description: 系统讲解LSTM与GRU的门控机制设计哲学与数学原理。从LSTM的“双轨”架构（细胞状态+隐藏状态）与三门结构（遗忘门、输入门、输出门）出发，推导细胞状态更新公式如何通过按元素加法路径缓解梯度消失，并深入剖析GRU的两大简化及其参数量优势。附带讨论双向LSTM与深层堆叠LSTM的应用场景。
cover: "/assets/images/posts/lstm_gru.png"
coverInContent: false
tags: [LSTM, GRU, 门控机制, 梯度消失, 深度学习]
category: Deep_Learning
draft: false
---

# LSTM & GRU —— 门控循环神经网络

## 一、引言：从“梯度牢笼”到“记忆高速”——门控机制如何解放循环网络

在上一篇博客中，我们直面了循环神经网络最致命的**数学困境：$\partial \mathbf{h}_t/\partial \mathbf{h}_k$ 是一个从 $k+1$ 到 $t$ 的矩阵连乘**——当谱范数小于1时，**梯度指数级衰减**，10步前的信息几乎归零；当大于1时，**梯度指数级爆炸**，训练瞬间崩溃。这一“**连乘诅咒**”将RNN的有效记忆范围锁死在10-20个时间步以内，使其在面对**长文本、长语音或长时间序列**时力不从心。

本篇博客正是破解这一“**记忆牢笼**”的关键章节。我们从LSTM的**双轨架构**——**细胞状态**（$c_t$，长时记忆“传送带”）与**隐藏状态**（$h_t$，短时输出）出发，揭示其核心设计哲学：**将信息传递路径从矩阵乘法改为按元素加法**。您将看到，细胞状态更新公式 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ 中，$\partial c_t/\partial c_{t-1} = \text{diag}(f_t)$ 是**对角矩阵而非满矩阵**——这意味着梯度不再经受矩阵连乘的指数级衰减，而是可以**沿细胞状态的“高速公路”几乎无损地流回早期时间步**，其代价仅是**由遗忘门 $f_t$ 控制的逐元素缩放**。

随后，我们将深入剖析**GRU**的两大简化 —— **三门变两门（重置门+更新门）、双状态变单状态** —— 及其约25%的参数量缩减如何在实际训练中带来**更快的收敛速度与相近的性能表现**，并讨论**双向LSTM**与**深层堆叠LSTM**在自然语言处理、语音识别等任务中的典型应用场景。

值得注意的是，本篇是“**时序架构**”的收官之篇——至此，我们完整走过了从**数据表示（张量）→优化目标（概率论）→梯度保真（激活+初始化）→梯度计算（自动微分）→完整训练闭环（MLP+优化器）→空间建模（CNN）→深度跨越（ResNet）→时序建模（RNN/BPTT）→门控记忆（LSTM/GRU）** 的全部脉络。您已拥有构建、训练并理解现代深度学习模型的全栈底层认知。现在，请带着“**如何让网络记住100步前的信息**”的疑问进入正文——理解了LSTM与GRU，您就掌握了序列建模的终极钥匙，也为理解**Transformer的自注意力机制**铺平了道路。

---

## 二、LSTM：从“单轨”到“双轨”的架构革命

### 2.1 核心思想：给记忆修一条“高速公路”

常规RNN的问题是它内部状态的更新方式是“粗暴”的——每一步的新信息都会与旧信息无差别地混合。LSTM的设计哲学是**赋予网络自行决定信息取舍的能力**。

与RNN只有一个隐藏状态 $h_t$ 在时间步之间传递不同，LSTM引入了**两个独立的状态向量**在时间轴上并行传递：

1. **细胞状态（Cell State, $c_t$）** ：这是LSTM的核心，原始论文中称之为 **“恒定误差旋转木马”（Constant Error Carousel, CEC）** 。可以把它想象成一条“信息高速公路”或“传送带”，负责在整个序列中传递**长期记忆**。

2. **隐藏状态（Hidden State, $h_t$）** ：与RNN中的隐藏状态类似，代表了当前时间步的**短期记忆**和**最终输出**。

> **💡 关键洞察**：在普通RNN中，信息在时间步之间传递必须经过**矩阵乘法**（$\mathbf{W}_{hh}\mathbf{h}_{t-1}$）。而在LSTM中，细胞状态的传递路径是**按元素的加法和乘法**（$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$），**没有额外的矩阵连乘**，信息可以直接在这条传送带上流动。

### 2.2 三个门：信息流动的“智能开关”

LSTM中引入了**3个门**，即**输入门（input gate）、遗忘门（forget gate）和输出门（output gate）** 。

LSTM中的“门”是一种让信息选择性通过的结构，设计灵感来源于数字电路中的逻辑门。它的实现非常简单：**一个以Sigmoid为激活函数的全连接层**，输入通常是当前时间步的输入 $x_t$ 和上一个时间步的隐藏状态 $h_{t-1}$ 的拼接向量。

Sigmoid函数将元素值映射到 **(0, 1)** 区间内：
- 输出接近 **1** → “允许”对应维度的信息**完全通过**
- 输出接近 **0** → “阻止”对应维度的信息通过，即“遗忘”或“忽略”它

**三个门的分工如下**：

| 门 | 符号 | 作用 |
|---|---|---|
| **遗忘门** | $f_t$ | 决定是否让上一时刻学到的信息通过或部分通过 |
| **输入门** | $i_t$ | 计算出候选值，决定哪些新信息写入细胞状态 |
| **输出门** | $o_t$ | 决定哪些信息输出到隐藏状态 |

### 2.3 LSTM的完整数学公式

假设隐藏单元个数为 $h$，给定时间步 $t$ 的小批量输入 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ 和上一时间步隐藏状态 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$。

**Step 1：三个门的计算**

$$
\mathbf{I}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i)
$$

$$
\mathbf{F}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f)
$$

$$
\mathbf{O}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o)
$$

其中 $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ 和 $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$ 是权重参数，$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$ 是偏差参数。

**Step 2：候选记忆细胞**

$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c)
$$

这里使用值域在 $[-1, 1]$ 的 $\tanh$ 函数作为激活函数。

**Step 3：细胞状态更新（核心！）** 

$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t
$$

其中 $\odot$ 表示按元素乘法。

这个公式是LSTM的灵魂：
- $\mathbf{F}_t \odot \mathbf{C}_{t-1}$：**遗忘**旧信息（遗忘门控制保留多少）
- $\mathbf{I}_t \odot \tilde{\mathbf{C}}_t$：**写入**新信息（输入门控制写入多少）

**Step 4：隐藏状态计算**

$$
\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
$$

输出门控制从细胞状态中读取多少信息到隐藏状态。

### 2.4 为什么LSTM能缓解梯度消失？——数学证明

这是理解LSTM最核心的部分。让我们从梯度传播的角度来看。

在普通RNN中，隐藏状态的更新是：

$$
\mathbf{h}_t = \phi(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)
$$

梯度传播的关键项是：

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi'(\mathbf{h}_{t-1}))
$$

这是**矩阵乘法**——谱范数决定了梯度是指数衰减还是爆炸。

而在LSTM中，细胞状态的更新是：

$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t
$$

**关键差异**：$\partial \mathbf{C}_t / \partial \mathbf{C}_{t-1}$ 是什么？

$$
\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \text{diag}(\mathbf{F}_t)
$$

**这是一个对角矩阵，而不是满矩阵！** 

这意味着：
1. **没有矩阵乘法**：梯度在细胞状态路径上的传播是**逐元素的**，不存在谱范数导致的全局指数衰减
2. **遗忘门可以学习**：如果网络需要长期记忆，遗忘门 $\mathbf{F}_t$ 可以学习为接近1的值，让梯度几乎无损地通过
3. **加法路径**：信息可以直接在细胞状态的“传送带”上流动，仅经过按元素的加权与相加

> **🔑 核心结论**：LSTM通过将信息传递路径从**矩阵乘法**改为**按元素加法**，从根本上改变了梯度传播的数学性质。梯度不再需要经过一连串的矩阵相乘，而是可以通过细胞状态的“高速公路”直接流回早期时间步。

从另一个角度看，门控机制也是为了**解决权重冲突问题**——**输入门**保护细胞状态不受无关**输入**的干扰，**输出门**则保护其他单元不受当前细胞状态中无关**记忆**的干扰。

---

## 三、GRU：LSTM的“精简版”

### 3.1 为什么需要GRU？

LSTM成功解决了长时依赖问题，但代价是**三个门 + 一个细胞状态**，结构复杂、参数众多。GRU（门控循环单元）由Cho等人于2014年提出，是LSTM的一个**更简单的变体**。

GRU的设计目标很明确：**在保持LSTM性能的同时，减少参数数量和计算复杂度**。

### 3.2 GRU的两大简化

**简化一：三门→两门**

GRU将LSTM中的三个门（遗忘门、输入门、输出门）**合并为两个门**——**重置门（Reset Gate）和更新门（Update Gate）** 。

具体来说，GRU把LSTM的**输入门和遗忘门组合在一起**，少了一个门。更新门 $z$ 的角色相当于LSTM里的遗忘门，而 $1-z$ 相当于LSTM中的输入门。

**简化二：双状态→单状态**

LSTM有两个状态向量在时间轴上传递——细胞状态 $c_t$（长期记忆）和隐藏状态 $h_t$（短期记忆）。

GRU**将细胞状态和隐藏状态合并**，只传递一个隐藏状态 $h_t$。在GRU里，$h_t$ 的角色比较像LSTM中的 $c_t$，可以保留得比较久。

> **💡 设计哲学**：GRU中遗忘门和输入门是**联动的**——如果有新的信息进来，才会忘掉之前的信息；如果没有新信息进来，就不会忘记信息。这个逻辑比LSTM的独立三门更简洁。

### 3.3 GRU的完整数学公式

**Step 1：重置门和更新门**

$$
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r)
$$

$$
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z)
$$

其中 $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ 和 $\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ 是权重参数。

**Step 2：候选隐藏状态**

$$
\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{hh} + \mathbf{b}_h)
$$

重置门 $\mathbf{R}_t$ 控制着过去信息的丢弃程度：
- 当重置门的值接近 **0** 时，意味着对应的隐藏状态元素将被重置为0，从而**丢弃上一时间步的历史信息**
- 当接近 **1** 时，表示**保留**上一时间步的隐藏状态

**Step 3：最终隐藏状态**

$$
\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t
$$

最终的隐藏状态是**候选隐藏状态**和**前一隐藏状态**的加权组合，权重由更新门控制：
- 当更新门接近 **1** 时，新状态几乎完全**继承过去状态**
- 当接近 **0** 时，新状态主要**由候选状态决定**

### 3.4 GRU vs LSTM：参数量的定量对比

LSTM的参数由3个门 + 1个候选细胞状态组成，每个都需要独立的权重矩阵：

$$
\text{LSTM参数量} = 4 \times (d \times h + h \times h + h)
$$

GRU只有2个门 + 1个候选隐藏状态：

$$
\text{GRU参数量} = 3 \times (d \times h + h \times h + h)
$$

**GRU的参数量约为LSTM的 $\frac{3}{4}$** 。

> **📌 实际表现**：在很多时候，人们更愿意使用GRU来替换LSTM，因为GRU比LSTM少一个门，参数更少，**相对容易训练且可以防止过拟合**（尤其是在训练样本少的时候）。而且，GRU的性能和LSTM几乎一样。

不过需要注意的是，虽然GRU参数更少，但由于重置门的计算中并行性较低，某些情况下LSTM的执行时间反而更短。

---

## 四、双向LSTM与深层堆叠

### 4.1 双向LSTM：同时看到“过去”和“未来”

标准的LSTM是**单向**的——信息只能从过去流向未来。但在很多任务中（如机器翻译、文本分类），**未来的上下文同样重要**。

**双向LSTM（Bidirectional LSTM）** 使用两个独立的LSTM层：
- **前向LSTM**：按时间正序处理序列（从 $t=1$ 到 $t=T$）
- **后向LSTM**：按时间逆序处理序列（从 $t=T$ 到 $t=1$）

然后将两个方向的隐藏状态**拼接**起来作为最终的表示。

> **💡 直观理解**：就像我们在做阅读理解时，不仅看前面的词，也会看后面的词来确定当前词的含义。双向LSTM让模型同时拥有了“回顾过去”和“展望未来”的能力。

**典型应用场景**：
- **自然语言处理**：情感分析、命名实体识别、机器翻译
- **语音识别**：利用前后音素信息提高识别准确率
- **蛋白质结构预测**：利用序列上下文信息

### 4.2 深层堆叠LSTM：增加网络的“深度”

**堆叠LSTM（Stacked LSTM / Deep LSTM）** 将多个LSTM层**垂直堆叠**在一起：

- 第1层LSTM接收原始输入序列
- 第2层LSTM接收第1层的输出作为输入
- 依此类推...

每一层LSTM都在**不同的时间抽象层次**上学习特征：
- **底层**：捕捉局部的、短期的模式
- **高层**：捕捉全局的、长期的依赖关系

> **📌 实践建议**：堆叠的层数足够大时，多层RNN的效果可能会比单层好。但堆叠层数增加会带来**更高的计算负荷**，且需要更多数据来避免过拟合。

### 4.3 PyTorch实现

```python
import torch
import torch.nn as nn

# ============ LSTM ============
# 单层LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# 双层堆叠LSTM
stacked_lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# 双向LSTM
bidirectional_lstm = nn.LSTM(
    input_size=10, 
    hidden_size=20, 
    num_layers=2, 
    bidirectional=True,  # 开启双向
    batch_first=True
)

# ============ GRU ============
# 单层GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# 双层堆叠GRU
stacked_gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# 双向GRU
bidirectional_gru = nn.GRU(
    input_size=10, 
    hidden_size=20, 
    num_layers=2, 
    bidirectional=True,
    batch_first=True
)

# ============ 前向传播示例 ============
batch_size, seq_len, input_size = 32, 50, 10
x = torch.randn(batch_size, seq_len, input_size)

# 双向双层LSTM
output, (h_n, c_n) = bidirectional_lstm(x)
# output: (batch_size, seq_len, hidden_size * 2)  # 双向 → 2倍
# h_n: (num_layers * 2, batch_size, hidden_size)
# c_n: (num_layers * 2, batch_size, hidden_size)

print(f"输出形状: {output.shape}")  # (32, 50, 40)
print(f"最终隐藏状态形状: {h_n.shape}")  # (4, 32, 20)
```

---

## 五、总结

| 特性 | 标准RNN | LSTM | GRU |
|------|---------|------|-----|
| **状态数量** | 1个（$h_t$） | 2个（$c_t, h_t$） | 1个（$h_t$） |
| **门控数量** | 0 | 3（输入/遗忘/输出） | 2（重置/更新） |
| **参数量** | 基准 | ~4倍于RNN | ~3倍于RNN（LSTM的3/4） |
| **梯度消失** | 严重 | ✅ 极大缓解 | ✅ 极大缓解 |
| **长时依赖** | 差 | ✅ 优秀 | ✅ 优秀 |
| **计算效率** | 高 | 低 | 中等 |
| **适用场景** | 短序列 | 长序列、复杂任务 | 长序列、资源受限 |

**选择建议**：
- **数据量大、任务复杂、计算资源充足** → 选择 **LSTM**
- **数据量适中、需要快速迭代、资源有限** → 选择 **GRU**
- **需要利用未来上下文信息** → 使用 **双向LSTM/GRU**
- **需要建模多层次的时间抽象** → 使用 **堆叠LSTM/GRU**

LSTM和GRU的门控机制，是深度学习历史上最重要的架构创新之一。它们不仅让循环神经网络真正具备了处理长序列的能力，其设计哲学——**用可学习的“门”来控制信息流动**——也深刻地影响了后来的Transformer、扩散模型等现代架构。

---

延伸阅读：
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)（Hochreiter & Schmidhuber, 1997）
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)（GRU原始论文，Cho et al., 2014）
- [Dive into Deep Learning - LSTM](https://d2l.ai/chapter_recurrent-neural-networks/lstm.html)
- [Dive into Deep Learning - GRU](https://d2l.ai/chapter_recurrent-neural-networks/gru.html)