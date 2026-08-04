---
title: Activation & Initialization —— 激活函数与权重初始化
published: 2025-08-06
description: 系统讲解激活函数与权重初始化的协同演进关系。从Sigmoid/Tanh的梯度饱和问题出发，推导ReLU及其变体（LeakyReLU、PReLU、ELU、GELU、Swish）如何解决梯度消失，并深入推导Xavier初始化（适用于Sigmoid/Tanh）和Kaiming初始化（适用于ReLU）的方差守恒数学原理。
cover: "/assets/images/posts/activation_initialization.png"
coverInContent: false
tags: [激活函数, 梯度消失, ReLU, GELU, 权重初始化, 深度学习]
category: Deep_Learning
draft: false
---

# Activation & Initialization —— 激活函数与权重初始化

## 一、引言：从“学什么”到“怎么学得动”

在上一篇博客中，我们确立了**深度学习**的优化目标 —— 通过**最大化似然（MLE）来定义损失函数**，为模型指明了“**学什么**”的方向。然而，一个更现实的问题随之浮现：**有了目标，梯度也计算出来了，但反向传播时，它真的能一路顺畅地传回底层吗？**

本篇博客正是解决这一“**梯度可达性**”问题的关键章节。如果说前两篇构建了数据的“骨架”与目标的“标尺”，那么本篇将赋予模型“可训练”的生命力。我们从Sigmoid与Tanh优雅但致命的**梯度饱和**出发 —— 其导数峰值仅0.25，这意味着10层网络反向传播时梯度会衰减百万倍，这也是深度学习在2010年前后陷入困境的根源。

随后，我们将系统梳理**ReLU及其变体（LeakyReLU、PReLU、ELU、GELU、Swish）** 的演进逻辑。您将看到，ReLU在正区间导数为1的简洁设计，如何彻底打开**梯度**通道，使训练数十层网络成为可能；而**GELU与Swish**如何通过平滑非线性进一步优化**Transformer**等现代架构的训练稳定性。

然而，**激活函数**只是故事的一半。**权重初始化**若与激活函数不匹配，即使ReLU也无法避免梯度爆炸 —— 我们将深入推导**Xavier初始化**（适用于Sigmoid/Tanh的方差守恒）与**Kaiming初始化**（专为ReLU设计的“信号补偿”机制）的数学原理，揭示二者在方差层面的协同关系。

值得注意的是，本篇对**梯度流动**的深刻理解，将直接服务于后续**自动微分与计算图**中关于“**梯度累积**”的实现细节，以及**MLP中反向传播**公式的逐层推导。现在，请带着“**如何让梯度在每一层都稳得住**”的问题进入正文——理解了本篇，您就掌握了让深度学习模型真正“跑起来”的两把核心钥匙。

---

## 二、饱和激活函数：Sigmoid与Tanh的「甜蜜陷阱」

### 2.1 Sigmoid：优雅但致命

Sigmoid函数的定义为：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

值域为 $(0, 1)$，在零点处取最大值 $\sigma(0) = 0.5$。

Sigmoid有一个非常优美的导数公式：

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**证明**：用商法则，

$$
\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2}
$$

而

$$
\sigma(x)(1-\sigma(x)) = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \frac{e^{-x}}{(1+e^{-x})^2}
$$

因此 $\sigma'(x) = \sigma(x)(1-\sigma(x))$。这个公式的优雅之处在于：**只要知道函数值，就能直接算出导数值**。

**然而，致命的缺陷隐藏在这个导数公式中**：

当 $x \to +\infty$ 时，$\sigma(x) \to 1$，$\sigma'(x) \to 0$；
当 $x \to -\infty$ 时，$\sigma(x) \to 0$，$\sigma'(x) \to 0$。

在 $x=0$ 处导数取最大值：

$$
\sigma'(0) = \sigma(0)(1-\sigma(0)) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
$$

**也就是说，Sigmoid的导数最大也只有0.25**。

这意味着什么？假设一个10层的网络，每层都使用Sigmoid激活函数，在最优情况下（每层输入都在0附近），反向传播的梯度每经过一层就乘以0.25。10层之后：

$$
0.25^{10} \approx 9.5 \times 10^{-7}
$$

梯度衰减了**一百万倍**！这就是梯度消失的数学根源。

> **📌 实验数据**：当输入绝对值大于5时，Sigmoid输出已接近饱和区，梯度接近零。在10层以上的网络中，反向传播时梯度会以指数级衰减。

### 2.2 Tanh：零中心的改进，但饱和依旧

Tanh函数的定义为：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

它与Sigmoid的关系为：$\tanh(x) = 2\sigma(2x) - 1$。

Tanh的导数公式为：

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

**Tanh相比Sigmoid的改进**：输出是**零中心（zero-centered）** 的，值域为 $(-1, 1)$。这缓解了Sigmoid非零中心导致的"锯齿形"（zigzag）梯度更新问题。

**但Tanh仍然存在梯度饱和问题**：当输入绝对值较大时，$\tanh(x) \to \pm 1$，导数 $\tanh'(x) \to 0$。虽然Tanh在零点处的导数最大值为1（比Sigmoid的0.25大），但在饱和区的梯度同样趋近于零。

### 2.3 饱和激活函数的三大缺陷总结

| 缺陷 | Sigmoid | Tanh |
|------|---------|------|
| **梯度饱和** | 最大导数仅0.25，极易梯度消失 | 仍存在饱和区，梯度趋近于零 |
| **非零中心** | 输出恒为正，导致zigzag更新 | ✅ 已解决，输出零中心 |
| **计算开销** | 需要exp运算，较慢 | 需要exp运算，较慢 |

---

## 三、ReLU及其变体：非饱和时代的开启

### 3.1 ReLU：简单粗暴的解决方案

ReLU（Rectified Linear Unit）的定义极为简单：

$$
\text{ReLU}(x) = \max(0, x)
$$

其导数为：

$$
\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}
$$

**ReLU的革命性优势**：

1. **正区间无梯度饱和**：当 $x > 0$ 时，导数为1，梯度可以完整地向前传播。这意味着在正区间，**梯度不会衰减**！

2. **计算简单**：不需要指数运算，只是简单的比较和取零。

3. **稀疏激活**：负值被置零，产生稀疏表示，有助于特征选择。

### 3.2 死亡ReLU（Dying ReLU）：ReLU的「阿喀琉斯之踵」

然而，ReLU并非完美。当神经元的输入**持续为负**时，该神经元的输出恒为0，梯度也为0，权重再也无法更新——这个神经元就"死"了。

数学上，若对于某神经元，在训练过程中 $\mathbf{w}^{\top}\mathbf{x} + b < 0$ 始终成立，则：

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \text{ReLU}} \cdot \text{ReLU}'(\mathbf{w}^{\top}\mathbf{x}+b) = 0
$$

权重永远无法更新。

**死亡ReLU的严重后果**：在深度超过20层的网络中，未经特殊处理的ReLU可能导致30%-50%的神经元死亡，显著降低模型的有效容量。

### 3.3 ReLU变体：给负区间留一条「活路」

#### LeakyReLU

LeakyReLU在负区间引入一个小的斜率 $\alpha$（通常取0.01）：

$$
\text{LeakyReLU}(x) = \begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases}
$$

导数为：

$$
\text{LeakyReLU}'(x) = \begin{cases} 1, & x > 0 \\ \alpha, & x \leq 0 \end{cases}
$$

负区间梯度不再是0，神经元不会"死亡"。实验表明，$\alpha=0.01$ 时神经元死亡率可降低至5%以下。

#### PReLU（Parametric ReLU）

PReLU将负区间的斜率 $\alpha$ 变为**可学习的参数**：

$$
f(x) = \max(x, \alpha x)
$$

其中 $\alpha$ 通过反向传播与权重一起更新。在ImageNet分类任务中，PReLU较ReLU可提升约1.2%的Top-1准确率。

#### ELU（Exponential Linear Unit）

ELU在负区间使用指数函数：

$$
\text{ELU}(x) = \begin{cases} x, & x > 0 \\ \alpha(e^x - 1), & x \leq 0 \end{cases}
$$

ELU具有ReLU的优势，且输出均值接近零，同时对噪声有一定鲁棒性。但需要计算exp，计算量稍大。

#### GELU（Gaussian Error Linear Unit）

GELU是Transformer等现代架构的**首选激活函数**。其定义为：

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

实际计算中常用近似公式：

$$
\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

GELU的核心思想是：**根据输入的大小"软性"地决定激活程度**——输入值大时接近完全激活，输入值小时部分激活，负值大时接近零但不完全为零。

GELU的平滑性和非零曲率提供了更丰富的梯度信息，使深度网络的训练更加稳定。

#### Swish

Swish由Google Brain团队通过自动搜索发现，定义为：

$$
\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}
$$

其中 $\beta$ 可以是常数（通常为1）或可训练参数。

Swish被称为**"自门控"（self-gated）激活函数**：输入 $x$ 乘以其自身的Sigmoid函数 $\sigma(x)$，Sigmoid充当一个"门"，控制有多少输入能够通过。

当 $x$ 很大时，$\sigma(x) \approx 1$，Swish $\approx x$（近似线性）；
当 $x$ 很小时，$\sigma(x) \approx 0$，Swish $\approx 0$（但梯度非零）。

Swish在负区间保持非零梯度，同时保留正区间的线性特性，避免了死亡ReLU问题。

### 3.4 激活函数对比总结

| 激活函数 | 梯度饱和 | 零中心 | 计算开销 | 死亡神经元 | 适用场景 |
|---------|---------|--------|---------|-----------|---------|
| Sigmoid | ✅ 严重 | ❌ | 高 | - | 二分类输出层 |
| Tanh | ✅ 存在 | ✅ | 高 | - | 传统RNN |
| ReLU | ❌ 正区间无 | ❌ | 低 | ✅ 有 | CNN通用 |
| LeakyReLU | ❌ | ❌ | 低 | ⚠️ 极少 | 防止死亡ReLU |
| PReLU | ❌ | ❌ | 低 | ⚠️ 极少 | 可学习负斜率 |
| ELU | ❌ | ✅ | 中 | ❌ | 需零中心输出 |
| GELU | ❌ | ❌ | 中 | ❌ | Transformer |
| Swish | ❌ | ❌ | 中 | ❌ | 深层网络 |

---

## 四、权重初始化：让网络赢在「起跑线」

选择合适的激活函数只是解决梯度问题的一半。**即使使用了ReLU，如果权重初始化不当，梯度仍然可能爆炸或消失**。

### 4.1 为什么初始化如此重要？

考虑一个全连接层：$z = \sum_{i=1}^{n} w_i x_i$。

如果权重 $w_i$ 初始值太大，$z$ 会很大，可能将激活函数推入饱和区（对Sigmoid/Tanh而言）。

如果权重 $w_i$ 初始值太小，$z$ 会很小，梯度在反向传播时会逐层衰减。

**理想情况是：每一层输出的方差保持稳定，既不放大也不衰减**。

### 4.2 Xavier初始化（Glorot初始化）

Xavier初始化由Glorot & Bengio在2010年提出，主要适用于 **Sigmoid和Tanh** 等饱和激活函数。

#### 核心假设

1. 权重 $w_i$ 独立同分布，均值为0，方差为 $\text{Var}(w)$
2. 输入 $x_i$ 独立同分布，均值为0，方差为 $\text{Var}(x)$
3. **激活函数在0附近近似线性**（即 $f(z) \approx z$）

#### 前向传播的方差约束

对于线性层 $z = \sum_{i=1}^{n_{in}} w_i x_i$，根据方差的性质（独立变量和的方差等于方差之和）：

$$
\text{Var}(z) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

为了保持方差一致（$\text{Var}(z) = \text{Var}(x)$），需要：

$$
n_{in} \cdot \text{Var}(w) = 1 \quad \Longrightarrow \quad \text{Var}(w) = \frac{1}{n_{in}}
$$

#### 反向传播的方差约束

反向传播时，梯度 $\frac{\partial L}{\partial x_i}$ 与权重的转置相关：

$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^{n_{out}} w_{ij} \cdot \frac{\partial L}{\partial z_j}
$$

同理，为保持梯度方差一致，需要：

$$
n_{out} \cdot \text{Var}(w) = 1 \quad \Longrightarrow \quad \text{Var}(w) = \frac{1}{n_{out}}
$$

#### 最终的方差选择

前向和反向传播对方差的要求不同（$1/n_{in}$ vs $1/n_{out}$），Xavier取两者的**调和平均**：

$$
\text{Var}(w) = \frac{2}{n_{in} + n_{out}}
$$

#### 实现方式

**正态分布**：$w \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$

**均匀分布**：若 $w \sim U(-a, a)$，则 $\text{Var}(w) = \frac{a^2}{3}$。令其等于 $\frac{2}{n_{in}+n_{out}}$，得：

$$
a = \sqrt{\frac{6}{n_{in} + n_{out}}}
$$

即 $w \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$

### 4.3 Kaiming初始化（He初始化）

Xavier初始化在ReLU上表现不佳，因为**ReLU会将大约一半的神经元输出置为零，相当于砍掉了一半的信号**。

何恺明在2015年的论文《Delving Deep into Rectifiers》中提出了专门针对ReLU的Kaiming初始化。

#### 核心洞察

ReLU的负值全部被截断为零，导致**输出的方差大约只有输入方差的一半**。为了补偿这个损失，权重的方差需要**翻倍**。

#### 前向传播的推导

设第 $l$ 层的输入为 $x$（已经过ReLU激活），输出为 $y = \sum_{i=1}^{n} w_i x_i$（线性变换后，尚未激活）。

由于 $x$ 是ReLU的输出，$x \geq 0$ 且均值为0（假设权重均值为0，输入在0附近对称分布）。

关键步骤：**ReLU将输入方差减半**

对于零均值的对称分布 $u$，经过ReLU后：

$$
\mathbb{E}[x^2] = \mathbb{E}[(\text{ReLU}(u))^2] = \frac{1}{2}\mathbb{E}[u^2]
$$

即 $\text{Var}(x) = \frac{1}{2}\text{Var}(u)$。

因此：

$$
\text{Var}(y) = n \cdot \text{Var}(w) \cdot \text{Var}(x) = n \cdot \text{Var}(w) \cdot \frac{1}{2}\text{Var}(u)
$$

为了保持方差一致（$\text{Var}(y) = \text{Var}(u)$）：

$$
n \cdot \text{Var}(w) \cdot \frac{1}{2} = 1 \quad \Longrightarrow \quad \text{Var}(w) = \frac{2}{n}
$$

#### 最终结果

Kaiming初始化的权重方差为：

$$
\text{Var}(w) = \frac{2}{n_{in}}
$$

**正态分布**：$w \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$

**均匀分布**：$w \sim U\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$

> **💡 关键区别**：Xavier用的是 $\frac{2}{n_{in}+n_{out}}$，Kaiming用的是 $\frac{2}{n_{in}}$。对于ReLU，Kaiming初始化保证每层方差守恒。

### 4.4 Python实现

```python
import numpy as np

def xavier_uniform(input_dim, output_dim):
    """Xavier均匀初始化，适用于Sigmoid/Tanh"""
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, (output_dim, input_dim))

def xavier_normal(input_dim, output_dim):
    """Xavier正态初始化，适用于Sigmoid/Tanh"""
    std = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, std, (output_dim, input_dim))

def kaiming_uniform(input_dim, output_dim):
    """Kaiming均匀初始化，适用于ReLU"""
    limit = np.sqrt(6 / input_dim)
    return np.random.uniform(-limit, limit, (output_dim, input_dim))

def kaiming_normal(input_dim, output_dim):
    """Kaiming正态初始化，适用于ReLU"""
    std = np.sqrt(2 / input_dim)
    return np.random.normal(0, std, (output_dim, input_dim))

# 示例：创建一个3层MLP的权重
d, h1, h2, q = 784, 256, 128, 10

# 使用Kaiming初始化（ReLU网络）
W1 = kaiming_normal(d, h1)   # 方差 = 2/784 ≈ 0.00255
W2 = kaiming_normal(h1, h2)  # 方差 = 2/256 ≈ 0.00781
W3 = kaiming_normal(h2, q)   # 方差 = 2/128 ≈ 0.01563
```

### 4.5 初始化方法选择指南

| 激活函数 | 推荐初始化 | 方差公式 |
|---------|-----------|---------|
| Sigmoid | Xavier | $2/(n_{in}+n_{out})$ |
| Tanh | Xavier | $2/(n_{in}+n_{out})$ |
| ReLU | Kaiming | $2/n_{in}$ |
| LeakyReLU | Kaiming（Leaky版） | $2/(1+\alpha^2)n_{in}$ |
| GELU | 近似Kaiming | $2/n_{in}$ |
| Swish | 近似Kaiming | $2/n_{in}$ |

---

## 五、总结：激活函数与初始化的「协同进化」

回顾整个发展历程，我们可以看到一条清晰的脉络：

**第一代（Sigmoid）** ：提出了可微的非线性激活，但梯度饱和严重，最大导数仅0.25，深层网络几乎无法训练。

**第二代（Tanh）** ：解决了零中心问题，但饱和问题依然存在。

**第三代（ReLU）** ：正区间导数为1，彻底解决了正区间的梯度消失问题，使训练深层网络成为可能。但引入了死亡ReLU的新问题。

**第四代（LeakyReLU/PReLU/ELU/GELU/Swish）** ：在保留ReLU优势的同时，通过给负区间赋予非零梯度来解决死亡ReLU问题。

而**权重初始化**与激活函数**相辅相成**：

- **Xavier初始化**为Sigmoid/Tanh提供了方差守恒的保障
- **Kaiming初始化**为ReLU家族补偿了"砍掉一半信号"的损失

**一个经验法则**：

> 使用ReLU及其变体 → 用**Kaiming初始化**
> 使用Sigmoid或Tanh → 用**Xavier初始化**

理解这些原理，你就掌握了深度学习训练稳定性的两个关键杠杆。下一篇文章，我们将深入探讨**卷积神经网络（CNN）** 的数学原理与反向传播。

---

延伸阅读:
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)（Kaiming初始化原始论文）
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)（Xavier初始化原始论文）
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)（Swish原始论文）
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)（GELU原始论文）