---
title: Residual Network —— ResNet残差网络
published: 2025-08-16
description: 系统讲解ResNet残差网络的设计哲学与数学原理：从退化问题的本质出发，推导残差块F(x)+x如何通过恒等映射直通路径缓解梯度消失，详解Bottleneck块如何将参数量减少约94%，并对比ResNet（加法融合）与DenseNet（拼接融合）的梯度流动差异。
cover: "/assets/images/posts/resnet.png"
coverInContent: false
tags: [ResNet, 残差网络, Bottleneck, 梯度消失, 深度学习]
category: Deep_Learning
draft: false
---

# Residual Network —— ResNet残差网络

## 一、引言：当“更深”不再意味着“更好”——残差连接的诞生

在上一篇博客中，我们系统掌握了**卷积、感受野与池化**等空间特征提取的核心工具。凭借这些操作，VGG将网络推至19层，GoogLeNet达到22层——一个自然而然的信念随之形成：**网络越深，表达能力越强，性能理当越好**。

然而，2015年一个令人不安的实验结果打破了这一直觉：在CIFAR-10上，一个56层的“纯”卷积网络，其训练误差竟显著高于20层的网络。这不是**过拟合**（训练误差高而非低），而是**退化问题** —— 深层网络在优化层面遇到了“天花板”：梯度可以借助**Batch Normalization**正常流动，但求解器却难以将一堆非线性层训练成“什么都不做”的恒等映射。

本篇博客正是破解这一困局的关键章节。我们将从**退化问题的本质**出发，揭示为何让网络学习 $H(x)=x$ 如此困难，而引入**残差块** $F(x)+x$ 后，目标便转换为学习 $F(x)=0$ —— 这一看似微小的结构变化，却使152层的ResNet在ImageNet上超越人类水平。随后，我们将从**链式法则**出发，严格证明**残差连接如何通过恒等映射的直通路径缓解梯度消失** —— 深层特征由连乘变为连加，梯度中始终包含一个无衰减的“1”。

在此基础上，我们将详解**Bottleneck块**如何通过1×1→3×3→1×1的结构将参数量减少约94%，使千层网络在**有限显存**下成为可能；并对比ResNet（**加法融合**）与DenseNet（**拼接融合**）在**梯度流动与特征复用**上的本质差异。

值得注意的是，本篇对“**跨层直连**”的设计哲学，将直接服务于后续**RNN与BPTT**——当时间步长达数百时，正是残差思想启发了**LSTM中的门控机制**。现在，请带着“**如何让梯度无损穿越深度**”的疑问进入正文——理解了ResNet，您就掌握了现代所有深度架构（从Transformer到Diffusion Model）得以存在的底层基石。

---

## 二、退化问题：深度网络为何“越深越差”？

### 2.1 退化 vs 梯度消失：两个不同的问题

在深入ResNet之前，有必要区分两个经常被混淆的概念：

| 问题 | 表现 | 根本原因 |
|------|------|---------|
| **梯度消失/爆炸** | 网络无法收敛（loss不下降） | 反向传播时梯度逐层相乘，趋于0或∞ |
| **退化问题** | 网络能收敛，但更深时准确率下降 | 深层网络难以优化，求解器无法找到最优解 |

Batch Normalization（BN）的提出解决了梯度消失/爆炸问题，使得数十层的网络能够收敛。但BN**并未解决退化问题**——即使网络能够收敛，56层的plain network仍然比20层差。

### 2.2 退化问题的本质：为什么恒等映射这么难学？

考虑一个浅层网络已经达到了不错的性能。现在我们在这个网络上**追加若干层**，构成一个更深的网络。

从理论上讲，如果追加的这些层**什么也不做**（即实现恒等映射 $y=x$），那么深层网络的性能至少不会比浅层差。

但问题在于：**让一个由多个非线性层（卷积+激活）组成的模块实现恒等映射，恰恰是神经网络最难做的事情之一**。

为什么呢？

一个典型的卷积块包含：卷积 → BN → ReLU → 卷积 → BN。要让这个模块的输出等于输入，需要精确地调整所有卷积核的权重，使得两层卷积的复合效果恰好是恒等映射。这相当于求解一个高度非线性的方程组——在随机初始化的情况下，几乎不可能通过梯度下降恰好收敛到这样的解。

**ResNet的洞见**：与其让网络学习 $H(x)=x$，不如让网络学习 $F(x)=H(x)-x$，然后通过 $H(x)=F(x)+x$ 来构造输出。

这样，**实现恒等映射只需要让 $F(x)=0$** ——让所有卷积层的输出为0，比让它们精确地实现恒等映射要容易得多！

---

## 三、残差块（Residual Block）：$F(x)+x$ 的数学原理

### 3.1 残差块的结构

一个基本的残差块可以表示为：

$$
\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

其中：
- $\mathbf{x}$ 是残差块的输入
- $F(\mathbf{x}, \{W_i\})$ 是残差函数（由若干卷积层组成）
- $\mathbf{y}$ 是残差块的输出

对于包含两层卷积的Basic Block：

$$
F(\mathbf{x}) = W_2 \cdot \sigma(W_1 \cdot \mathbf{x})
$$

其中 $\sigma$ 是ReLU激活函数。

整个残差块的前向传播为：

$$
\mathbf{y} = W_2 \cdot \sigma(W_1 \cdot \mathbf{x}) + \mathbf{x}
$$

### 3.2 为什么残差学习更容易？一个直观例子

假设我们要将输入 $x=5$ 映射到目标输出 $H(x)=5.1$。

- **普通网络**：需要学习 $F'(x)=5.1$，即权重需要将5映射到5.1——变化幅度为2%
- **残差网络**：$F(x)=H(x)-x=0.1$，只需要学习将5映射到0.1——变化幅度为98%

如果目标从5.1变为5.2：
- 普通网络：输出从5.1变到5.2，变化2%
- 残差网络：残差从0.1变到0.2，变化**100%**

**残差结构对输出的变化更敏感**，这使得梯度更新时权重的调整幅度更大，学习效率更高。

### 3.3 残差块如何缓解梯度消失：数学推导

这是理解ResNet最核心的部分。让我们从数学上证明残差连接如何缓解梯度消失。

**普通网络**（无残差连接）：

假设第 $l$ 层的输出为 $\mathbf{x}_{l+1} = F_l(\mathbf{x}_l)$，其中 $F_l$ 是第 $l$ 层的非线性变换。

根据链式法则，损失 $L$ 对第 $l$ 层输入的梯度为：

$$
\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \cdot \prod_{i=l}^{L-1} \frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i} = \frac{\partial L}{\partial \mathbf{x}_L} \cdot \prod_{i=l}^{L-1} \frac{\partial F_i}{\partial \mathbf{x}_i}
$$

如果每一层的雅可比矩阵 $\partial F_i/\partial \mathbf{x}_i$ 的谱范数都小于1（在饱和激活函数下很容易发生），那么连乘的结果会**指数级衰减**到0——这就是梯度消失。

**残差网络**：

残差块的前向传播为：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\mathbf{x}_l)
$$

从第 $l$ 层到第 $L$ 层（$L > l$）：

$$
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} F_i(\mathbf{x}_i)
$$

**这是关键**：在普通网络中，深层特征是浅层特征的**连乘**；而在残差网络中，深层特征是浅层特征的**连加**！

现在计算梯度：

$$
\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \cdot \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \cdot \left( 1 + \sum_{i=l}^{L-1} \frac{\partial F_i}{\partial \mathbf{x}_l} \right)
$$

**梯度由两项组成**：

1. **$1$（来自恒等映射的直通路径）** ：梯度可以直接无损地传播
2. **$\sum \partial F_i/\partial \mathbf{x}_l$（来自残差分支）** ：即使这一项很小，有 **$1$ 的存在保证了梯度不会消失**

> **🔑 核心结论**：残差连接为梯度提供了一条“**高速公路**”——无论网络有多深，梯度至少有一条路径可以**直接**从输出层流到输入层，而不经过任何带参数的层。

这就是为什么ResNet可以训练到**1000层以上**，而普通网络在30层左右就已经无法训练了。

### 3.4 ResNet v2：让恒等映射更“纯净”

何恺明等人在后续论文《Identity Mappings in Deep Residual Networks》中对残差块进行了改进。

**ResNet v1**（原始版本）：

$$
\mathbf{x}_{l+1} = f(\mathbf{x}_l + F(\mathbf{x}_l))
$$

其中 $f$ 是ReLU激活函数（**后激活**）。

**ResNet v2**（改进版本）：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + F(f(\mathbf{x}_l))
$$

将BN和ReLU移到**残差分支内部**，恒等映射路径上**没有任何操作**（**预激活**）。

**为什么这样更好？**

在v1中，恒等映射路径上还有一个ReLU（$f$），这破坏了“纯净”的恒等映射。在v2中，恒等映射路径**完全无参数、无激活**，信号可以**真正无损**地传播。

实验表明，ResNet v2的训练速度更快，泛化性能更好。

---

## 四、瓶颈块（Bottleneck）：1×1-3×3-1×1的设计艺术

### 4.1 为什么需要Bottleneck？

Basic Block（两个3×3卷积）在ResNet-18和ResNet-34中表现良好。但当网络加深到50层以上时，Basic Block的**参数量会急剧膨胀**。

以ResNet-50为例：如果全部使用Basic Block，参数量将远超硬件承载能力。

**Bottleneck Block**应运而生，用于**深层网络**（ResNet-50/101/152）。

### 4.2 Bottleneck的结构

Bottleneck Block采用**“1×1 → 3×3 → 1×1”** 的三层结构：

1. **1×1卷积（降维）** ：将通道数压缩（如256→64），大幅减少计算量
2. **3×3卷积（特征提取）** ：在低维空间进行空间特征提取
3. **1×1卷积（升维）** ：将通道数恢复（如64→256）

### 4.3 参数量的数学对比

以输入通道256、输出通道256为例：

**Basic Block**（两个3×3卷积）：

$$
\text{参数量} = 256 \times 3 \times 3 \times 256 + 256 \times 3 \times 3 \times 256 = 1,179,648
$$

**Bottleneck Block**（1×1→3×3→1×1）：

$$
\begin{aligned}
\text{参数量} &= 256 \times 1 \times 1 \times 64 & \text{(1×1降维)} \\
&+ 64 \times 3 \times 3 \times 64 & \text{(3×3特征提取)} \\
&+ 64 \times 1 \times 1 \times 256 & \text{(1×1升维)} \\
&= 16,384 + 36,864 + 16,384 = 69,632
\end{aligned}
$$

**Bottleneck的参数量仅为Basic Block的 $\frac{69,632}{1,179,648} \approx 5.9\%$** ！

> **💡 1×1卷积的本质**：1×1卷积在每个像素位置上对**所有通道进行线性组合**，相当于一个**跨通道的全连接层**。它不关心空间信息，只负责通道维度的信息整合。

### 4.4 Bottleneck的工程实现

```python
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck Block
    适用于 ResNet-50/101/152
    """
    expansion = 4  # 输出通道数 = 输入通道数 * 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1x1 降维：将通道数压缩到 out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 空间特征提取
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 升维：恢复到 out_channels * expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut路径：如果维度不匹配，用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # 恒等映射路径
        
        # 残差路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # 残差 + 恒等映射
        out += identity
        out = self.relu(out)
        return out
```

> **📌 设计细节**：
> - `expansion=4` 表示输出通道是输入通道的4倍（如64→256）
> - 当 `stride=2` 时，特征图尺寸减半，shortcut路径也需要用步长为2的1×1卷积来匹配尺寸
> - 所有卷积层 `bias=False`，因为BN层已经包含了偏置项

---

## 五、DenseNet vs ResNet：加法 vs 拼接

### 5.1 DenseNet的核心思想

DenseNet（密集连接卷积网络）由黄高等人于2016年提出，获得了CVPR 2017最佳论文奖。

DenseNet的基本思路与ResNet一致——**建立跨层连接**来改善梯度流动。但实现方式截然不同：

| | ResNet | DenseNet |
|---|---|---|
| **连接方式** | **加法**（element-wise addition） | **拼接**（concatenation） |
| **信息传递** | 只连接前一层的输出 | 连接**前面所有层**的输出 |
| **特征复用** | 隐式（通过残差学习） | 显式（所有层共享特征） |

DenseNet中，第 $l$ 层的输入是**前面所有层**输出的拼接：

$$
\mathbf{x}_l = \text{Concat}(\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1})
$$

### 5.2 DenseNet为什么更“窄”？

在ResNet中，每一层都会产生新的特征图，网络宽度（通道数）会逐渐增加。

在DenseNet中，由于每一层都能接收到前面所有层的特征，**不需要每一层都学习很多特征图**。DenseNet将每一层设计得很“窄”——通常每层只学习 **$k=12$ 或 $32$ 个特征图**（称为**增长率（growth rate）** ）。

这种设计使得DenseNet在**参数效率**上优于ResNet。

### 5.3 梯度流动的对比

**ResNet的梯度路径**：
- 通过恒等映射的**加法**路径直接传播
- 梯度 = 1 + 残差梯度

**DenseNet的梯度路径**：
- 每一层都能**直接**接收到来自后面所有层的梯度信号
- 靠近输入层的梯度不会因为层层传递而消失

从梯度传播的角度看，**DenseNet比ResNet更激进**——它建立了 $O(L^2)$ 条连接（$L$ 为层数），确保每一层都能直接与所有后续层通信。

### 5.4 DenseNet的优缺点

| 优点 | 缺点 |
|------|------|
| 缓解梯度消失 | 显存占用大（需要保存所有中间特征图） |
| 特征重用，参数效率高 | 前向计算量较大 |
| 减轻过拟合 | 实现复杂度高于ResNet |

### 5.5 DenseBlock的PyTorch实现

```python
class DenseLayer(nn.Module):
    """DenseNet的单个密集层"""
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # 先通过1x1卷积降维（Bottleneck设计）
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, 
                               padding=1, bias=False)
    
    def forward(self, x):
        # 注意：这里用拼接，不是加法！
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)  # 在通道维度上拼接

class DenseBlock(nn.Module):
    """Dense Block：包含多个Dense Layer"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## 六、总结

| 概念 | 核心思想 | 关键公式 |
|------|---------|---------|
| **退化问题** | 深层网络难以优化，不是过拟合 | 56层训练误差 > 20层 |
| **残差块** | 让网络学习 $F(x)=H(x)-x$ 而非 $H(x)$ | $H(x)=F(x)+x$ |
| **梯度缓解** | 恒等映射提供梯度直通路径 | $\partial L/\partial \mathbf{x}_l = \partial L/\partial \mathbf{x}_L \cdot (1 + \sum \partial F_i/\partial \mathbf{x}_l)$ |
| **Bottleneck** | 1×1降维→3×3提取→1×1升维 | 参数量减少约94% |
| **ResNet v2** | 恒等映射路径纯净无操作 | 预激活（Pre-activation） |
| **DenseNet** | 拼接而非加法，连接所有前层 | $\mathbf{x}_l = \text{Concat}(\mathbf{x}_0, \ldots, \mathbf{x}_{l-1})$ |

**ResNet的遗产远不止于计算机视觉**。从GPT到BERT，从Transformer到Diffusion Model，**残差连接**已经成为了几乎所有现代深度架构的**基础组件**。可以说，没有残差连接，就没有今天的大模型时代。

理解ResNet，不仅是在理解一个卷积网络架构，更是在理解**深度学习如何克服了“深度”本身带来的诅咒**。

---

延伸阅读：
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（ResNet原始论文）
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)（ResNet v2）
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)（DenseNet原始论文）
- [Why ResNet Works? Residuals Generalize](https://arxiv.org/abs/2010.05467)（残差网络的泛化理论分析）