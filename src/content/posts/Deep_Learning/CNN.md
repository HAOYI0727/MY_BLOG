---
title: CNN —— 卷积神经网络
published: 2025-08-14
description: 系统讲解卷积神经网络（CNN）的四大核心操作：从严格卷积与互相关的数学区别出发，推导感受野的递推计算公式，剖析空洞卷积如何在不增加参数的前提下扩大感受野，详解Max Pooling与Average Pooling的反向传播机制，以及1×1卷积的跨通道信息整合、降维/升维与增加非线性的三大作用。
cover: "/assets/images/posts/cnn.png"
coverInContent: false
tags: [CNN, 卷积神经网络, 感受野, 反向传播, 深度学习]
category: Deep_Learning
draft: false
---

# CNN —— 卷积神经网络

## 一、引言：为什么CNN能“看懂”图像？

在上一篇文章中，我们讨论了激活函数与权重初始化如何让深度网络“站得稳”。而真正让深度学习在计算机视觉领域大放异彩的，是**卷积神经网络（CNN）** 的核心设计——**局部连接**、**权重共享**和**平移不变性**。

全连接层（MLP）处理图像时，每个神经元都要连接到输入的所有像素。对于一张 $224 \times 224$ 的RGB图像，仅输入层就有 $224 \times 224 \times 3 = 150,528$ 个神经元，如果连接到1000个隐藏层神经元，参数数量就高达1.5亿。这不仅仅是计算量大，更重要的是**全连接层忽略了图像的空间结构**——相邻像素之间的关系远比相距很远的像素更重要。

CNN通过三个核心机制解决了这个问题：
1. **局部连接**：每个神经元只关注输入的一个局部区域
2. **权重共享**：同一个卷积核滑过整个图像，参数大幅减少
3. **空间层次化**：浅层提取边缘等低级特征，深层组合出高级语义

本文将从**卷积运算的数学本质**出发，深入探讨感受野、空洞卷积、池化层反向传播和1×1卷积等CNN核心操作。

---

## 二、卷积运算：严格数学定义 vs 深度学习实现

### 2.1 严格数学定义：卷积需要“翻转”

在信号处理和数学分析中，两个函数 $f$ 和 $g$ 的卷积定义为：

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
$$

对于离散的二维信号（如图像），卷积定义为：

$$
y_{ij} = \sum_{u=1}^{U} \sum_{v=1}^{V} w_{uv} \cdot x_{i-u+1, j-v+1}
$$

注意下标中的 **$i-u+1$ 和 $j-v+1$** ——这意味着卷积核 $w$ 在与输入 $x$ 计算之前，**要先进行水平和垂直方向的翻转（180°旋转）** 。

### 2.2 深度学习实现：实际上是“互相关”

然而，在深度学习框架（PyTorch、TensorFlow等）中，卷积层实际执行的是**互相关（cross-correlation）运算**：

$$
y_{ij} = \sum_{u=1}^{U} \sum_{v=1}^{V} w_{uv} \cdot x_{i+u-1, j+v-1}
$$

互相关与严格卷积的**唯一区别**在于：**卷积核不需要翻转**，直接与输入进行滑动点积。

> **💡 核心洞察**：深度学习框架中所谓的“卷积层”，实际上是互相关运算。之所以这样称呼，是因为：
> 1. **计算效率更高**：省去了翻转操作，减少了不必要的计算开销
> 2. **特征提取能力等价**：卷积核的参数是通过训练学习的，如果网络需要翻转后的核，它可以直接学到翻转后的权重
> 3. **历史惯例**：早期研究沿用了“卷积”这一术语，并成为行业惯例

如果权重矩阵是对称的（很多情况下确实如此），那么卷积和互相关在数值上完全相同。

### 2.3 二维互相关的数学表达

一个完整的卷积层（含多个卷积核）的计算可以表示为：

$$
\mathbf{Z}^p = \sum_{d=1}^{D} \mathbf{W}^{p,d} \otimes \mathbf{X}^d + b^p, \quad \mathbf{Y}^p = f(\mathbf{Z}^p)
$$

其中：
- $\mathbf{X}^d$ 是第 $d$ 个输入通道的特征图
- $\mathbf{W}^{p,d}$ 是第 $p$ 个卷积核的第 $d$ 个通道
- $\otimes$ 表示互相关（滑动点积）运算
- $b^p$ 是偏置项
- $f$ 是激活函数

### 2.4 输出尺寸的计算

对于一个输入尺寸为 $I \times I$、卷积核大小为 $K \times K$、填充（padding）为 $P$、步长（stride）为 $S$ 的卷积层，输出尺寸为：

$$
O = \frac{I - K + 2P}{S} + 1
$$

```python
import numpy as np

def conv2d_naive(X, W, b, stride=1, padding=0):
    """
    二维互相关运算的朴素实现（教学用）
    X: (H_in, W_in) 单通道输入
    W: (K_h, K_w) 卷积核
    b: 标量偏置
    """
    h_in, w_in = X.shape
    k_h, k_w = W.shape
    
    # 填充
    if padding > 0:
        X_pad = np.pad(X, ((padding, padding), (padding, padding)), mode='constant')
    else:
        X_pad = X
    
    # 计算输出尺寸
    h_out = (h_in - k_h + 2 * padding) // stride + 1
    w_out = (w_in - k_w + 2 * padding) // stride + 1
    
    Y = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            # 提取感受野区域
            x_region = X_pad[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            # 逐元素乘积累加（这就是互相关）
            Y[i, j] = np.sum(x_region * W) + b
    return Y

# 示例
X = np.random.randn(5, 5)   # 5x5 输入
W = np.random.randn(3, 3)   # 3x3 卷积核
Y = conv2d_naive(X, W, b=0, stride=1, padding=0)
print(f"输入尺寸: {X.shape}, 输出尺寸: {Y.shape}")  # (3, 3)
```

---

## 三、感受野（Receptive Field）：神经元“看到”了多大区域？

### 3.1 什么是感受野？

感受野（Receptive Field）指的是**卷积神经网络中某一层输出特征图上的一个元素，在原始输入图像上对应的区域大小**。

**直观理解**：越深层的神经元，看到的输入区域越大。两个 $3 \times 3$ 卷积层堆叠（stride=1），第二层的一个神经元能看到第一层 $3 \times 3$ 的区域，而这个区域又对应原始输入的 $5 \times 5$ 区域。

### 3.2 感受野的递推计算公式

从顶层向底层递推计算感受野：

$$
RF_i = RF_{i+1} + (K_i - 1) \times S_{i+1}
$$

其中：
- $RF_i$ 是第 $i$ 层（更靠近输入）的感受野大小
- $RF_{i+1}$ 是第 $i+1$ 层（更靠近输出）的感受野大小
- $K_i$ 是第 $i$ 层的卷积核（或池化核）尺寸
- $S_{i+1}$ 是第 $i+1$ 层到第 $i$ 层之间的**累积步长**

**从输出层开始**：最顶层的感受野为 $RF_{top} = 1$（输出层的一个像素对应自己），然后逐层向下计算。

### 3.3 实例：为什么VGG用两个3×3替代一个5×5？

VGG网络的一个重要设计原则是：**用多个小卷积核堆叠替代大卷积核**。

验证堆叠两个 $3 \times 3$ 卷积核（stride=1）的感受野：

- 顶层（输出层）：$RF = 1$
- 经过第二个 $3 \times 3$ 卷积：$RF = 1 + (3-1) \times 1 = 3$
- 经过第一个 $3 \times 3$ 卷积：$RF = 3 + (3-1) \times 1 = 5$

**结论**：两个 $3 \times 3$ 卷积的感受野等于一个 $5 \times 5$ 卷积。

**参数量的优势**：假设输入和输出通道数均为 $C$：
- 一个 $5 \times 5$ 卷积的参数：$5 \times 5 \times C \times C = 25C^2$
- 两个 $3 \times 3$ 卷积的参数：$2 \times (3 \times 3 \times C \times C) = 18C^2$

参数量减少了 **28%**，同时还增加了**非线性**（两层激活函数）。

```python
def compute_receptive_field(layers):
    """
    计算感受野
    layers: list of (kernel_size, stride) 从输出层到输入层
    """
    rf = 1
    for k, s in layers:
        rf = rf + (k - 1) * s
    return rf

# VGG风格：两个3x3 vs 一个5x5
print(f"两个3x3 (stride=1) 感受野: {compute_receptive_field([(3,1), (3,1)])}")  # 5
print(f"一个5x5 (stride=1) 感受野: {compute_receptive_field([(5,1)])}")          # 5
```

---

## 四、空洞卷积（Dilated Convolution）：不增加参数，扩大感受野

### 4.1 为什么要用空洞卷积？

在语义分割等**密集预测**任务中，我们需要：
1. **大感受野**：捕捉全局上下文信息
2. **高分辨率特征图**：精确定位每个像素

传统方法通过**下采样（池化或步长卷积）** 扩大感受野，但这会降低特征图分辨率，丢失空间细节。

**空洞卷积**（又称扩张卷积，Atrous/Dilated Convolution）通过**在卷积核元素之间插入空洞**，在**不增加参数数量**的前提下扩大感受野。

### 4.2 空洞卷积的数学定义

空洞卷积引入了一个新参数——**扩张率（dilation rate）** $r$：

- $r=1$：标准卷积（无空洞）
- $r=2$：在卷积核元素之间插入1个空洞（间隔为1）
- $r=n$：元素间隔为 $n-1$ 个零值

**有效卷积核尺寸**（感受野）为：

$$
K' = K + (K - 1) \times (r - 1)
$$

例如：$3 \times 3$ 卷积核，$r=2$ 时，有效尺寸为 $3 + (3-1) \times (2-1) = 5$，感受野从 $3 \times 3$ 扩大到 $5 \times 5$。$r=3$ 时，感受野扩大到 $7 \times 7$。

### 4.3 空洞卷积的优势与应用

**核心优势**：
1. **指数级扩大感受野**：堆叠多层空洞卷积，感受野可以指数增长
2. **不增加参数量**：卷积核大小不变，只是采样位置变了
3. **保持空间分辨率**：不需要下采样

**典型应用**：
- **DeepLab系列**：使用空洞卷积进行语义分割
- **ASPP（Atrous Spatial Pyramid Pooling）** ：使用不同扩张率的并行空洞卷积捕获多尺度上下文信息

### 4.4 空洞卷积的“Gridding效应”与解决方案

空洞卷积也有**缺点**：

> **Gridding效应（网格效应）** ：由于卷积核是稀疏采样的，相邻的输出像素来自相互独立的输入子集，彼此之间缺乏相关性，导致局部信息丢失。

**解决方案**：
- **HDC（Hybrid Dilated Convolution）** ：在一个组内使用**递增的扩张率**（如 $r=1,2,3$），而不是固定的扩张率
- **contract_dilation**：在ResNet中设置不同阶段的扩张率，使卷积核更密集地覆盖输入特征图

```python
import torch.nn as nn

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super().__init__()
        # dilation参数控制空洞间隔
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, 
            dilation=dilation_rate,  # 扩张率
            padding=dilation_rate     # 保持输出尺寸不变
        )
    
    def forward(self, x):
        return self.conv(x)

# 不同扩张率的空洞卷积
r1 = DilatedConvBlock(64, 64, dilation_rate=1)  # 标准卷积，感受野3x3
r2 = DilatedConvBlock(64, 64, dilation_rate=2)  # 感受野5x5
r3 = DilatedConvBlock(64, 64, dilation_rate=3)  # 感受野7x7
```

---

## 五、池化层（Pooling）的反向传播

### 5.1 池化层的特点：没有参数，但有梯度

池化层（Pooling Layer）是CNN中一个**特殊的存在**：

- **没有可学习的参数**：不需要更新权重
- **改变特征图尺寸**：$2 \times 2$ 池化将尺寸减半
- **需要反向传播**：虽然没有参数，但梯度需要传递给前一层

池化层反向传播的核心原则是：

> **梯度总和守恒**：池化前 $N$ 个像素的梯度之和，必须等于池化后1个像素的梯度。

### 5.2 Max Pooling的反向传播

Max Pooling在前向传播中，**只保留池化窗口中的最大值**，丢弃其他值。

反向传播时：
- **最大值所在位置的神经元**：接收完整的梯度
- **其他位置的神经元**：梯度为0

为了在反向传播中知道哪个位置是最大值，前向传播时**必须记录最大值的位置索引（max_id / argmax）** 。

**数学表达**：

$$
\frac{\partial L}{\partial x_{ij}} = 
\begin{cases}
\frac{\partial L}{\partial y_{pq}}, & \text{如果 } x_{ij} \text{ 是池化窗口中的最大值} \\
0, & \text{否则}
\end{cases}
$$

### 5.3 Average Pooling的反向传播

Average Pooling在前向传播中，**计算池化窗口内所有值的平均值**。

反向传播时：
- **梯度平均分配**：将1个梯度等分为 $n$ 份，分配给池化窗口中的 $n$ 个像素

> **⚠️ 常见错误**：不能简单地把梯度复制 $n$ 份传回去，否则梯度总和会变为原来的 $n$ 倍，造成梯度爆炸。

**数学表达**（池化窗口大小为 $k \times k$）：

$$
\frac{\partial L}{\partial x_{ij}} = \frac{1}{k^2} \cdot \frac{\partial L}{\partial y_{pq}}, \quad \forall (i,j) \in \text{窗口}
$$

### 5.4 Python实现

```python
import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.argmax = None  # 记录最大值位置
        
    def forward(self, X):
        """前向传播：记录最大值位置"""
        n, c, h, w = X.shape
        kh = kw = self.kernel_size
        sh = sw = self.stride
        
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1
        
        Y = np.zeros((n, c, h_out, w_out))
        self.argmax = np.zeros((n, c, h_out, w_out, 2), dtype=int)
        
        for i in range(h_out):
            for j in range(w_out):
                # 提取池化窗口
                window = X[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                # 找到最大值
                Y[:, :, i, j] = np.max(window, axis=(2, 3))
                # 记录最大值的位置（用于反向传播）
                max_idx = np.argmax(window.reshape(n, c, -1), axis=2)
                self.argmax[:, :, i, j, 0] = max_idx // kw
                self.argmax[:, :, i, j, 1] = max_idx % kw
        return Y
    
    def backward(self, dY):
        """反向传播：梯度只传给最大值位置"""
        n, c, h_out, w_out = dY.shape
        kh = kw = self.kernel_size
        sh = sw = self.stride
        
        h_in = (h_out - 1) * sh + kh
        w_in = (w_out - 1) * sw + kw
        dX = np.zeros((n, c, h_in, w_in))
        
        for i in range(h_out):
            for j in range(w_out):
                for b in range(n):
                    for ch in range(c):
                        # 获取最大值的位置
                        row = self.argmax[b, ch, i, j, 0]
                        col = self.argmax[b, ch, i, j, 1]
                        # 梯度只传给最大值位置
                        dX[b, ch, i*sh+row, j*sw+col] = dY[b, ch, i, j]
        return dX


class AvgPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, X):
        """前向传播：计算平均值"""
        n, c, h, w = X.shape
        kh = kw = self.kernel_size
        sh = sw = self.stride
        
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1
        
        Y = np.zeros((n, c, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                window = X[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                Y[:, :, i, j] = np.mean(window, axis=(2, 3))
        return Y
    
    def backward(self, dY):
        """反向传播：梯度平均分配"""
        n, c, h_out, w_out = dY.shape
        kh = kw = self.kernel_size
        sh = sw = self.stride
        
        h_in = (h_out - 1) * sh + kh
        w_in = (w_out - 1) * sw + kw
        dX = np.zeros((n, c, h_in, w_in))
        
        # 每个梯度平均分给 kh * kw 个位置
        for i in range(h_out):
            for j in range(w_out):
                dX[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] += \
                    dY[:, :, i:i+1, j:j+1] / (kh * kw)
        return dX
```

---

## 六、1×1卷积：通道维度上的“全能选手”

### 6.1 1×1卷积的本质

$1 \times 1$ 卷积与标准卷积**完全一样**，唯一的特殊点在于卷积核尺寸为 $1 \times 1$。

它**不关心空间维度上的局部关系**（$1 \times 1$ 只覆盖一个像素），而是**专注于通道维度上的信息整合**。

数学上，$1 \times 1$ 卷积对每个像素位置，在**所有通道上进行线性组合**：

$$
y_{ij}^{p} = \sum_{d=1}^{D} w_{p,d} \cdot x_{ij}^{d} + b_p
$$

这本质上等价于**在每个像素位置上执行一个全连接层**。

### 6.2 1×1卷积的三大作用

**作用一：跨通道的信息交互与整合**

$1 \times 1$ 卷积将不同通道在同一位置的信息进行线性组合，实现**跨通道的信息融合**。这相当于对特征图的“深度”维度进行了一次特征重组合。

**作用二：通道数的降维与升维**

$1 \times 1$ 卷积**不改变特征图的高度和宽度**，只改变通道数：
- 使用 $C_{out} < C_{in}$ 个 $1 \times 1$ 卷积核 → **降维**（减少通道数）
- 使用 $C_{out} > C_{in}$ 个 $1 \times 1$ 卷积核 → **升维**（增加通道数）

**作用三：增加非线性**

在 $1 \times 1$ 卷积后接非线性激活函数（如ReLU），可以在**不改变特征图空间尺寸**的前提下，大幅增加网络的非线性表达能力。

### 6.3 经典应用：GoogLeNet的Inception模块

GoogLeNet的Inception模块使用 $1 \times 1$ 卷积进行**通道降维**，大幅减少参数量。

以Inception模块的3a分支为例：
- 输入：$28 \times 28 \times 192$
- 如果不使用 $1 \times 1$ 降维，3×3和5×5分支的参数量为：

$$
3 \times 3 \times 192 \times 128 + 5 \times 5 \times 192 \times 32 \approx 1.6 \times 10^6
$$

- 在3×3前加96通道的 $1 \times 1$ 卷积，在5×5前加16通道的 $1 \times 1$ 卷积后：

$$
1 \times 1 \times 192 \times 96 + 3 \times 3 \times 96 \times 128 + 1 \times 1 \times 192 \times 16 + 5 \times 5 \times 16 \times 32 \approx 0.5 \times 10^6
$$

**参数量减少到原来的三分之一**！

### 6.4 经典应用：ResNet的Bottleneck结构

ResNet-50及更深版本使用**Bottleneck结构**，通过 $1 \times 1$ 卷积先降维再升维：

1. **$1 \times 1$ 卷积（降维）** ：将通道数从256降到64
2. **$3 \times 3$ 卷积**：在低维空间提取空间特征
3. **$1 \times 1$ 卷积（升维）** ：将通道数从64升回256

这种设计使得 $3 \times 3$ 卷积的参数量从 $3 \times 3 \times 256 \times 256$ 减少到 $3 \times 3 \times 64 \times 64$，**参数量减少了16倍**，让152层的ResNet得以在可接受的参数量下训练。

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    """ResNet Bottleneck块（简化版）"""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        # 1x1 降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3x3 空间特征提取
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1x1 升维
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity  # 残差连接
        return self.relu(out)

# 示例：Bottleneck将256通道压缩到64再恢复
block = Bottleneck(in_channels=256, mid_channels=64, out_channels=256)
print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")
# 输出约为 70,000 参数量（相比直接用3x3的约1,500,000，减少了95%以上）
```

---

## 七、总结

本文系统梳理了CNN的四大核心操作：

| 操作 | 核心要点 | 关键公式 |
|------|---------|---------|
| **卷积（互相关）** | 深度学习实现=互相关，无需翻转 | $y_{ij} = \sum_{u,v} w_{uv} \cdot x_{i+u-1,j+v-1}$ |
| **感受野** | 深层神经元看到更大的输入区域 | $RF_i = RF_{i+1} + (K_i-1) \times S_{i+1}$ |
| **空洞卷积** | 插入空洞扩大感受野，不增参数 | $K' = K + (K-1)(r-1)$ |
| **池化反向传播** | Max→只传最大值；Avg→平均分配 | Max: $\partial L/\partial x_{max} = \partial L/\partial y$; Avg: $\partial L/\partial x_{ij} = (1/k^2)\partial L/\partial y$ |
| **1×1卷积** | 跨通道信息整合，降维/升维，增非线性 | $y_{ij}^{p} = \sum_{d=1}^{D} w_{p,d} \cdot x_{ij}^{d} + b_p$ |

理解这些核心操作的数学原理，你就掌握了CNN的“底层语法”。无论是设计新的网络架构，还是调试现有模型，这些知识都是不可或缺的工具。

---

延伸阅读：
- [A guide to receptive field arithmetic for CNNs](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)（空洞卷积原始论文）
- [Network In Network](https://arxiv.org/abs/1312.4400)（1×1卷积的起源）
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（ResNet，Bottleneck结构）