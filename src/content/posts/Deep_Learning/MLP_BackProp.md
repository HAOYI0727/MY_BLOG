---
title: MLP & Back Propagation —— 多层感知机与反向传播
published: 2025-08-10
description: 系统讲解多层感知机（MLP）的前向传播矩阵运算与反向传播链式法则，以计算图为工具手动推导3层MLP的误差回传与参数梯度公式，提炼出误差回传、权重梯度、偏置梯度的核心公式。同时讨论通用近似定理（UAT）的砖块构造直觉，以及Batch Normalization训练时mini-batch统计与推理时滑动平均（EMA）机制。
cover: "/assets/images/posts/mlp_backprop.png"
coverInContent: false
tags: [MLP, 反向传播, 计算图, 链式法则, 批处理, 深度学习]
category: Deep_Learning
draft: false
---

# MLP & Back Propagation —— 多层感知机与反向传播

## 一、引言：为什么还要手推反向传播？

在PyTorch、TensorFlow等框架中，一行`loss.backward()`就能自动完成所有梯度计算。自动微分（Automatic Differentiation）确实大大简化了深度学习算法的实现。然而，**如果你只想停留在"调包侠"的层面，确实不需要理解反向传播；但如果你想真正理解深度学习在做什么，反向传播的推导是绕不开的必修课**。

本文将从最基础的矩阵前向传播开始，以**计算图（Computational Graph）** 为工具，手动推导一个3层MLP的反向传播过程，并顺带讨论通用近似定理的直觉解释和Batch Normalization的滑动平均机制。

---

## 二、前向传播：矩阵形式的全连接层

### 2.1 单个全连接层的数学表达

一个全连接层（Fully Connected Layer，也称稠密层或Affine层）接收输入向量，通过线性变换加偏置后输出。设输入为 $\mathbf{x} \in \mathbb{R}^{d}$，则该层的输出为：

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

其中 $\mathbf{W} \in \mathbb{R}^{h \times d}$ 是权重矩阵，$\mathbf{b} \in \mathbb{R}^{h}$ 是偏置向量。经过激活函数 $\phi$（如ReLU、Sigmoid等）后得到该层的输出：

$$
\mathbf{a} = \phi(\mathbf{z})
$$

这里的 $\mathbf{a}$ 即为下一层的输入。

### 2.2 批处理版本：向量化计算

实际训练中我们一次处理一个mini-batch，设批量大小为 $m$，输入矩阵为 $\mathbf{X} \in \mathbb{R}^{m \times d}$（每行一个样本），则前向传播为：

$$
\mathbf{Z} = \mathbf{X}\mathbf{W}^{\top} + \mathbf{b}
$$

注意这里 $\mathbf{W}^{\top}$ 的维度是 $h \times d$，$\mathbf{Z}$ 的维度是 $m \times h$，每一行对应一个样本在该层的输出。

> **💡 维度检验小技巧**：矩阵乘法中，中间维度必须匹配。$\mathbf{X}_{m \times d} \times \mathbf{W}^{\top}_{h \times d}$ 的转置是 $\mathbf{W}_{d \times h}$，所以 $\mathbf{X}\mathbf{W}^{\top}$ 得到 $m \times h$，正好是每个样本的 $h$ 维输出。偏置 $\mathbf{b}$ 通过广播（broadcasting）加到每一行上。

### 2.3 Python实现：前向传播

```python
import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        # 使用Xavier初始化
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)
        
    def forward(self, X):
        """
        X: (batch_size, input_dim)
        返回: (batch_size, output_dim)
        """
        self.X = X  # 保存用于反向传播
        self.Z = X @ self.W.T + self.b  # (m, h)
        return self.Z
```

---

## 三、以计算图视角手推3层MLP的反向传播

### 3.1 什么是计算图？

计算图是一种**有向无环图（DAG）** ，用于可视化运算符和变量在计算中的依赖关系。正向传播沿着图从输入到输出顺序计算，而反向传播则**按相反的顺序（从输出层到输入层）** 计算和存储中间变量及参数的梯度。

反向传播的本质就是**链式法则（Chain Rule）** 的反复应用。让我们用一个具体的3层MLP来走一遍完整流程。

### 3.2 网络结构定义

考虑一个3层MLP（输入层→隐藏层1→隐藏层2→输出层）：

- 输入层：$d$ 维
- 隐藏层1：$h_1$ 个神经元，权重 $\mathbf{W}^{(1)}$，偏置 $\mathbf{b}^{(1)}$，激活函数 $\phi_1$
- 隐藏层2：$h_2$ 个神经元，权重 $\mathbf{W}^{(2)}$，偏置 $\mathbf{b}^{(2)}$，激活函数 $\phi_2$
- 输出层：$q$ 个神经元，权重 $\mathbf{W}^{(3)}$，偏置 $\mathbf{b}^{(3)}$（输出层通常不加激活函数，或用Softmax）

为简化推导，我们**先考虑单个样本** $\mathbf{x} \in \mathbb{R}^{d}$，批量版本可以看作单样本版本的向量化堆叠。

前向传播的数学表达：

$$
\begin{aligned}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{a}^{(1)} &= \phi_1(\mathbf{z}^{(1)}) \\
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)} \\
\mathbf{a}^{(2)} &= \phi_2(\mathbf{z}^{(2)}) \\
\mathbf{z}^{(3)} &= \mathbf{W}^{(3)}\mathbf{a}^{(2)} + \mathbf{b}^{(3)} \\
\hat{\mathbf{y}} &= \mathbf{z}^{(3)} \quad (\text{输出层无激活})
\end{aligned}
$$

损失函数采用均方误差（MSE）：$L = \frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2$，其中 $\mathbf{y}$ 是真实标签。

### 3.3 计算图可视化

```
x → [W¹,b¹] → z¹ → [φ₁] → a¹ → [W²,b²] → z² → [φ₂] → a² → [W³,b³] → z³ → [Loss] → L
     ↑                  ↑                  ↑
   (参数)            (参数)            (参数)
```

箭头方向代表**数据依赖关系**：$L$ 依赖于 $\mathbf{z}^{(3)}$，$\mathbf{z}^{(3)}$ 依赖于 $\mathbf{W}^{(3)}$ 和 $\mathbf{a}^{(2)}$，依此类推。

### 3.4 反向传播：从输出到输入的梯度回传

反向传播的核心任务是计算损失 $L$ 对每个参数（$\mathbf{W}^{(l)}$ 和 $\mathbf{b}^{(l)}$）的梯度。我们从输出层开始，逐层向前推导。

#### 第1步：损失函数对输出 $\mathbf{z}^{(3)}$ 的梯度

$$
\frac{\partial L}{\partial \mathbf{z}^{(3)}} = \frac{\partial}{\partial \mathbf{z}^{(3)}} \left( \frac{1}{2}\|\mathbf{z}^{(3)} - \mathbf{y}\|^2 \right) = \mathbf{z}^{(3)} - \mathbf{y}
$$

记 **误差项** $\boldsymbol{\delta}^{(3)} = \frac{\partial L}{\partial \mathbf{z}^{(3)}} = \hat{\mathbf{y}} - \mathbf{y}$。

#### 第2步：输出层参数 $\mathbf{W}^{(3)}$ 和 $\mathbf{b}^{(3)}$ 的梯度

根据链式法则：

$$
\frac{\partial L}{\partial \mathbf{W}^{(3)}} = \frac{\partial L}{\partial \mathbf{z}^{(3)}} \cdot \frac{\partial \mathbf{z}^{(3)}}{\partial \mathbf{W}^{(3)}} = \boldsymbol{\delta}^{(3)} (\mathbf{a}^{(2)})^{\top}
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{(3)}} = \boldsymbol{\delta}^{(3)}
$$

这里 $\boldsymbol{\delta}^{(3)}$ 是 $q \times 1$ 的列向量，$(\mathbf{a}^{(2)})^{\top}$ 是 $1 \times h_2$ 的行向量，外积得到 $q \times h_2$ 的矩阵。

#### 第3步：误差继续回传至隐藏层2的输出 $\mathbf{a}^{(2)}$

$$
\frac{\partial L}{\partial \mathbf{a}^{(2)}} = (\mathbf{W}^{(3)})^{\top} \frac{\partial L}{\partial \mathbf{z}^{(3)}} = (\mathbf{W}^{(3)})^{\top} \boldsymbol{\delta}^{(3)}
$$

#### 第4步：通过激活函数 $\phi_2$ 回传至 $\mathbf{z}^{(2)}$

$$
\frac{\partial L}{\partial \mathbf{z}^{(2)}} = \frac{\partial L}{\partial \mathbf{a}^{(2)}} \odot \phi_2'(\mathbf{z}^{(2)})
$$

其中 $\odot$ 表示逐元素相乘（Hadamard积）。记 $\boldsymbol{\delta}^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}}$。

#### 第5步：隐藏层2的参数 $\mathbf{W}^{(2)}$ 和 $\mathbf{b}^{(2)}$ 的梯度

$$
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^{\top}, \quad \frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}
$$

#### 第6步：继续回传至隐藏层1

$$
\frac{\partial L}{\partial \mathbf{a}^{(1)}} = (\mathbf{W}^{(2)})^{\top} \boldsymbol{\delta}^{(2)}
$$

$$
\boldsymbol{\delta}^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{a}^{(1)}} \odot \phi_1'(\mathbf{z}^{(1)})
$$

#### 第7步：隐藏层1的参数梯度

$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^{\top}, \quad \frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}
$$

### 3.5 三个关键结论

观察上述推导，我们可以提炼出**三层MLP反向传播的三个核心公式**：

**（1）误差回传公式**（从第 $l+1$ 层传到第 $l$ 层）：

$$
\boldsymbol{\delta}^{(l)} = \left( (\mathbf{W}^{(l+1)})^{\top} \boldsymbol{\delta}^{(l+1)} \right) \odot \phi_l'(\mathbf{z}^{(l)})
$$

**（2）权重梯度公式**：

$$
\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^{\top}
$$

**（3）偏置梯度公式**：

$$
\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
$$

其中 $\mathbf{a}^{(0)} = \mathbf{x}$。

> **🔑 核心洞察**：反向传播的本质就是**将误差从输出层逐层"分发"回每一层**。每一层收到的"误差信号" $\boldsymbol{\delta}^{(l)}$ 包含了损失对该层**线性变换前**输入的敏感度，再结合该层的激活函数导数和输入，就能算出参数梯度。

### 3.6 批量版本的矩阵形式

对于batch size为 $m$ 的情况，上述所有向量变为矩阵，逐元素乘法变为逐元素乘法（仍为Hadamard积），公式形式不变：

$$
\mathbf{A}^{(0)} = \mathbf{X} \in \mathbb{R}^{m \times d}
$$

$$
\mathbf{Z}^{(l)} = \mathbf{A}^{(l-1)}(\mathbf{W}^{(l)})^{\top} + \mathbf{b}^{(l)}, \quad \mathbf{A}^{(l)} = \phi_l(\mathbf{Z}^{(l)})
$$

误差回传：

$$
\boldsymbol{\Delta}^{(l)} = \left( \boldsymbol{\Delta}^{(l+1)} \mathbf{W}^{(l+1)} \right) \odot \phi_l'(\mathbf{Z}^{(l)})
$$

权重梯度（注意除以batch size取平均）：

$$
\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{1}{m} (\boldsymbol{\Delta}^{(l)})^{\top} \mathbf{A}^{(l-1)}
$$

### 3.7 Python实现：三层MLP的反向传播

```python
import numpy as np

class ThreeLayerMLP:
    def __init__(self, d, h1, h2, q):
        # 初始化权重和偏置
        self.W1 = np.random.randn(h1, d) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h2, h1) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(q, h2) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(q)
        
    def forward(self, X):
        """前向传播，保存中间变量"""
        self.X = X
        self.Z1 = X @ self.W1.T + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = self.A1 @ self.W2.T + self.b2
        self.A2 = np.maximum(0, self.Z2)  # ReLU
        self.Z3 = self.A2 @ self.W3.T + self.b3
        return self.Z3
    
    def backward(self, y, learning_rate=0.01):
        """反向传播，计算梯度并更新参数"""
        m = self.X.shape[0]
        
        # 输出层误差
        delta3 = self.Z3 - y  # (m, q)
        
        # 输出层参数梯度
        dW3 = delta3.T @ self.A2 / m  # (q, h2)
        db3 = np.mean(delta3, axis=0)  # (q,)
        
        # 回传到第二隐藏层
        delta2 = (delta3 @ self.W3) * (self.Z2 > 0).astype(float)  # (m, h2)
        dW2 = delta2.T @ self.A1 / m  # (h2, h1)
        db2 = np.mean(delta2, axis=0)  # (h2,)
        
        # 回传到第一隐藏层
        delta1 = (delta2 @ self.W2) * (self.Z1 > 0).astype(float)  # (m, h1)
        dW1 = delta1.T @ self.X / m  # (h1, d)
        db1 = np.mean(delta1, axis=0)  # (h1,)
        
        # SGD更新
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return dW1, db1, dW2, db2, dW3, db3
```

---

## 四、通用近似定理（Universal Approximation Theorem）的直觉解释

### 4.1 定理在说什么？

通用近似定理（UAT）是神经网络理论中最令人惊叹的结果之一。它最经典的版本（Cybenko, 1989；Hornik et al., 1989）指出：

> **任何定义在紧致集上的连续函数，都可以被一个具有单隐藏层（足够宽）的前馈神经网络以任意精度近似。**

换句话说，**只要隐藏层有足够多的神经元，一个简单的MLP就能逼近任意复杂的函数**。

### 4.2 直觉理解：从"砖块"到"建筑"

理解UAT最直观的方式是**"砖块构造法"（bump construction）** 。

想象一下：**两个Sigmoid神经元可以组合成一个"凸起"函数（bump function）** ——这个函数只在某个小区间内非零，其他地方几乎为零。具体来说：

- 一个Sigmoid神经元 $\sigma(wx + b)$ 可以看作一个"阶跃"的平滑版本
- 两个Sigmoid相减：$\sigma(wx + b_1) - \sigma(wx + b_2)$，可以形成一个**"脉冲"或"凸起"**
- 通过调节 $w$ 控制凸起的宽度，调节 $b_1, b_2$ 控制凸起的位置

现在，**任意连续函数都可以看作无数个这样的"凸起"的叠加**——这正是傅里叶分析或勒贝格积分的基本思想：用矩形条（或更平滑的基函数）去逼近任意函数。

只要隐藏层有足够多的神经元，每个神经元对可以生成一个"凸起"，所有凸起叠加起来就逼近了目标函数。

> **⚠️ 重要提醒**：UAT是**存在性定理**，它告诉我们"存在这样一个网络"，但**没有告诉我们如何找到它**（即如何训练）。而且，虽然单隐藏层在理论上足够，但实践中**深度网络往往比宽度网络更高效**——用$2k$层比用$2^k$个神经元的单层要高效得多。

### 4.3 一个简单的可视化思考

想象你要用乐高积木拼出一个复杂的曲线：

- **浅层宽网络**（单隐藏层，很多神经元）：就像你有很多种不同形状的积木，每种只用一块
- **深层窄网络**（多层，每层少量神经元）：就像你只有几种基本形状的积木，但可以反复组合复用

UAT告诉我们：**只要有足够多的基本积木（神经元），总能拼出任何形状**。但实际操作中，深度网络通过**层次化组合**能用更少的参数实现更复杂的函数。

---

## 五、Batch Normalization：训练时的均值和方差滑动更新

### 5.1 为什么需要Batch Normalization？

深层神经网络在训练时面临一个棘手的问题：**每一层的输入分布随着前一层参数的变化而变化**——这被称为"内部协变量偏移"（Internal Covariate Shift）。Batch Normalization（BN）通过**对每一层的输入进行归一化**来解决这个问题，使每层的输入保持近似零均值和单位方差。

### 5.2 BN的核心操作

对于某个全连接层的输入 $\mathbf{x}$（一个mini-batch，大小为 $m$），BN执行以下变换：

**Step 1：计算batch的均值和方差**

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

**Step 2：归一化**

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

其中 $\epsilon$ 是一个很小的常数（如 $10^{-5}$），防止除零。

**Step 3：缩放和平移（可学习参数）**

$$
y_i = \gamma \hat{x}_i + \beta
$$

这里的 $\gamma$ 和 $\beta$ 是可训练参数，用于恢复模型的表达能力——如果标准化破坏了有用的特征分布，网络可以学习如何"撤销"部分标准化效果。

### 5.3 训练时的滑动平均更新

这里有一个**关键的细节**：在训练时，BN使用的是**当前mini-batch的均值和方差**进行归一化。但到了推理（测试）阶段，我们面对的往往只有一个样本（或很小的batch），无法计算有统计意义的均值和方差。

因此，BN在训练过程中需要**维护全局统计量**——即整个训练集的均值和方差的估计值。这个维护是通过**指数加权移动平均（Exponential Moving Average, EMA）** 实现的：

$$
\mu_{\text{global}} = \text{momentum} \cdot \mu_{\text{global}} + (1 - \text{momentum}) \cdot \mu_{\mathcal{B}}
$$

$$
\sigma_{\text{global}}^2 = \text{momentum} \cdot \sigma_{\text{global}}^2 + (1 - \text{momentum}) \cdot \sigma_{\mathcal{B}}^2
$$



其中：
- $\mu_{\text{global}}$ 和 $\sigma_{\text{global}}^2$ 是累积的全局均值和方差
- $\mu_{\mathcal{B}}$ 和 $\sigma_{\mathcal{B}}^2$ 是当前mini-batch的均值和方差
- **momentum** 是一个超参数，通常设为0.9或0.99

> **📌 momentum的直觉**：momentum越大，全局统计量更新越平滑，对历史数据的"记忆"越长（0.99相当于平均过去100个batch）；momentum越小，更新越快，对当前batch更敏感（0.9相当于平均过去10个batch）。

### 5.4 训练 vs 推理的行为差异

| 阶段 | 使用的均值和方差 | 是否更新全局统计量 |
|------|-----------------|-------------------|
| **训练** | 当前mini-batch的 $\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2$ | ✅ 更新 $\mu_{\text{global}}, \sigma_{\text{global}}^2$ |
| **推理** | 累积的 $\mu_{\text{global}}, \sigma_{\text{global}}^2$ | ❌ 不更新 |



在推理模式下（如PyTorch中调用`model.eval()`），BN层**冻结**全局统计量，直接用训练时累积的均值和方差进行归一化。这保证了推理时无论输入batch大小如何，都能得到稳定一致的输出。

### 5.5 Python实现：BN层的前向与滑动更新

```python
class BatchNorm:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # 可训练参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # 全局统计量（非训练参数，通过滑动平均更新）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # 保存中间变量用于反向传播
        self.x_centered = None
        self.std_inv = None
        self.x_norm = None
        
    def forward(self, X, training=True):
        """
        X: (batch_size, num_features)
        training: True表示训练模式，False表示推理模式
        """
        if training:
            # 计算当前batch的统计量
            batch_mean = np.mean(X, axis=0)      # (num_features,)
            batch_var = np.var(X, axis=0)        # (num_features,)
            
            # 归一化
            self.x_centered = X - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.eps)
            self.x_norm = self.x_centered * self.std_inv
            
            # 滑动平均更新全局统计量
            self.running_mean = self.momentum * self.running_mean + \
                                (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + \
                               (1 - self.momentum) * batch_var
        else:
            # 推理模式：使用累积的全局统计量
            self.x_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # 缩放和平移
        out = self.gamma * self.x_norm + self.beta
        return out
```

---

## 六、总结

本文从矩阵前向传播出发，逐层推导了3层MLP的反向传播过程，核心要点可以概括为：

1. **前向传播**是线性变换 + 非线性激活的复合，可以用简洁的矩阵形式表达
2. **反向传播**本质是链式法则的反复应用，误差从输出层逐层回传
3. **三个核心公式**：误差回传 $\boldsymbol{\delta}^{(l)} = ((\mathbf{W}^{(l+1)})^{\top} \boldsymbol{\delta}^{(l+1)}) \odot \phi_l'(\mathbf{z}^{(l)})$，权重梯度 $\partial L/\partial \mathbf{W}^{(l)} = \boldsymbol{\delta}^{(l)}(\mathbf{a}^{(l-1)})^{\top}$，偏置梯度 $\partial L/\partial \mathbf{b}^{(l)} = \boldsymbol{\delta}^{(l)}$
4. **通用近似定理**告诉我们单隐藏层网络在理论上能逼近任意连续函数，但深度网络在实践中效率更高
5. **Batch Normalization**在训练时使用mini-batch统计量归一化，同时通过**指数加权移动平均**维护全局统计量供推理使用

理解这些原理，你就掌握了深度学习最核心的"底层逻辑"。下一篇文章，我们将在此基础上探讨更复杂的网络结构——卷积神经网络（CNN）的反向传播。

---

延伸阅读：
- [Dive into Deep Learning - Backpropagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html)
- [Neural Networks and Deep Learning - Universality Theorem](http://neuralnetworksanddeeplearning.com/chap4.html)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)