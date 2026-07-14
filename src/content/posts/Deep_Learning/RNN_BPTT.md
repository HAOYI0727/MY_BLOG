---
title: RNN & BPTT —— 循环神经网络与随时间反向传播
published: 2025-08-18
description: 系统讲解循环神经网络（RNN）的数学原理与随时间反向传播（BPTT）算法。从RNN的循环结构与共享参数出发，推导BPTT的梯度表达式，揭示梯度消失与梯度爆炸的数学根源，并介绍梯度裁剪作为应对梯度爆炸的工程解法。附带讨论LSTM如何通过门控机制缓解梯度消失。
cover: "/assets/images/posts/rnn_bptt.png"
coverInContent: false
tags: [RNN, BPTT, 梯度消失, 梯度爆炸, 梯度裁剪, LSTM, 深度学习]
category: Deep_Learning
draft: false
---

# RNN & BPTT —— 循环神经网络与随时间反向传播

## 一、引言：RNN的“记忆”与“遗忘”

在前几篇文章中，我们讨论了前馈网络（MLP、CNN）的反向传播。这些网络有一个共同特点：**输入之间是独立的**——一张图片的分类结果不依赖于前一张图片。

但序列数据（文本、语音、时间序列）打破了这一假设。**今天的单词依赖于昨天的单词，明天的股价依赖于今天的股价**。循环神经网络（RNN）正是为此而生——它通过**隐藏状态的循环**，让网络拥有了“记忆”。

然而，RNN的训练远比前馈网络复杂。核心原因在于：**误差不仅要在层之间反向传播，还要在时间步之间反向传播**。这个算法被称为**BPTT（Backpropagation Through Time，随时间反向传播）**。

BPTT在数学上优雅而直接，但在实践中却遭遇了致命困境——**梯度消失与梯度爆炸**，这一问题最早由Sepp Hochreiter在1991年发现。本文将深入BPTT的数学推导，剖析梯度问题的本质，并介绍梯度裁剪这一实用的工程解法。

---

## 二、RNN的结构与前向传播

### 2.1 标准RNN的数学定义

一个标准的循环神经网络在时间步 $t$ 的隐藏状态 $\mathbf{h}_t$ 和输出 $\mathbf{o}_t$ 可以表示为：

$$
\mathbf{h}_t = \phi(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)
$$

$$
\mathbf{o}_t = \psi(\mathbf{W}_{ho}\mathbf{h}_t + \mathbf{b}_o)
$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{d}$ 是时间步 $t$ 的输入
- $\mathbf{h}_t \in \mathbb{R}^{h}$ 是隐藏状态（网络的“记忆”）
- $\mathbf{o}_t \in \mathbb{R}^{q}$ 是输出
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$：输入到隐藏层的权重
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$：隐藏层到隐藏层的权重（**循环的核心**）
- $\mathbf{W}_{ho} \in \mathbb{R}^{q \times h}$：隐藏层到输出的权重
- $\phi$ 是隐藏层激活函数（通常为 $\tanh$ 或 ReLU）
- $\psi$ 是输出层激活函数（通常为 Softmax 或恒等映射）

> **💡 关键洞察**：$\mathbf{W}_{hh}$ 是RNN的“记忆矩阵”——它决定了当前隐藏状态如何继承上一时刻的信息。正是这个矩阵的循环使用，让RNN拥有了处理变长序列的能力。

### 2.2 按时间步展开的计算图

BPTT的核心思想是：**将循环网络按时间步“展开”成一个前馈网络**。

对于一个长度为 $T$ 的序列，展开后的计算图为：

```
x₁ → h₁ → o₁ → L₁
↑    ↑
Wxh  Whh
     ↓
x₂ → h₂ → o₂ → L₂
↑    ↑
Wxh  Whh
     ↓
...
x_T → h_T → o_T → L_T
```

注意：**所有时间步共享同一组参数**（$\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{ho}$）。这是RNN与前馈网络的关键区别——也是BPTT梯度推导的难点所在。

### 2.3 损失函数

对于序列数据，总损失是所有时间步损失之和：

$$
L = \sum_{t=1}^{T} L_t(\mathbf{o}_t, \mathbf{y}_t)
$$

其中 $\mathbf{y}_t$ 是时间步 $t$ 的真实标签。

---

## 三、BPTT的梯度推导

### 3.1 梯度的“两条路径”

在BPTT中，损失 $L$ 对参数 $\mathbf{W}_{hh}$ 的梯度需要**沿着两条路径**传播：

1. **同一时间步内**：通过 $\partial \mathbf{o}_t / \partial \mathbf{W}_{hh}$ 直接传播
2. **跨时间步**：通过隐藏状态 $\mathbf{h}_t$ 对 $\mathbf{h}_{t-1}$ 的依赖关系传播

对于参数 $\mathbf{W}_{hh}$，总梯度为所有时间步贡献之和：

$$
\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial \mathbf{W}_{hh}}
$$

### 3.2 单个时间步的梯度展开

根据链式法则，$\partial L_t / \partial \mathbf{W}_{hh}$ 可以展开为：

$$
\frac{\partial L_t}{\partial \mathbf{W}_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \cdot \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}
$$

这个公式的含义是：**时间步 $t$ 的损失，受到从时间步 $k$ 到 $t$ 的所有隐藏状态的影响**。

其中最关键的是 $\partial \mathbf{h}_t / \partial \mathbf{h}_k$，它表示了**隐藏状态在时间上的传播**。

### 3.3 隐藏状态的时间传播——核心推导

根据隐藏状态的更新公式 $\mathbf{h}_i = \phi(\mathbf{W}_{xh}\mathbf{x}_i + \mathbf{W}_{hh}\mathbf{h}_{i-1} + \mathbf{b}_h)$，我们有：

$$
\frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi'(\mathbf{h}_{i-1}))
$$

其中 $\text{diag}(\phi'(\mathbf{h}_{i-1}))$ 是对角矩阵，对角线上是激活函数导数在各隐藏神经元上的值。

因此，从时间步 $k$ 到 $t$ 的梯度传播为：

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=k+1}^{t} \left( \mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi'(\mathbf{h}_{i-1})) \right)
$$

**这就是BPTT的核心公式——一个从 $k+1$ 到 $t$ 的矩阵连乘**。

### 3.4 完整的梯度表达式

将上述结果代入，得到 $\partial L_t / \partial \mathbf{W}_{hh}$ 的完整表达式：

$$
\frac{\partial L_t}{\partial \mathbf{W}_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial \mathbf{h}_t} \cdot \left( \prod_{i=k+1}^{t} \mathbf{W}_{hh}^{\top} \text{diag}(\phi'(\mathbf{h}_{i-1})) \right) \cdot \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}
$$

最终，总梯度为：

$$
\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial \mathbf{h}_t} \cdot \left( \prod_{i=k+1}^{t} \mathbf{W}_{hh}^{\top} \text{diag}(\phi'(\mathbf{h}_{i-1})) \right) \cdot \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}
$$

> **🔑 核心公式的直观理解**：这个双重求和告诉我们——**每一个时间步的损失，都受到所有历史时间步的影响**，而影响的强度由一连串矩阵乘积决定。距离越远（$t-k$ 越大），连乘的项越多，梯度就越容易出问题。

---

## 四、梯度消失与梯度爆炸：BPTT的“阿喀琉斯之踵”

### 4.1 数学根源：矩阵连乘的指数行为

观察 $\partial \mathbf{h}_t / \partial \mathbf{h}_k$ 的表达式：

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \left( \mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi'(\mathbf{h}_{i-1})) \right)
$$

这个连乘的行为取决于矩阵 $\mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi'(\mathbf{h}_{i-1}))$ 的**谱范数**（即最大奇异值）：

- 如果谱范数 **< 1**：连乘结果**指数级衰减** → **梯度消失**
- 如果谱范数 **> 1**：连乘结果**指数级增长** → **梯度爆炸**

对于 $\tanh$ 激活函数，$\phi'(\mathbf{h}) = 1 - \tanh^2(\mathbf{h}) \in (0, 1]$，其最大值恰好为1。这意味着：

$$
\| \mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi'(\mathbf{h})) \| \leq \| \mathbf{W}_{hh} \| \cdot \max(\phi') = \| \mathbf{W}_{hh} \|
$$

- 当 $\| \mathbf{W}_{hh} \| < 1$ 时 → **梯度消失**（更常见）
- 当 $\| \mathbf{W}_{hh} \| > 1$ 时 → **梯度爆炸**（虽少见，但破坏性极强）

### 4.2 梯度消失：为什么RNN“记不住”长序列？

当 $\| \mathbf{W}_{hh} \| < 1$ 时：

$$
\left\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right\| \approx (\gamma \cdot \| \mathbf{W}_{hh} \|)^{t-k}
$$

其中 $\gamma = \max(\phi') \leq 1$。当 $t-k$ 较大时（即依赖关系较长），梯度**指数级趋近于0**。

**直观表现**：

- **短期依赖有效**：相邻时间步的梯度（如 $t-k=1$）衰减很小，模型可以学习局部关系
- **长期依赖失效**：远时间步的梯度（如 $t-k=10$）几乎为0，模型无法捕捉跨度较大的依赖

> **📌 实例**：在一个100个单词的句子中，第1个单词对第100个单词的影响，在BPTT中需要经过99次矩阵连乘。如果每次乘法的谱范数为0.9，那么 $0.9^{99} \approx 3 \times 10^{-5}$——梯度几乎完全消失了。

### 4.3 梯度爆炸：虽不常见，但破坏性极强

当 $\| \mathbf{W}_{hh} \| > 1$ 时，梯度会**指数级增长**。

梯度爆炸虽然不如梯度消失常见（因为权重初始化通常会控制 $\| \mathbf{W}_{hh} \|$ 在1附近），但一旦发生，后果极其严重：

- 梯度值变为 `NaN`（无穷大）
- 参数更新步长极大，模型权重发散
- 训练彻底崩溃

在实际训练中，**梯度爆炸更容易被检测到**（梯度值突然变得极大），而**梯度消失更隐蔽**（训练缓慢但不崩溃）。

---

## 五、梯度裁剪（Gradient Clipping）：应对梯度爆炸的工程利器

### 5.1 为什么需要梯度裁剪？

梯度爆炸的解决方案相对直接——**在反向传播过程中限制梯度的大小**。

梯度裁剪（Gradient Clipping）的核心思想是：**如果梯度的范数超过某个阈值，就将其缩放到该阈值以内，同时保持梯度方向不变**。

### 5.2 基于范数的梯度裁剪（Norm-based Clipping）

最常用的方法是**基于全局范数的梯度裁剪**。

设所有参数的梯度为 $\mathbf{g} = [g_1, g_2, \ldots, g_N]$，其 $L_2$ 范数为：

$$
\| \mathbf{g} \|_2 = \sqrt{\sum_{i=1}^{N} g_i^2}
$$

设定阈值 $\theta$（通常称为 `max_norm`），裁剪后的梯度为：

$$
\mathbf{g}_{\text{clipped}} = 
\begin{cases}
\mathbf{g}, & \text{if } \| \mathbf{g} \|_2 \leq \theta \\
\theta \cdot \frac{\mathbf{g}}{\| \mathbf{g} \|_2}, & \text{if } \| \mathbf{g} \|_2 > \theta
\end{cases}
$$

**关键特性**：
- 梯度方向**保持不变**（只改变模长）
- 梯度范数被限制在 $\theta$ 以内
- 所有参数梯度**统一缩放**，保持相对大小关系

> **💡 为什么是全局裁剪而非逐层裁剪？** 如果逐层独立裁剪，会破坏不同层之间梯度的相对比例，影响优化 dynamics。全局裁剪保持了梯度的方向信息，是更合理的选择。

### 5.3 阈值 $\theta$ 如何设置？

阈值的选择是梯度裁剪中最关键的超参数：

**经验法则**：

1. **从较大的值开始**（如 1.0 或 5.0），观察训练过程中梯度的平均范数
2. **如果出现梯度爆炸**（loss突然变为NaN），降低阈值
3. **如果训练过于稳定但收敛慢**，适当提高阈值

**常见实践**：
- 很多研究将阈值设为 **1.0**
- 一些工作中阈值设为 **5.0、10.0 甚至更大**
- 建议监控梯度范数的统计分布，将阈值设为平均范数的 **2-3 倍**

**PyTorch中的实现**：

```python
import torch
import torch.nn as nn

# 在反向传播之后、优化器更新之前进行梯度裁剪
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`clip_grad_norm_` 函数会计算所有参数的梯度总范数，如果超过 `max_norm`，则统一缩放。

### 5.4 Python实现：从零实现梯度裁剪

```python
import numpy as np

def clip_gradients(parameters, gradients, max_norm):
    """
    基于全局L2范数的梯度裁剪
    
    Args:
        parameters: 模型参数列表
        gradients: 与parameters对应的梯度列表
        max_norm: 梯度范数的阈值
    
    Returns:
        clipped_gradients: 裁剪后的梯度列表
    """
    # 计算所有梯度的L2范数平方
    total_norm_sq = 0.0
    for grad in gradients:
        total_norm_sq += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm_sq)
    
    # 如果范数超过阈值，进行缩放
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        clipped_gradients = [grad * clip_coef for grad in gradients]
    else:
        clipped_gradients = gradients
    
    return clipped_gradients, total_norm

# 使用示例
# 假设有一个3层RNN的参数
params = [np.random.randn(10, 10) for _ in range(3)]  # 模拟参数
grads = [np.random.randn(10, 10) * 5 for _ in range(3)]  # 模拟梯度

clipped_grads, norm = clip_gradients(params, grads, max_norm=1.0)
print(f"原始梯度范数: {np.sqrt(sum(np.sum(g**2) for g in grads)):.4f}")
print(f"裁剪后梯度范数: {np.sqrt(sum(np.sum(g**2) for g in clipped_grads)):.4f}")
```

### 5.5 两种梯度裁剪方式的对比

| 方式 | 操作 | 适用场景 |
|------|------|---------|
| **按值裁剪** (`clip_by_value`) | 将每个梯度值截断到 `[min, max]` 区间 | 需要严格控制每个梯度值范围 |
| **按范数裁剪** (`clip_by_norm` / `clip_by_global_norm`) | 缩放梯度向量使其范数不超过阈值 | **RNN/ LSTM 的推荐做法**，保持梯度方向 |

对于RNN训练，**强烈推荐使用基于全局范数的裁剪**，因为它保持了梯度的方向信息。

---

## 六、从BPTT到LSTM：门控机制如何缓解梯度消失

虽然梯度裁剪能有效应对**梯度爆炸**，但对于**梯度消失**，裁剪无能为力——它只能限制过大的梯度，无法“放大”消失的梯度。

真正从根本上缓解梯度消失的是**LSTM和GRU等门控RNN架构**。

### 6.1 LSTM的核心思想：加法路径替代乘法路径

LSTM引入了一个**细胞状态（Cell State）** $\mathbf{c}_t$，其更新方式为：

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

其中 $\mathbf{f}_t$ 是遗忘门，$\mathbf{i}_t$ 是输入门，$\odot$ 是逐元素乘法。

**关键差异**：在普通RNN中，$\partial \mathbf{h}_t / \partial \mathbf{h}_{t-1}$ 是**矩阵乘法**（$\mathbf{W}_{hh}^{\top} \cdot \text{diag}(\phi')$）；而在LSTM中，细胞状态的传递路径上是**逐元素乘法**（$\mathbf{f}_t \odot \mathbf{c}_{t-1}$）。

逐元素乘法的好处是：
- **没有矩阵乘法**，不存在谱范数导致的指数衰减
- 遗忘门 $\mathbf{f}_t$ 可以学习为接近1的值，让梯度几乎无损地通过
- 梯度在细胞状态路径上的衰减是**逐元素的、可控的**，而非全局性的

### 6.2 GRU的简化设计

GRU（门控循环单元）是LSTM的简化版本，合并了细胞状态和隐藏状态：

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

其中 $\mathbf{z}_t$ 是更新门。与LSTM类似，GRU通过 $(1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1}$ 这一**加法路径**，为梯度提供了直接传播的通道。

---

## 七、完整的RNN训练流程：BPTT + 梯度裁剪

```python
import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 参数初始化（使用较小的值，减少梯度爆炸风险）
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_ho = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((output_dim, 1))
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x_sequence):
        """
        前向传播：按时间步展开
        x_sequence: list of (input_dim, 1) 向量
        """
        T = len(x_sequence)
        h = np.zeros((T + 1, self.hidden_dim, 1))
        o = np.zeros((T, self.hidden_dim, 1))  # 简化：输出=隐藏状态
        
        for t in range(T):
            h[t] = np.tanh(self.W_xh @ x_sequence[t] + 
                           self.W_hh @ h[t-1] + self.b_h)
            o[t] = self.W_ho @ h[t] + self.b_o
        
        return o, h
    
    def bptt(self, x_sequence, y_sequence, max_norm=1.0):
        """
        随时间反向传播 + 梯度裁剪
        """
        T = len(x_sequence)
        
        # 前向传播
        o, h = self.forward(x_sequence)
        
        # 初始化梯度
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_ho = np.zeros_like(self.W_ho)
        db_h = np.zeros_like(self.b_h)
        db_o = np.zeros_like(self.b_o)
        
        # 输出层梯度
        delta_o = o - y_sequence  # 假设MSE损失
        
        # 从最后一个时间步反向传播
        delta_h = np.zeros((self.hidden_dim, 1))
        for t in range(T-1, -1, -1):
            # 输出层参数梯度
            dW_ho += delta_o[t] @ h[t].T
            db_o += delta_o[t]
            
            # 从输出层到隐藏层的误差
            delta_h = self.W_ho.T @ delta_o[t] + delta_h
            # 通过tanh激活函数
            delta_h = delta_h * (1 - h[t]**2)
            
            # 隐藏层参数梯度（当前时间步）
            dW_xh += delta_h @ x_sequence[t].T
            db_h += delta_h
            
            # 对Whh的梯度：需要累加所有历史时间步
            # 这里简化处理，实际BPTT需要循环k
            dW_hh += delta_h @ h[t-1].T
            
            # 准备传递给下一层（更早的时间步）
            delta_h = self.W_hh.T @ delta_h
        
        # 梯度裁剪（基于全局范数）
        grads = [dW_xh, dW_hh, dW_ho, db_h, db_o]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > max_norm:
            scale = max_norm / total_norm
            dW_xh *= scale
            dW_hh *= scale
            dW_ho *= scale
            db_h *= scale
            db_o *= scale
        
        return dW_xh, dW_hh, dW_ho, db_h, db_o
```

---

## 八、总结

| 概念 | 核心要点 | 关键公式 |
|------|---------|---------|
| **RNN前向传播** | 隐藏状态循环传递，所有时间步共享参数 | $\mathbf{h}_t = \phi(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)$ |
| **BPTT** | 将RNN按时间步展开，沿时间反向传播误差 | $\partial L/\partial \mathbf{W}_{hh} = \sum_{t=1}^{T} \sum_{k=1}^{t} \partial L_t/\partial \mathbf{h}_t \cdot (\prod_{i=k+1}^{t} \mathbf{W}_{hh}^{\top} \text{diag}(\phi'(\mathbf{h}_{i-1}))) \cdot \partial \mathbf{h}_k/\partial \mathbf{W}_{hh}$ |
| **梯度消失** | 矩阵连乘谱范数<1，梯度指数衰减 | $\| \partial \mathbf{h}_t/\partial \mathbf{h}_k \| \approx (\gamma \cdot \| \mathbf{W}_{hh} \|)^{t-k}$ |
| **梯度爆炸** | 矩阵连乘谱范数>1，梯度指数增长 | 同上，条件相反 |
| **梯度裁剪** | 限制梯度范数，保持方向不变 | $\mathbf{g}_{\text{clipped}} = \theta \cdot \mathbf{g} / \| \mathbf{g} \|_2$ 当 $\| \mathbf{g} \|_2 > \theta$ |

**BPTT的三大启示**：

1. **时间是有代价的**：序列越长，BPTT的计算和内存开销越大，梯度问题越严重
2. **矩阵连乘是双刃剑**：它让RNN能够传递信息，也让梯度指数级消失或爆炸
3. **门控机制是出路**：LSTM和GRU通过加法路径替代乘法路径，从根本上缓解了梯度消失

理解BPTT的数学原理，不仅有助于理解为什么RNN难以训练长序列，也为理解更先进的序列模型（如Transformer中的自注意力机制）奠定了基础——**Transformer彻底抛弃了循环结构，用并行化的注意力机制替代了时序上的串行传递**，从根本上规避了BPTT的梯度困境。

---

延伸阅读：
- [Backpropagation Through Time: What It Does and How to Do It](http://proceedings.mlr.press/v9/glorot10a.html)（Werbos, 1990）
- [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)（Pascanu et al., 2013）
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)（Hochreiter & Schmidhuber, 1997）
- [Dive into Deep Learning - BPTT](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)