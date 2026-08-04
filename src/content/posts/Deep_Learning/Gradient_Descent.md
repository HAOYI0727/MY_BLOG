---
title: Gradient Descent Optimizer —— 梯度下降优化器
published: 2025-08-12
description: 系统梳理梯度下降优化器的完整进化路径：从BGD、SGD到MBGD的效率与稳定性权衡，从动量法与NAG解决峡谷地形震荡，到AdaGrad实现参数级学习率、RMSProp引入指数加权平均解决学习率消亡，最终融合为集大成者Adam，并深入剖析AdamW如何通过解耦权重衰减修复Adam的正则化失效问题。
cover: "/assets/images/posts/gradient_descent.png"
coverInContent: false
tags: [梯度下降, SGD, 动量法, AdamW, 权重衰减, 深度学习]
category: Deep_Learning
draft: false
---

# Gradient Descent Optimizer —— 梯度下降优化器

## 一、引言：从“梯度之源”到“参数之跃”

在上一篇博客中，我们亲手推导了**参数梯度的完整计算公式** —— 从输出层的**误差信号** $\boldsymbol{\delta}^{(3)}$ 逐层回传至输入层，最终得到了每一层**权重与偏置的梯度**。至此，模型“学什么”与“梯度怎么算”均已明晰。然而，最后一个关键问题浮出水面：**有了梯度，如何用它来更新参数，才能让模型又快又稳地收敛？**

本篇博客正是回答这一“最终落地”问题的核心章节。我们从最基础的**批量梯度下降（BGD）** 出发，审视其全量计算的稳健性与大规模数据下的效率瓶颈；随后引入**随机梯度下降（SGD）** 与**小批量梯度下降（MBGD）**，揭示“效率与噪声”之间的经典权衡——而这正是优化器演进的起点。

随着对损失景观理解的深入，我们将逐次展开优化器的完整进化图谱：**动量法**与**NAG**如何通过惯性积累抑制峡谷震荡；**AdaGrad**如何为稀疏特征定制参数级学习率；**RMSProp**如何用指数加权平均解决AdaGrad的学习率消亡问题；以及**Adam**如何集一阶矩（动量）与二阶矩（自适应学习率）之大成。最后，我们将深入剖析**AdamW**的核心改进——通过**解耦权重衰减**，修复了Adam中L2正则化被自适应学习率缩放的隐藏缺陷，使其成为Transformer等现代架构的首选。

值得注意的是，本篇对 **优化器“方向与步长”** 的深刻理解，将直接服务于后续**卷积神经网络**中大规模参数的高效训练——当您面对百万级参数时，选择何种优化器将决定模型能否在有限时间内达到最优。现在，请带着“**如何用梯度驱动参数走向最低点**”的问题进入正文——理解了本篇，您就掌握了让模型真正“学进去”的最后一环。

---

## 二、梯度下降三兄弟：BGD、SGD与MBGD

### 2.1 批量梯度下降（BGD）：最稳健的“老实人”

**批量梯度下降（Batch Gradient Descent, BGD）** 每次迭代使用**全部训练数据**来计算损失函数的梯度，然后更新模型参数。

$$g_t = \frac{1}{n}\sum_{i=1}^{n} \nabla_\theta \ell(f_{\theta_t}(x_i), y_i)$$

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

其中 $n$ 是样本总数，$\eta$ 是学习率。

**优点**：由于使用的是全量数据的真实梯度方向，BGD的收敛曲线**相对平滑**，对于凸函数能够保证收敛到全局最优解。

**缺点**：当数据集规模庞大时，每次迭代的计算量巨大，对内存要求高。在动辄百万、千万甚至亿级样本的数据集上，计算一次真实梯度的成本是无法接受的。

### 2.2 随机梯度下降（SGD）：轻装上阵的“冒险家”

**随机梯度下降（Stochastic Gradient Descent, SGD）** 每次迭代只**随机选取一个样本**，根据该样本计算梯度并更新参数。

$$g_t = \nabla_\theta \ell(f_{\theta_t}(x_i), y_i)$$
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

**优点**：每次只计算一个样本的梯度，计算量小、更新速度快，对内存需求低，适合处理大规模数据集和在线学习场景。

**缺点**：梯度方向波动大，收敛路径曲折。SGD每次只依据一个样本更新，导致收敛过程具有较大的**随机性和波动性**，可能在最优解附近震荡。

不过，这种随机性也使得SGD有更大机会**跳出局部最优解**，尤其是在处理非凸函数时——这反而成了它的一个优势。

### 2.3 小批量梯度下降（MBGD）：集大成者

**小批量梯度下降（Mini-Batch Gradient Descent, MBGD）** 是BGD和SGD的折中方案——每次迭代使用一小部分样本（一个小批量）来计算梯度和更新参数。

$$g_t = \frac{1}{b}\sum_{i=1}^{b} \nabla_\theta \ell(f_{\theta_t}(x_i), y_i)$$
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

其中 $b$ 是批量大小（batch size），通常取 $16$ 到 $256$ 之间。

**优点**：结合了BGD的稳定性和SGD的随机性，通常能更稳定地收敛，且收敛速度比BGD快。同时，小批量的随机性也带来了一定的跳出局部最优的能力。此外，小批量计算可以充分利用GPU的**向量化并行计算**能力。

**实际应用**：MBGD在实际的深度学习应用中**最为广泛**，如图像分类、自然语言处理等领域。

```python
import numpy as np

def gradient_descent(X, y, theta, learning_rate, batch_size, n_epochs, method='mbgd'):
    """梯度下降的三种变体实现"""
    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        if method == 'bgd':
            # 批量梯度下降：使用全部数据
            gradients = compute_gradients(X, y, theta)
            theta -= learning_rate * gradients
            
        elif method == 'sgd':
            # 随机梯度下降：每次一个样本
            for i in range(n_samples):
                idx = np.random.randint(n_samples)
                gradients = compute_gradients(X[idx:idx+1], y[idx:idx+1], theta)
                theta -= learning_rate * gradients
                
        elif method == 'mbgd':
            # 小批量梯度下降：打乱数据后分批
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                gradients = compute_gradients(X[batch_indices], y[batch_indices], theta)
                theta -= learning_rate * gradients
                
    return theta
```

### 2.4 三种方法的收敛特性对比

| 特性 | BGD | SGD | MBGD |
|------|-----|-----|------|
| 每次使用的样本数 | 全部 $n$ | 1个 | $b$ 个（$16 \sim 256$） |
| 梯度准确性 | 最准确 | 噪声最大 | 适中 |
| 收敛速度（每次迭代） | 慢 | 快 | 适中 |
| 收敛稳定性 | 最稳定 | 波动大 | 较稳定 |
| 跳出局部最优能力 | 弱 | 强 | 中等 |
| 计算效率（大规模数据） | 低 | 高 | 高 |

从SGD到MBGD，我们解决了“**效率与噪声的权衡**”问题。但SGD/MBGD仍然面临一个根本性的挑战：**在“峡谷”地形中震荡严重**。


## 三、动量（Momentum）与Nesterov加速：给梯度加上“惯性”

### 3.1 问题的根源：峡谷地形的震荡

标准梯度下降法在参数空间存在 **“峡谷”地形**（即某一维度梯度远大于另一维度）时，会沿着陡峭方向来回震荡，导致有效前进速度大幅降低。

这种“近视”的更新方式存在两个本质缺陷：
1. **仅考虑当前点的瞬时梯度信息**，无法利用历史梯度轨迹
2. **在病态曲面上**，梯度方向可能与最优方向存在较大偏差

### 3.2 动量法：模拟物理世界的惯性

**动量法（Momentum）** 由Rumelhart等人在1986年提出，其核心思想是引入 **“速度”变量**来累积历史梯度信息。

我们可以将动量算法**视为模拟连续时间下牛顿动力学下的粒子**。想象一个在冰面上滑行的冰球——每当它沿着表面最陡的部分下降时，它会积累继续在该方向上滑行的速度。**负梯度是推动粒子的力，速度是粒子的动量**。

动量法的更新规则分为两步：

$$v_{t+1} = \gamma v_t + \eta \nabla_\theta J(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

其中 $\gamma \in (0,1)$ 是**动量系数**（通常取 $0.9$），控制历史信息的衰减速率。

从展开式可以更清楚地看到动量的“记忆”本质：

$$v_t = \eta \sum_{i=1}^{t} \gamma^{t-i} \nabla_\theta J(\theta_i)$$

速度 $v_t$ 是**所有历史梯度的指数加权移动平均**——近期梯度权重更大，但长期趋势也被保留。

**动量系数的物理意义**：如果将动量系数 $\gamma = 0.9$，则最大速度可以达到纯梯度下降的 $1/(1-\gamma) = 10$ 倍。这意味着在梯度方向一致的维度上，动量可以**积累动能实现加速**；而在梯度方向频繁变化的维度上，动量效应会**抵消震荡分量**。

```python
def sgd_with_momentum(theta, grad, v, lr=0.01, momentum=0.9):
    """带动量的SGD更新"""
    v = momentum * v + lr * grad
    theta = theta - v
    return theta, v

# 模拟在峡谷地形中的对比
# 纯SGD会在陡峭方向来回震荡，而动量法能平滑震荡、加速前进
```

### 3.3 Nesterov加速梯度（NAG）： “看得更远”的动量

**Nesterov加速梯度（Nesterov Accelerated Gradient, NAG）** 是动量的一个改进版本。

两者的关键区别在于**计算梯度的位置**：

- **标准动量**：在当前参数位置 $\theta_t$ 计算梯度
- **Nesterov动量**：在“预期位置” $\theta_t + \gamma v_t$ 计算梯度

NAG的更新规则：

$$v_{t+1} = \gamma v_t + \eta \nabla_\theta J(\theta_t + \gamma v_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

**物理直觉**：标准动量像一个滑雪者，每次到弯道才根据当前坡度调整方向；而Nesterov动量像一个**有前瞻性的滑雪者**——他先“看向”前方（预期位置），根据前方的坡度预先调整方向。

这种“**前瞻**”机制使得NAG在理论上具有更快的收敛速度，其收敛速率可以达到 $O(1/t^2)$，而标准梯度下降仅为 $O(1/t)$。


## 四、自适应学习率：为每个参数定制步长

### 4.1 AdaGrad：让“稀有特征”获得更大的更新

尽管动量和NAG解决了**方向**的问题，但所有参数仍然共享**同一个全局学习率**。这带来了新的问题：

- 对于**频繁出现的特征**，其梯度已经非常稳定，不需要大步更新
- 对于**稀疏特征**（如推荐系统中的低频用户ID），每次出现都应该给予较大的更新幅度

**AdaGrad（Adaptive Gradient）** 由Duchi等人在2011年提出，首次引入了**参数级学习率调整**的机制。

AdaGrad的核心思想是：**为每个参数维护一个累积梯度平方和，用其开方来缩放当前梯度**。

$$G_{t} = G_{t-1} + g_t^2 \quad \text{（逐元素平方和）}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

其中 $\odot$ 表示逐元素乘法，$\epsilon$ 是一个防止除零的小常数。

**直观理解**：
- 如果某个参数的历史梯度**很大**，$G_t$ 就大，学习率就被**缩小**
- 如果某个参数的历史梯度**很小**（稀疏特征），$G_t$ 就小，学习率就被**放大**

**致命缺陷**：$G_t$ 是**所有历史梯度平方的无衰减累加**。随着训练进行，$G_t$ 不断增大，学习率**单调递减**，最终趋近于零——模型将**停止学习**。

### 4.2 RMSProp：引入“遗忘”机制

**RMSProp**由Geoff Hinton在2012年的课程笔记中提出，其核心改进是**用指数加权移动平均替代累积平方和**。

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$$

其中 $\beta$ 通常取 $0.9$ 或 $0.99$。

**关键改进**：指数加权平均使得**远距离的历史梯度信息被逐渐“遗忘”** 。学习率不再单调递减到零，而是能够根据近期梯度的变化动态调整。

```python
def rmsprop(theta, grad, cache, lr=0.001, beta=0.9, epsilon=1e-8):
    """RMSProp更新"""
    cache = beta * cache + (1 - beta) * grad**2
    theta = theta - lr * grad / (np.sqrt(cache) + epsilon)
    return theta, cache
```


## 五、Adam：集大成者

### 5.1 Adam的融合设计

**Adam（Adaptive Moment Estimation）** 由Kingma和Ba在2015年提出，它融合了**动量法**（一阶矩）和**RMSProp**（二阶矩）的思想。

Adam同时维护两个变量：

1. **一阶矩估计 $m_t$** （动量的角色）：梯度的指数加权平均
   $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

2. **二阶矩估计 $v_t$** （自适应学习率的角色）：梯度平方的指数加权平均
   $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

### 5.2 偏差修正：解决初始化问题

在训练初期，$m_t$ 和 $v_t$ 被初始化为 $0$，导致估计值**严重偏小**。Adam通过**偏差修正**来解决这个问题：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

最终的参数更新为：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

**Adam的默认超参数**：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。这些默认值在大多数任务中表现稳定，这也是Adam广受欢迎的原因之一。

```python
def adam(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam优化器更新"""
    # 更新一阶矩和二阶矩
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    
    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # 参数更新
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return theta, m, v
```


## 六、AdamW：解耦权重衰减

### 6.1 Adam的隐藏问题：权重衰减的“耦合”

Adam虽然强大，但在**泛化能力**上常常不如带动量的SGD。研究发现，问题出在Adam对**权重衰减（Weight Decay）** 的处理上。

在标准Adam中，L2正则化（权重衰减）是通过**在梯度中直接加上 $\lambda \theta$ 项**来实现的：

$$g_t = \nabla_\theta J(\theta_t) + \lambda \theta_t$$

问题在于：**这个 $\lambda \theta_t$ 项也会被 $v_t$（梯度平方的指数加权平均）缩放**。

具体来说，Adam的参数更新为：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot (m_t + \lambda \theta_t)$$

权重衰减项 $\lambda \theta_t$ **除以了 $\sqrt{v_t}$** ——这意味着：
- 对于**历史梯度较大**的参数（$\sqrt{v_t}$ 大），权重衰减的效果被**削弱**
- 对于**历史梯度较小**的参数（$\sqrt{v_t}$ 小），权重衰减的效果被**放大**

这种**耦合**导致L2正则化在Adam中无法起到预期的效果。

### 6.2 AdamW的解决方案：解耦

**AdamW**由Loshchilov和Hutter在2017年提出，其核心思想非常直接：**将权重衰减从自适应更新中解耦出来**。

在AdamW中，权重衰减**不经过 $v_t$ 的缩放**，而是在参数更新**之后**独立施加：

**Step 1**：正常的Adam更新（不包含权重衰减）
$$\theta_{t+1}' = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

**Step 2**：独立施加权重衰减
$$\theta_{t+1} = \theta_{t+1}' - \eta \lambda \theta_{t+1}'$$

**关键差异**：
- **Adam**：$\theta_{t+1} = \theta_t - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$ ——权重衰减被自适应学习率缩放
- **AdamW**：$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ ——权重衰减独立于自适应机制

```python
def adamw(theta, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, 
          epsilon=1e-8, weight_decay=0.01):
    """AdamW优化器更新"""
    # 更新一阶矩和二阶矩（与Adam相同）
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    
    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # 参数更新（Adam风格，不含权重衰减）
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    # 解耦的权重衰减（关键区别！）
    theta = theta - lr * weight_decay * theta
    
    return theta, m, v
```

### 6.3 为什么AdamW更好？

**更有效的正则化**：由于权重衰减不再被自适应学习率缩放，它对所有参数施加**一致且可预测**的惩罚力度。

**更好的泛化性能**：实验表明，AdamW在大型Transformer架构（如BERT、GPT）上取得了**显著的性能提升**。

**更稳定的训练动态**：解耦使得训练过程更加可预测和稳定，这对大规模模型尤为重要。

**实际应用**：目前在**自然语言处理（NLP）领域**，AdamW已成为事实上的标准优化器。在计算机视觉（CV）领域，带动量的SGD仍然广泛使用。


## 七、总结：优化器的进化图谱

回顾整条进化路径，每一个优化器都是对前代**特定痛点**的精准回应：

| 优化器 | 核心贡献 | 解决的问题 |
|--------|---------|-----------|
| **BGD** | 全量数据梯度 | 基础框架 |
| **SGD** | 单样本梯度 | 计算效率 |
| **MBGD** | 小批量梯度 | 效率与稳定性的平衡 |
| **Momentum** | 速度变量 + 指数移动平均 | 峡谷地形的震荡 |
| **NAG** | 前瞻梯度计算 | 动量方向的“预见性” |
| **AdaGrad** | 参数级学习率 | 稀疏特征的处理 |
| **RMSProp** | 指数加权平均（二阶矩） | AdaGrad的学习率消亡 |
| **Adam** | 一阶矩 + 二阶矩 + 偏差修正 | 集动量与自适应于一身 |
| **AdamW** | 解耦权重衰减 | Adam中正则化失效 |

这条进化路线清晰地展示了深度学习中“**问题驱动**”的创新模式：每一个新算法的诞生，都是为了解决前一代在特定场景下的不足。

在实际选择优化器时，可以参考以下经验：

- **NLP任务（Transformer/BERT/GPT）** ：优先选择 **AdamW**
- **CV任务（ResNet等）** ：**带动量的SGD** 往往表现更好
- **强化学习**：**Adam** 使用较为普遍
- **推荐系统**：根据具体场景灵活选择

理解每种优化器背后的**数学原理与设计动机**，远比记住调参公式更重要——它能帮助你在面对新问题时，做出更明智的选择。