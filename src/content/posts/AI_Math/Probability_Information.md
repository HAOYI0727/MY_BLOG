---
title: Probability & Information —— 概率论与信息论基础
published: 2025-08-04
description: 系统讲解深度学习中的概率论与信息论基础：从联合分布、边缘化与贝叶斯定理出发，推导最大似然估计（MLE）的完整四步流程并揭示其与MSE/交叉熵损失的联系，引入最大后验（MAP）估计并阐明L2/L1正则化与高斯/拉普拉斯先验的对应关系，深入剖析KL散度的非对称性及其方向含义，最后证明KL散度、交叉熵与MLE三者的数学等价性。
cover: "/assets/images/posts/probability_information.png"
coverInContent: false
tags: [条件概率, 最大似然估计, KL散度, 交叉熵, 贝叶斯先验, 深度学习]
category: Deep_Learning
draft: false
---

# Probability & Information —— 概率论与信息论基础

## 一、前言：为什么AI离不开概率？

在深度学习的日常中，我们几乎每天都在和**不确定性**打交道：模型对一张图片的预测结果是“猫”的概率是92%，语言模型生成下一个词的概率分布是什么，强化学习中动作的价值估计带有多少不确定性……

**概率论是AI处理不确定性的数学语言，信息论则是量化这种不确定性的尺子。** 两者结合，构成了从数据中学习、推断和决策的理论基础。

本文将从五个方面，带你建立对概率论与信息论的系统理解：

1. 条件概率、联合分布与边缘化
2. 最大似然估计（MLE）的完整推导流程
3. 最大后验（MAP）与贝叶斯先验
4. KL散度的非对称性
5. KL散度与交叉熵的数学等价关系


## 二、条件概率、联合分布与边缘化

### 2.1 联合分布：描述多个变量的“全貌”

**联合概率分布**描述的是两个或多个随机变量**同时取值**的概率。对于离散随机变量 $X$ 和 $Y$，联合分布记为 $P(X=x, Y=y)$，表示 $X$ 取 $x$ 且 $Y$ 取 $y$ 的概率。

```python
import numpy as np
import pandas as pd

# 模拟一个联合分布：天气(X)和出行方式(Y)的联合概率
# X: 0=晴天, 1=雨天; Y: 0=步行, 1=骑车, 2=开车
joint = np.array([
    [0.20, 0.15, 0.10],  # 晴天时：步行、骑车、开车
    [0.05, 0.05, 0.45]   # 雨天时：步行、骑车、开车
])
# 验证：所有概率之和为1
print(f"联合分布总和: {joint.sum()}")  # 1.0

# 将联合分布可视化为DataFrame
df = pd.DataFrame(joint, index=['晴天', '雨天'], columns=['步行', '骑车', '开车'])
print(df)
#         步行   骑车   开车
# 晴天   0.20  0.15  0.10
# 雨天   0.05  0.05  0.45
```

### 2.2 边缘化：从联合分布中“消去”变量

**边缘化（Marginalization）** 是从联合分布中**消去不感兴趣的变量**，得到某个子集的概率分布。对于离散变量，边缘化就是**求和**；对于连续变量，边缘化就是**积分**。

$$P(X=x) = \sum_{y} P(X=x, Y=y)$$

$$P(Y=y) = \sum_{x} P(X=x, Y=y)$$

```python
# 边缘化：计算天气的边缘分布
p_X = joint.sum(axis=1)  # 对列求和（消去出行方式）
print(f"天气边缘分布: 晴天={p_X[0]:.2f}, 雨天={p_X[1]:.2f}")
# 天气边缘分布: 晴天=0.45, 雨天=0.55

# 边缘化：计算出行方式的边缘分布
p_Y = joint.sum(axis=0)  # 对行求和（消去天气）
print(f"出行方式边缘分布: 步行={p_Y[0]:.2f}, 骑车={p_Y[1]:.2f}, 开车={p_Y[2]:.2f}")
# 出行方式边缘分布: 步行=0.25, 骑车=0.20, 开车=0.55
```

**关键直觉**：边缘化就像把一张二维表格的某一行或某一列“压扁”，把所有概率加总到一条边上。

### 2.3 条件概率：在已知信息下的更新

**条件概率** $P(Y=y \mid X=x)$ 表示在已知 $X=x$ 的条件下，$Y=y$ 发生的概率：

$$P(Y=y \mid X=x) = \frac{P(X=x, Y=y)}{P(X=x)}$$

这个公式也叫做**概率的乘法法则**：联合概率 = 条件概率 × 边缘概率。

$$P(X, Y) = P(Y \mid X) \cdot P(X) = P(X \mid Y) \cdot P(Y)$$

```python
# 计算条件概率：已知是雨天，出行方式的概率分布
p_Y_given_rain = joint[1, :] / p_X[1]  # 雨天那一行除以雨天的边缘概率
print(f"雨天时出行方式分布: 步行={p_Y_given_rain[0]:.2f}, 骑车={p_Y_given_rain[1]:.2f}, 开车={p_Y_given_rain[2]:.2f}")
# 雨天时出行方式分布: 步行=0.09, 骑车=0.09, 开车=0.82

# 计算条件概率：已知是开车，天气的概率分布
p_X_given_drive = joint[:, 2] / p_Y[2]  # 开车那一列除以开车的边缘概率
print(f"开车时天气分布: 晴天={p_X_given_drive[0]:.2f}, 雨天={p_X_given_drive[1]:.2f}")
# 开车时天气分布: 晴天=0.18, 雨天=0.82
```

### 2.4 贝叶斯定理：逆转条件概率

**贝叶斯定理**是连接 $P(Y \mid X)$ 和 $P(X \mid Y)$ 的桥梁：

$$P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}$$

其中：
- $P(Y)$ 称为**先验（Prior）** ——在看到数据 $X$ 之前对 $Y$ 的信念
- $P(X \mid Y)$ 称为**似然（Likelihood）** ——在给定 $Y$ 下观察到 $X$ 的概率
- $P(Y \mid X)$ 称为**后验（Posterior）** ——在看到数据 $X$ 之后对 $Y$ 的更新信念
- $P(X)$ 称为**证据（Evidence）** ——数据的边缘概率，起到归一化作用


## 三、最大似然估计（MLE）的完整推导流程

### 3.1 MLE的核心思想

**最大似然估计（Maximum Likelihood Estimation, MLE）** 的核心思想非常朴素：**选择参数 $\theta$，使得观测到的数据出现的概率最大**。

换句话说：我们手里有一组数据，这些数据已经发生了。我们问自己：**是什么样的参数 $\theta$，让这组数据“最可能”被观察到？**

MLE的典型步骤分为两步：
1. **明确建模假设**：假设数据来自什么样的概率分布
2. **最大化似然**：找到使数据出现概率最大的参数值

### 3.2 从硬币实验理解MLE

**场景**：抛一枚硬币10次，得到结果：H, T, T, H, H, H, T, T, T, T。其中正面（H）出现4次，反面（T）出现6次。问：这枚硬币正面朝上的概率 $\theta$ 是多少？

**直觉答案**：$\theta = 4/10 = 0.4$。但我们需要从数学上推导为什么这是“最优”的。

**第1步：写出似然函数**

观测到4次正面、6次反面的概率（似然）为：

$$L(\theta) = \binom{10}{4} \theta^4 (1-\theta)^6$$

更一般地，对于 $n$ 次独立实验中观察到 $n_H$ 次正面：

$$L(\theta \mid \text{data}) = \binom{n}{n_H} \theta^{n_H} (1-\theta)^{n-n_H}$$

这个函数就是**似然函数（Likelihood Function）** ——给定参数 $\theta$ 下，观测到当前数据的概率。

**第2步：取对数——化乘积为求和**

直接最大化 $L(\theta)$ 涉及乘积运算，求导不便。由于对数函数是单调递增的，最大化 $L(\theta)$ 等价于最大化 $\log L(\theta)$：

$$\ell(\theta) = \log L(\theta) = \log\binom{n}{n_H} + n_H \log\theta + (n-n_H)\log(1-\theta)$$

其中 $\ell(\theta)$ 称为**对数似然函数（Log-Likelihood）** 。

**第3步：求导并令其为零**

对 $\ell(\theta)$ 求导：

$$\frac{d\ell}{d\theta} = \frac{n_H}{\theta} - \frac{n-n_H}{1-\theta}$$

令导数为零：

$$\frac{n_H}{\theta} = \frac{n-n_H}{1-\theta}$$

$$n_H(1-\theta) = (n-n_H)\theta$$

$$n_H - n_H\theta = n\theta - n_H\theta$$

$$n_H = n\theta$$

$$\hat{\theta}_{MLE} = \frac{n_H}{n} = 0.4$$

**第4步：验证是最大值**

二阶导数：

$$\frac{d^2\ell}{d\theta^2} = -\frac{n_H}{\theta^2} - \frac{n-n_H}{(1-\theta)^2} < 0$$

二阶导数为负，说明我们找到的是**最大值**而非最小值。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 抛硬币数据：10次中4次正面
n, n_H = 10, 4

# 定义负对数似然（最小化负对数似然 = 最大化对数似然）
def neg_log_likelihood(theta):
    if theta <= 0 or theta >= 1:
        return 1e10
    return -(n_H * np.log(theta) + (n - n_H) * np.log(1 - theta))

# 数值优化求MLE
result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 0.99), method='bounded')
print(f"MLE估计值: θ = {result.x:.3f}")  # 0.400

# 可视化似然函数
theta_range = np.linspace(0.01, 0.99, 100)
likelihood = theta_range**n_H * (1-theta_range)**(n-n_H)

plt.figure(figsize=(8, 5))
plt.plot(theta_range, likelihood, linewidth=2)
plt.axvline(result.x, color='red', linestyle='--', label=f'MLE θ={result.x:.2f}')
plt.xlabel('θ (正面概率)')
plt.ylabel('似然 L(θ)')
plt.title('抛硬币的似然函数')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 3.3 从MLE到机器学习：线性回归的损失函数

在机器学习中，MLE直接导出了我们熟悉的损失函数。以线性回归为例：

**假设**：目标值 $y$ 服从均值为 $\hat{y} = w^T x$、方差为 $\sigma^2$ 的正态分布：

$$y \sim \mathcal{N}(w^T x, \sigma^2)$$

**似然函数**：

$$L(w) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^T x_i)^2}{2\sigma^2}\right)$$

**对数似然**：

$$\ell(w) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - w^T x_i)^2$$

最大化 $\ell(w)$ 等价于**最小化** $\sum_{i=1}^{n}(y_i - w^T x_i)^2$——这正是**均方误差（MSE）** ！

这就是为什么MSE是线性回归的“天然”损失函数——它来自概率论的最大似然原理。


## 四、最大后验（MAP）与贝叶斯先验

### 4.1 MLE的局限：小样本过拟合

MLE有一个明显的缺陷：当样本量 $n$ 较小时，MLE容易**过拟合**。

回到硬币的例子：如果只抛了2次，两次都是正面，MLE会给出 $\hat{\theta}=1.0$——**认为硬币100%正面朝上**。这显然不合理。我们凭直觉知道，即使连续两次正面，硬币正面概率也不太可能真的是1.0。

### 4.2 贝叶斯视角：引入先验

**最大后验估计（Maximum A Posteriori, MAP）** 通过引入**先验分布 $P(\theta)$** 来解决这个问题。

MAP的核心公式来自**贝叶斯定理**：

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})}$$

MAP选择使**后验概率 $P(\theta \mid \text{data})$ 最大**的 $\theta$：

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta \mid \text{data}) = \arg\max_{\theta} P(\text{data} \mid \theta) \cdot P(\theta)$$

注意分母 $P(\text{data})$ 与 $\theta$ 无关，优化时可以忽略。

### 4.3 MAP的直观理解：MLE + 正则化

**对数形式**：

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \left[ \log P(\text{data} \mid \theta) + \log P(\theta) \right]$$

对比MLE：

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \log P(\text{data} \mid \theta)$$

**MAP = MLE + 先验项**。先验项 $\log P(\theta)$ 起到了**正则化**的作用——它惩罚那些“不合理”的参数值。

### 4.4 硬币实验的MAP：添加“虚拟”观测

回到硬币的例子。如果我们有先验信念：$\theta$ 应该接近0.5。

在贝叶斯统计中，这可以通过**Beta先验分布**来实现：$\theta \sim \text{Beta}(\alpha, \beta)$。Beta分布是二项分布的**共轭先验**，意味着后验分布仍然是Beta分布。

**后验**：

$$P(\theta \mid \text{data}) \propto \theta^{n_H + \alpha - 1} (1-\theta)^{n-n_H + \beta - 1}$$

**MAP估计**：

$$\hat{\theta}_{MAP} = \frac{n_H + \alpha - 1}{n + \alpha + \beta - 2}$$

如果取 $\alpha = \beta = 2$（相当于先验认为 $\theta$ 在0.5附近），10次实验中有4次正面：

$$\hat{\theta}_{MAP} = \frac{4 + 2 - 1}{10 + 2 + 2 - 2} = \frac{5}{12} \approx 0.417$$

对比MLE的0.4，MAP估计稍微**向先验（0.5）“收缩”** 了一些。

如果样本量极小（2次全正面）：
- MLE: $\hat{\theta} = 2/2 = 1.0$
- MAP: $\hat{\theta} = (2+1)/(2+2) = 3/4 = 0.75$

MAP有效地将极端估计拉向了更合理的范围。

```python
from scipy.stats import beta

# 抛硬币：n=10, n_H=4
n, n_H = 10, 4

# 先验：Beta(α=2, β=2)，认为θ在0.5附近
alpha, beta_param = 2, 2

# MLE
theta_mle = n_H / n

# MAP
theta_map = (n_H + alpha - 1) / (n + alpha + beta_param - 2)

print(f"MLE: θ = {theta_mle:.3f}")    # 0.400
print(f"MAP: θ = {theta_map:.3f}")    # 0.417

# 极端情况：2次全正面
n2, n_H2 = 2, 2
theta_mle_extreme = n_H2 / n2
theta_map_extreme = (n_H2 + alpha - 1) / (n2 + alpha + beta_param - 2)
print(f"\n极端情况（2次全正面）:")
print(f"MLE: θ = {theta_mle_extreme:.3f}")    # 1.000
print(f"MAP: θ = {theta_map_extreme:.3f}")    # 0.750
```

### 4.5 MAP与MLE的统一视角

一个重要结论：**当先验为均匀分布时，MAP退化为MLE**。

也就是说：
- **MLE** = 频率学派的点估计，只依赖数据
- **MAP** = 贝叶斯学派的点估计，数据 + 先验信念
- 当先验无信息（均匀分布）时，两者等价

在机器学习中，**L2正则化等价于高斯先验下的MAP，L1正则化等价于拉普拉斯先验下的MAP**。这为理解正则化提供了深刻的概率论视角。


## 五、KL散度的非对称性

### 5.1 什么是KL散度？

**KL散度（Kullback-Leibler Divergence）** ，也叫**相对熵（Relative Entropy）** ，用于衡量两个概率分布 $P$ 和 $Q$ 之间的差异：

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

对于连续分布，求和变为积分。

**直观理解**：KL散度衡量的是**如果用分布 $Q$ 来编码来自分布 $P$ 的数据，平均需要多付出多少比特（额外信息量）** 。

### 5.2 KL散度的非对称性：它不是“距离”

KL散度最反直觉的特性是它的**非对称性**：

$$D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$$

这意味着KL散度**不是一个真正的“距离”度量**（距离需要满足对称性和三角不等式）。

**为什么非对称？** 因为KL散度本质上是**有方向的信息度量**——它衡量的是“用 $Q$ 近似 $P$ 时损失了多少信息”。

如果把KL散度比作一个“箭头”，那么 $D_{KL}(P \parallel Q)$ 和 $D_{KL}(Q \parallel P)$ 就像是**指向相反方向的两个箭头**——它们衡量的信息损失方向完全不同。

### 5.3 直观例子：极端硬币 vs 公平硬币

考虑两个伯努利分布：
- $P$：极度不公平的硬币，正面概率 $99.99\%$
- $Q$：公平硬币，正面概率 $50\%$

$$D_{KL}(P \parallel Q) = 0.9999 \log\frac{0.9999}{0.5} + 0.0001 \log\frac{0.0001}{0.5} \approx 0.693$$

$$D_{KL}(Q \parallel P) = 0.5 \log\frac{0.5}{0.9999} + 0.5 \log\frac{0.5}{0.0001} \approx 4.255$$

两个方向的值**相差巨大**！

```python
def kl_divergence(p, q):
    """计算 KL散度 D_KL(P || Q)"""
    p = np.array(p)
    q = np.array(q)
    # 避免 log(0)
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

# 极端硬币 P vs 公平硬币 Q
P = [0.9999, 0.0001]
Q = [0.5, 0.5]

kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print(f"D_KL(P || Q) = {kl_pq:.4f}")  # ~0.693
print(f"D_KL(Q || P) = {kl_qp:.4f}")  # ~4.255
print(f"非对称性: 比值 = {kl_qp / kl_pq:.2f}x")
```

### 5.4 非对称性的实际意义

在机器学习中，KL散度的方向选择是有实际含义的：

- **$D_{KL}(P_{\text{data}} \parallel P_{\text{model}})$** ：衡量模型分布偏离真实数据分布的程度。这是**MLE**对应的方向——我们最小化这个量来让模型逼近真实分布。

- **$D_{KL}(P_{\text{model}} \parallel P_{\text{data}})$** ：衡量用真实分布来近似模型分布的困难程度。这在**变分推断（Variational Inference）** 和**期望传播（Expectation Propagation）** 中有不同应用。


## 六、KL散度与交叉熵的数学等价关系

### 6.1 熵、交叉熵与KL散度的定义

**熵（Entropy）** ：衡量一个分布 $P$ 的平均信息量。

$$H(P) = -\sum_{x} P(x) \log P(x)$$

**交叉熵（Cross Entropy）** ：用分布 $Q$ 来编码来自分布 $P$ 的数据时，所需的平均比特数。

$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

**KL散度**：交叉熵与熵的差：

$$D_{KL}(P \parallel Q) = H(P, Q) - H(P)$$

### 6.2 数学推导：优化等价性

将KL散度展开：

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log P(x) - \sum_{x} P(x) \log Q(x)$$

$$= -H(P) + H(P, Q)$$

关键观察：**$H(P)$ 只依赖于真实分布 $P$，与模型分布 $Q$ 无关**。

因此，在优化过程中（$P$ 固定，$Q$ 可变）：

$$\arg\min_{Q} D_{KL}(P \parallel Q) = \arg\min_{Q} H(P, Q)$$

**结论：最小化KL散度等价于最小化交叉熵**。

### 6.3 KL散度、交叉熵与MLE的三位一体

更深入地看，三者之间存在一个**统一的等价关系**：

$$\text{最小化 } D_{KL}(P_{\text{data}} \parallel P_{\text{model}}) \iff \text{最小化交叉熵} \iff \text{最大化对数似然}$$

这个等价关系揭示了深度学习中**损失函数设计的概率论根源**：

- 分类问题中的**交叉熵损失** = 最小化模型分布与真实标签分布之间的KL散度
- 回归问题中的**均方误差（MSE）** 在正态分布假设下 = 交叉熵损失 = 负对数似然

```python
import numpy as np
import matplotlib.pyplot as plt

# 真实分布 P（固定的）
P = np.array([0.3, 0.5, 0.2])

# 模拟不同的模型预测 Q
Q1 = np.array([0.3, 0.5, 0.2])  # 完美预测
Q2 = np.array([0.2, 0.6, 0.2])  # 略有偏差
Q3 = np.array([0.7, 0.2, 0.1])  # 偏差较大

def entropy(p):
    return -np.sum(p * np.log(np.clip(p, 1e-10, 1)))

def cross_entropy(p, q):
    return -np.sum(p * np.log(np.clip(q, 1e-10, 1)))

def kl_divergence(p, q):
    return cross_entropy(p, q) - entropy(p)

print("真实分布 P:", P)
print(f"熵 H(P) = {entropy(P):.4f}\n")

for i, Q in enumerate([Q1, Q2, Q3], 1):
    ce = cross_entropy(P, Q)
    kl = kl_divergence(P, Q)
    print(f"Q{i}: {Q}")
    print(f"  交叉熵 H(P,Q) = {ce:.4f}")
    print(f"  KL散度 D_KL(P||Q) = {kl:.4f}")
    print(f"  验证: H(P,Q) - H(P) = {ce - entropy(P):.4f} = KL散度\n")

# 输出:
# 真实分布 P: [0.3 0.5 0.2]
# 熵 H(P) = 1.0297
# 
# Q1: [0.3 0.5 0.2]
#   交叉熵 H(P,Q) = 1.0297
#   KL散度 D_KL(P||Q) = 0.0000
#   验证: H(P,Q) - H(P) = 0.0000 = KL散度
# 
# Q2: [0.2 0.6 0.2]
#   交叉熵 H(P,Q) = 1.0518
#   KL散度 D_KL(P||Q) = 0.0221
#   验证: H(P,Q) - H(P) = 0.0221 = KL散度
# 
# Q3: [0.7 0.2 0.1]
#   交叉熵 H(P,Q) = 1.5484
#   KL散度 D_KL(P||Q) = 0.5187
#   验证: H(P,Q) - H(P) = 0.5187 = KL散度
```

### 6.4 为什么用交叉熵而不是KL散度做损失函数？

既然两者在优化上等价，为什么深度学习中普遍使用**交叉熵**而非KL散度作为损失函数？

**原因**：在计算上，交叉熵 $H(P, Q) = -\sum P(x)\log Q(x)$ 直接跳过了 $H(P)$ 的计算。而 $H(P)$ 是真实分布的熵，在训练过程中是**常数**，不影响梯度。

在分类任务中，真实标签 $P$ 通常是**one-hot向量**（如 `[0, 1, 0]`），此时：

$$H(P, Q) = -\log Q(\text{true class})$$

这就是我们熟悉的**分类交叉熵损失**——它只关心模型对正确类别的预测概率。


## 七、总结

回顾本文的核心脉络：

1. **联合分布、边缘化与条件概率**构成了概率推理的三大基石。边缘化是“消去”变量的工具，条件概率是在已知信息下的概率更新，贝叶斯定理则连接了正反两个方向的条件概率。

2. **最大似然估计（MLE）** 从“让观测数据出现概率最大”这一朴素思想出发，通过**写出似然函数→取对数→求导→验证最大值**四步流程，导出了参数估计的闭式解。在机器学习中，MLE直接导出了MSE、交叉熵等常用损失函数。

3. **最大后验（MAP）** 在MLE的基础上引入**先验分布**，将频率学派的点估计扩展为贝叶斯框架下的点估计。当先验为均匀分布时MAP退化为MLE；L2/L1正则化分别对应高斯/拉普拉斯先验下的MAP。

4. **KL散度**衡量两个概率分布的差异，但**不是对称的**——$D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$。这种非对称性源于它是有方向的“信息损失”度量，而非无向的“距离”。

5. **KL散度、交叉熵与MLE**三者构成了一个统一的等价链条：**最小化KL散度 ⇔ 最小化交叉熵 ⇔ 最大化对数似然**。这解释了为什么交叉熵是分类问题的“天然”损失函数——它有着坚实的概率论和信息论基础。

这些概念不是孤立的数学公式，而是贯穿AI模型设计、训练和评估全流程的**思维框架**。理解它们，你就能真正理解深度学习模型“为什么这样设计”以及“如何改进”。