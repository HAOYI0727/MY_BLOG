---
title: Expectation-Maximization Algorithm —— EM算法
published: 2025-07-28
description: 系统讲解期望最大化（EM）算法的完整数学原理：从极大似然估计在隐变量存在时的困境出发，推导E步与M步的迭代框架；基于Jensen不等式证明ELBO证据下界与收敛性；通过二硬币模型与高斯混合模型（GMM）两个完整实例展示EM的具体计算流程；揭示K-Means是EM在硬分配下的特例这一深层联系。
cover: "/assets/images/posts/em.png"
coverInContent: false
tags: [EM算法, 期望最大化, 隐变量, 高斯混合模型, K-Means, 机器学习]
category: Machine_Learning
draft: false
---

# Expectation-Maximization Algorithm —— EM算法

## 引言

上一篇文章中，我们详细讨论了 **K-Means**——它通过“**分配—更新**”两步交替迭代，**将数据点划分为 $K$ 个簇**。K-Means 的实现简洁到令人惊叹：**分配步骤将每个点硬性划归最近的簇中心，更新步骤重新计算簇内均值**。这种“硬分配”策略虽然高效，却也暴露了一个根本性的局限：**它无法表达不确定性** —— 一个位于两个簇边界的数据点，要么属于 A，要么属于 B，没有中间状态。

现实数据中的不确定性无处不在。当两个高斯分布重叠时，一个观测点可能以 60% 的概率来自分布 A、40% 的概率来自分布 B——这种“**软归属**”信息在硬分配中被完全丢弃了。**EM 算法（Expectation-Maximization）** 正是为这种场景而生的通用框架，它由 Dempster、Laird 和 Rubin 于 1977 年正式提出，是处理**含隐变量概率模型**参数估计的最经典方法。

EM 算法的核心逻辑可以概括为两步迭代：**E 步（Expectation）在给定当前参数下计算隐变量的后验分布** —— 即“**软分配**”的概率；**M 步（Maximization）基于这些软分配权重重新估计模型参数**。这一框架的理论根基在于 **Jensen 不等式**所构造的 **ELBO（证据下界）** —— E 步让下界紧贴目标函数，M 步提升下界，从而间接提升似然函数。本文将用**二硬币模型**展示 EM 的直观计算流程，再用**高斯混合模型（GMM）** 展示其在**连续数据**上的完整推导，最后揭示一个贯穿始终的联系：**K-Means 是 EM 在“硬分配”极限下的特例**，对应于各向同性高斯混合模型在**方差趋于零**时的退化情形。

> [!note]
> 
> 读完本文，你将掌握 **EM 算法**的完整数学框架，并理解它为何是统计学和机器学习中最具影响力的方法论之一。至此，我们从 K-Means 的“**硬聚类**”走到了 GMM-EM 的“**软聚类**”，从确定性分配走向了概率建模。
> 
> 下一篇文章，我们将进入 **HMM（隐马尔可夫模型）** —— 它将 EM 的思想延伸到**序列数据**，引入时间维度的**隐变量**结构，完成第三阶段从“**独立同分布**”到“**时序依赖**”的最后一跃。

---

## 一、从极大似然估计到隐变量的困境

### 1.1 极大似然估计

在进入EM算法之前，先快速回顾一下**极大似然估计（Maximum Likelihood Estimation, MLE）** 。

假设我们有一个**概率模型**，其参数为 $\theta$，观测数据为 $X = \{x_1, x_2, ..., x_N\}$。MLE的目标是**找到一组参数 $\theta$，使得观测数据出现的概率最大**。

**似然函数**为：

$$
L(\theta) = p(X|\theta) = \prod_{i=1}^{N} p(x_i|\theta)
$$

为了计算方便，通常**取对数**，得到**对数似然函数**：

$$
\ell(\theta) = \log p(X|\theta) = \sum_{i=1}^{N} \log p(x_i|\theta)
$$

**MLE的估计值**为：

$$
\theta_{MLE} = \arg\max_{\theta} \ell(\theta)
$$

当模型**简单**时（如单个高斯分布、二项分布等），我们可以**直接对 $\ell(\theta)$ 求导并令导数为零**，得到**解析解**。

### 1.2 隐变量带来的困境

现在考虑一个更复杂的情况：数据中存在**隐变量** $Z = \{z_1, z_2, ..., z_N\}$，我们观测不到 $Z$，只能观测到 $X$。

此时，**似然函数**变成：

$$
p(X|\theta) = \sum_{Z} p(X, Z|\theta)
$$

或者对于**连续隐变量**：

$$
p(X|\theta) = \int p(X, Z|\theta) \, dZ
$$

**对数似然**为：

$$
\ell(\theta) = \log \sum_{Z} p(X, Z|\theta)
$$


注意看：**对数和求和的位置颠倒了**。

- 在**简单**情况下，**对 $\log p(X|\theta)$ 求导**，其中 $p(X|\theta)$ 是**乘积**形式，取对数后变成**求和**，求导很方便。
- 但现在，$\log$ 外面是一个**求和**（对隐变量求和），而这个求和又**无法移到 $\log$ 外面**。对数里面是求和，求导变得极其困难。

这就是隐变量带来的**核心困境：边际似然函数中的“和的对数”无法直接优化。**

### 1.3 一个直观的例子：二硬币模型

让我们用一个经典例子来感受这个困境。

假设有两枚硬币A和B，它们抛出正面的概率分别是 $\theta_A$ 和 $\theta_B$，但我们**不知道**。我们做了5轮实验，每轮随机选择一枚硬币，抛10次，记录正面次数。

**情况一：我们知道每轮用的是哪枚硬币**

| 轮次 | 硬币 | 正面次数 |
|------|------|----------|
| 1 | A | 5 |
| 2 | A | 9 |
| 3 | B | 4 |
| 4 | A | 4 |
| 5 | B | 5 |

这种情况下，没有隐变量。我们可以直接估计：

$$
\theta_A = \frac{5+9+4}{10+10+10} = \frac{18}{30} = 0.6
\quad \theta_B = \frac{4+5}{10+10} = \frac{9}{20} = 0.45
$$

**情况二：我们不知道每轮用的是哪枚硬币**

| 轮次 | 硬币 | 正面次数 |
|------|------|----------|
| 1 | ? | 5 |
| 2 | ? | 9 |
| 3 | ? | 4 |
| 4 | ? | 4 |
| 5 | ? | 5 |

现在，“每轮用的是哪枚硬币”就是**隐变量** $Z$。我们只知道观测数据 $X$（每轮的正面次数），却不知道 $Z$。

如果我们想用MLE估计 $\theta_A$ 和 $\theta_B$，需要最大化：$\ell(\theta_A, \theta_B) = \log \sum_{Z} p(X, Z|\theta_A, \theta_B)$

这个式子中的求和（对所有可能的硬币分配组合）使得直接优化变得极其困难。**这就是EM算法要解决的问题。**

---

## 二、EM算法的核心思想

### 2.1 EM算法的引入

回到二硬币模型。我们陷入了一个**循环困境**：

- 如果知道 **$\theta_A$ 和 $\theta_B$**，就能推断每轮用的是**哪枚硬币**（即估计**隐变量** $Z$）
- 如果知道每轮用的是**哪枚硬币**，就能估计 **$\theta_A$ 和 $\theta_B$**

EM算法的解决思路为**先随便猜一个，然后交替迭代**。
1. **随机初始化** $\theta_A$ 和 $\theta_B$（比如都设为0.5）
2. **E步**：基于当前的 $\theta$，计算隐变量 $Z$ 的**概率分布**（而不是硬性指定）
3. **M步**：基于E步得到的**隐变量分布**，用**极大似然估计**更新 $\theta$
4. 重复步骤2-3，直到**收敛**

### 2.2 E步：期望的计算

E步（**Expectation Step**，期望步骤）：**基于当前的参数估计 $\theta^{(t)}$**，计算**完整数据对数似然 $\log p(X, Z|\theta)$ 在隐变量后验分布 $p(Z|X, \theta^{(t)})$ 下的期望**。

数学上，E步构造如下函数：

$$
\boxed{Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X, \theta^{(t)}} \left[ \log p(X, Z|\theta) \right]}
$$

也就是说：

$$
Q(\theta, \theta^{(t)}) = \sum_{Z} p(Z|X, \theta^{(t)}) \log p(X, Z|\theta)
$$

**这个 $Q$ 函数是EM算法的核心**。它不是原始的似然函数，而是**完整数据对数似然的期望**。

### 2.3 M步：最大化的步骤

M步（**Maximization Step**，最大化步骤）：**寻找使 $Q$ 函数最大化的新参数**：

$$
\boxed{\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)})}
$$

**$Q$ 函数通常比原始的边际似然容易最大化**——在 $Q$ 函数中，隐变量 $Z$ 已经被**积分掉了**（通过期望），剩下的只是**关于 $\theta$ 的优化问题**，而且往往有**解析解**。

### 2.4 EM算法的完整流程

- 输入：观测数据：$X$；隐变量模型：$p(X, Z \mid \theta)$；初始参数：$\theta^{(0)}$
- 输出：“参数估计：$\theta$
- 算法流程：
  1. **初始化**：$t = 0$
  2. **重复以下步骤直至收敛**：
     a. **E步（期望步）** —— **计算联合对数似然在隐变量后验分布下的期望**：$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}} \left[ \log p(X, Z \mid \theta) \right]$
     b. **M步（最大化步）** —— **更新参数，使 $Q$ 函数最大化**：$\theta^{(t+1)} = \arg\max_{\theta} \, Q(\theta, \theta^{(t)})$
     c. **迭代更新**：$t = t + 1$
  3. **返回**：$\theta^{(t)}$

---

## 三、EM算法的数学推导

### 3.1 Jensen不等式：建立下界

EM算法的数学基础是 **Jensen不等式**。对于**凹函数** $f$（如 $\log$ 函数），Jensen不等式告诉我们：

$$
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
$$

现在，我们想**最大化** $\log p(X|\theta)$。**引入任意一个关于隐变量 $Z$ 的分布 $q(Z)$**：

$$
\log p(X|\theta) = \log \sum_Z p(X, Z|\theta) = \log \sum_Z q(Z) \cdot \frac{p(X, Z|\theta)}{q(Z)} = \log \mathbb{E}_{q(Z)} \left[ \frac{p(X, Z|\theta)}{q(Z)} \right]
$$

> [!note] 关于上述变换的说明
> 
> - $q(Z)$：**任意一个关于隐变量 $Z$ 的概率分布（满足 $\sum_Z q(Z)=1$，且 $q(Z)>0$）。**
> 
> - 数学技巧：原式是 $\sum_Z p(X,Z)$，乘以一个 $1 = \frac{q(Z)}{q(Z)}$，拆成了 $\sum_Z q(Z) \cdot \frac{p}{q}$。
> 
> - 目的：这一步是为了**把求和伪装成数学期望**。因为 $q(Z)$ 正好是一个**概率权重**，所以 $\sum_Z q(Z) \cdot [\cdot]$ 就等于“**在 $q(Z)$ 这个分布下，对括号里的值求期望**”。

由于 $\log$ 是**凹函数**，应用**Jensen不等式**：

$$
\log p(X|\theta) \geq \mathbb{E}_{q(Z)} \left[ \log \frac{p(X, Z|\theta)}{q(Z)} \right]
$$

右边这个量被称为 **ELBO（Evidence Lower Bound）** ——**证据下界**：

$$
\boxed{\text{ELBO}(q, \theta) = \mathbb{E}_{q(Z)} \left[ \log \frac{p(X, Z|\theta)}{q(Z)} \right]}
$$

### 3.2 ELBO与KL散度的关系

ELBO与对数似然之间有一个重要的关系：

$$
\log p(X|\theta) = \text{ELBO}(q, \theta) + \text{KL}(q(Z) \| p(Z|X, \theta))
$$

其中 $\text{KL}(q \| p)$ 是**KL散度**，衡量两个分布之间的差异，**恒为非负**。

> [!note] ELBO与对数似然之间的重要关系数学推导
> 
> 这个等式不是凭空冒出来的，它是**贝叶斯公式的简单代数变形**。
> 
> - **联合分布**可以拆解为：$p(X, Z|\theta) = p(Z|X, \theta) \cdot p(X|\theta)$
> 
> - 两边同时**取对数**：$\log p(X, Z|\theta) = \log p(Z|X, \theta) + \log p(X|\theta)$
> - 两边同时**在分布 $q(Z)$ 下取期望 $\mathbb{E}_q[\cdot]$**：$\mathbb{E}_q[\log p(X, Z|\theta)] = \mathbb{E}_q[\log p(Z|X, \theta)] + \log p(X|\theta)$
> 
> - 把 $\log p(X|\theta)$ 单独拎到左边，整理一下：$\log p(X|\theta) = \mathbb{E}_q[\log p(X, Z|\theta)] - \mathbb{E}_q[\log p(Z|X, \theta)]$
> 
> **在右边巧妙地加一项、减一项 $\mathbb{E}_q[\log q(Z)]$（恒等变形）**：
> 
> $$\log p(X|\theta) = \mathbb{E}_q[\log \frac{p(X, Z|\theta)}{q(Z)}] + \mathbb{E}_q[\log \frac{q(Z)}{p(Z|X, \theta)}]$$
> 
> **第一项**正好就是 **ELBO**，**第二项**正好就是 **$KL(q \parallel p)$**。
> 
> 于是，得到ELBO与对数似然之间的重要关系式：
> $$\boxed{\log p(X|\theta) = \text{ELBO}(q, \theta) + D_{KL}(q(Z) \parallel p(Z|X, \theta))}$$

因此：

$$ \log p(X|\theta) \geq \text{ELBO}(q, \theta) $$

**当且仅当 $q(Z) = p(Z|X, \theta)$ 时，KL散度为0，ELBO等于对数似然**。

### 3.3 EM算法的ELBO视角

从ELBO的视角来看，EM算法的两步实际上是在**交替优化**两个变量：

- **E步**：**固定 $\theta$**，选择 $q(Z) = p(Z|X, \theta)$，使**ELBO等于对数似然（KL散度为0）**
- **M步**：**固定 $q$**，**最大化ELBO**（等价于最大化 $Q$ 函数），提升对数似然

这个视角揭示了EM算法的本质：它通过**交替优化ELBO**，逐步**提升对数似然的下界**，从而**间接提升对数似然本身**。

### 3.4 收敛性证明

EM算法最优雅的性质是**它保证每次迭代后观测数据的对数似然不会下降**。即 $$\boxed{\log p(X|\theta^{(t)}) \leq \log p(X|\theta^{(t+1)})}$$

**证明概要**：

- 在**E步**中，选择 $q(Z) = p(Z|X, \theta^{(t)})$，此时**ELBO等于对数似然**：$\log p(X|\theta^{(t)}) = \text{ELBO}(q, \theta^{(t)})$

- 在**M步**中，**最大化 $Q$ 函数**（即ELBO）：$\text{ELBO}(q, \theta^{(t+1)}) \geq \text{ELBO}(q, \theta^{(t)})$

- 由ELBO的定义，对于任意 $\theta$：$\log p(X|\theta) \geq \text{ELBO}(q, \theta)$
 
- 特别地：$\log p(X|\theta^{(t+1)}) \geq \text{ELBO}(q, \theta^{(t+1)}) \geq \text{ELBO}(q, \theta^{(t)}) = \log p(X|\theta^{(t)})$

- 因此：$\log p(X|\theta^{(t+1)}) \geq \log p(X|\theta^{(t)})$

> [!important] 
> 
> EM算法**保证收敛到局部最优（似然函数的局部最大值），但不保证收敛到全局最优**。不同的初始值可能导致**不同的局部最优解**。

---

## 四、实例一：二硬币模型

### 4.1 问题设定

回到二硬币模型。我们有5轮实验，每轮抛10次，但不知道用的是哪枚硬币：

| 轮次 | 正面次数 |
|------|----------|
| 1 | 5 |
| 2 | 9 |
| 3 | 4 |
| 4 | 4 |
| 5 | 5 |

目标是估计 $\theta_A$ 和 $\theta_B$。

### 4.2 EM算法求解

- **初始化**：$\theta_A^{(0)} = 0.6$，$\theta_B^{(0)} = 0.5$

- **第1轮**：

1. **E步：对每一轮，计算它来自硬币A和硬币B的概率**。
   
> [!tip] 
> 
> 以第一轮（5正5反）为例：
> 
> - 如果是硬币A：$p = 0.6^5 \times 0.4^5 = 0.000796$
> - 如果是硬币B：$p = 0.5^5 \times 0.5^5 = 0.000977$
> 
> 归一化后：$$P(A|第一轮) = \frac{0.000796}{0.000796 + 0.000977} \approx 0.45 \quad P(B|第一轮) = \frac{0.000977}{0.000796 + 0.000977} \approx 0.55$$
> 
> 同理计算其他轮次。

2. **M步：基于E步得到的概率，计算每枚硬币的“期望正面次数”和“期望总次数”**。
   
> [!tip] 
> 
> 以硬币A为例，它“贡献”的**正面次数 = 各轮正面次数 × 该轮来自A的概率之和**。
> 
> 计算得到新参数：$$ \theta_A^{(1)} = \frac{21.3}{21.3 + 8.6} \approx 0.71 \quad \theta_B^{(1)} = \frac{11.7}{11.7 + 8.4} \approx 0.58 $$

- **迭代**：重复E步和M步，直到 $\theta$ 收敛。

### 4.3 二硬币模型的Python实现

```python
import numpy as np

def em_coin(observations, theta_A, theta_B, max_iter=100, tol=1e-6):
    """
    二硬币模型的EM算法
    observations: 每轮实验的正面次数（假设每轮抛10次）
    theta_A, theta_B: 初始参数
    """
    n_rounds = len(observations)
    n_flips = 10  # 每轮抛10次
    
    for t in range(max_iter):
        # E步：计算每轮来自A和B的概率
        prob_A = np.zeros(n_rounds)
        prob_B = np.zeros(n_rounds)
        
        for i, heads in enumerate(observations):
            tails = n_flips - heads
            # 在给定参数下，出现该观测结果的概率
            p_A = (theta_A ** heads) * ((1 - theta_A) ** tails)
            p_B = (theta_B ** heads) * ((1 - theta_B) ** tails)
            prob_A[i] = p_A / (p_A + p_B)
            prob_B[i] = p_B / (p_A + p_B)
        
        # M步：更新参数
        # 硬币A的期望正面次数和期望总次数
        expected_heads_A = np.sum(prob_A * observations)
        expected_total_A = np.sum(prob_A * n_flips)
        theta_A_new = expected_heads_A / expected_total_A
        
        expected_heads_B = np.sum(prob_B * observations)
        expected_total_B = np.sum(prob_B * n_flips)
        theta_B_new = expected_heads_B / expected_total_B
        
        # 检查收敛
        if abs(theta_A_new - theta_A) < tol and abs(theta_B_new - theta_B) < tol:
            print(f"收敛于第 {t+1} 轮迭代")
            break
        
        theta_A, theta_B = theta_A_new, theta_B_new
        print(f"第 {t+1} 轮: θA={theta_A:.4f}, θB={theta_B:.4f}")
    
    return theta_A, theta_B

# 运行EM算法
observations = [5, 9, 4, 4, 5]  # 每轮正面次数
theta_A, theta_B = em_coin(observations, 0.6, 0.5)
print(f"\n最终结果: θA={theta_A:.4f}, θB={theta_B:.4f}")
```

---

## 五、实例二：高斯混合模型（GMM）

### 5.1 高斯混合模型的定义

**高斯混合模型（Gaussian Mixture Model, GMM）** 是EM算法最经典的应用场景。

GMM假设**数据由 $K$ 个高斯分布混合而成**。每个数据点 $x_i$ 的生成过程是：

- **以概率 $\pi_k$ 选择一个高斯分量 $k$**
- **从该高斯分布 $\mathcal{N}(\mu_k, \Sigma_k)$ 中采样一个点**

其中 $\pi_k$ 是**混合系数**（$\sum_k \pi_k = 1$），$\mu_k$ 和 $\Sigma_k$ 是第 $k$ 个高斯分量的**均值和协方差矩阵**。

**隐变量**是每个数据点 $x_i$ **来自哪个高斯分量**——这个信息无法观测。

### 5.2 GMM的EM算法推导

**完整数据的对数似然**为：

$$
\log p(X, Z|\theta) = \sum_{i=1}^{N} \sum_{k=1}^{K} z_{ik} \left[ \log \pi_k + \log \mathcal{N}(x_i|\mu_k, \Sigma_k) \right]
$$

其中 $z_{ik} \in \{0, 1\}$ 表示**第 $i$ 个点是否属于第 $k$ 个分量**。

**E步**：计算**后验概率**

$$
\gamma_{ik} = p(z_{ik}=1|x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i|\mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(x_i|\mu_j^{(t)}, \Sigma_j^{(t)})}
$$

**M步**：更新参数

$$
N_k = \sum_{i=1}^{N} \gamma_{ik}，\quad \pi_k^{(t+1)} = \frac{N_k}{N}，\quad \mu_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} x_i
$$

$$
\Sigma_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^T
$$

### 5.3 GMM的Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM:
    """高斯混合模型（使用EM算法）"""
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pi = None      # 混合系数
        self.mu = None      # 均值
        self.sigma = None   # 协方差矩阵
        self.gamma = None   # 后验概率（责任）
    
    def _initialize(self, X):
        """初始化参数（使用K-Means++思想）"""
        n_samples, n_features = X.shape
        
        # 初始化均值：随机选择K个样本
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.mu = X[indices].copy()
        
        # 初始化协方差：单位矩阵
        self.sigma = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # 初始化混合系数：均匀分布
        self.pi = np.ones(self.n_components) / self.n_components
    
    def _e_step(self, X):
        """E步：计算后验概率"""
        n_samples = X.shape[0]
        self.gamma = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算每个点属于第k个分量的概率密度
            rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k])
            self.gamma[:, k] = self.pi[k] * rv.pdf(X)
        
        # 归一化
        self.gamma = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)
    
    def _m_step(self, X):
        """M步：更新参数"""
        n_samples, n_features = X.shape
        
        # 更新混合系数
        N_k = np.sum(self.gamma, axis=0)
        self.pi = N_k / n_samples
        
        # 更新均值
        for k in range(self.n_components):
            self.mu[k] = np.sum(self.gamma[:, k:k+1] * X, axis=0) / N_k[k]
        
        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.mu[k]
            weighted_diff = self.gamma[:, k:k+1] * diff
            self.sigma[k] = (weighted_diff.T @ diff) / N_k[k]
            # 加小值保证正定
            self.sigma[k] += 1e-6 * np.eye(n_features)
    
    def fit(self, X):
        """训练GMM"""
        self._initialize(X)
        
        prev_log_likelihood = -np.inf
        
        for t in range(self.max_iter):
            # E步
            self._e_step(X)
            
            # M步
            self._m_step(X)
            
            # 计算对数似然
            log_likelihood = 0
            for i in range(X.shape[0]):
                likelihood = 0
                for k in range(self.n_components):
                    rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k])
                    likelihood += self.pi[k] * rv.pdf(X[i])
                log_likelihood += np.log(likelihood + 1e-10)
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"GMM收敛于第 {t+1} 轮迭代")
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X):
        """预测每个点最可能属于的簇"""
        self._e_step(X)
        return np.argmax(self.gamma, axis=1)
    
    def predict_proba(self, X):
        """预测每个点属于每个簇的概率"""
        self._e_step(X)
        return self.gamma

# 生成数据
np.random.seed(42)
n_samples = 300

# 三个高斯分量
X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples // 3)
X2 = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], n_samples // 3)
X3 = np.random.multivariate_normal([0, 5], [[0.5, 0], [0, 0.5]], n_samples // 3)
X = np.vstack([X1, X2, X3])

# 训练GMM
gmm = GMM(n_components=3, max_iter=100)
gmm.fit(X)
labels = gmm.predict(X)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c='red', marker='X', s=200, label='簇中心')
plt.title('GMM聚类结果（EM算法）')
plt.legend()
plt.axis('equal')
plt.show()
```

---

## 六、EM视角下的K-Means与GMM

### 6.1 K-Means是EM的硬版本

在后续的文章中，我们会详细介绍K-Means算法。我们可以从EM的视角审视它 —— **K-Means可以看作是EM算法的一个特例**。

| EM框架 | K-Means | GMM-EM |
|---------|---------|--------|
| **隐变量** | 簇分配 $c_i$ | 簇分配 $c_i$ |
| **参数** | **簇中心 $\mu_k$** | **均值 $\mu_k$、协方差 $\Sigma_k$、混合系数 $\pi_k$** |
| **E步** | **硬分配**：每个点100%属于最近的簇 | **软分配**：每个点以概率属于各簇 |
| **M步** | 用簇内点的**均值**更新中心 | 用**加权平均**更新所有参数 |

**关键区别**：K-Means做的是**硬分配（Hard Assignment）** —— 每个数据点**只能属于一个簇**；而GMM-EM做的是**软分配（Soft Assignment）** —— 每个数据点可以**以一定的概率属于多个簇**。

从数学上看，K-Means等价于GMM的一个**特殊情形**：当**所有高斯分量的协方差矩阵 **$\Sigma_k = \sigma^2 I$**（各向同性且相同）且 $\sigma^2 \to 0$** 时，后验概率 $\gamma_{ik}$ 退化为 **one-hot** 向量，GMM-EM 退化为 **K-Means**。

### 6.2 硬分类 vs 软分类

| 对比维度 | K-Means（硬分类） | GMM（软分类） |
|---------|------------------|--------------|
| **分配方式** | 每个点属于**一个**簇 | 每个点以**概率**属于各簇 |
| **不确定性表达** | **无法表达** | **概率分布表达不确定性** |
| **簇形状** | 仅限**球形** | 可以处理**任意椭圆形状** |
| **簇大小** | 假设大小相近 | 可以处理**不同大小**的簇 |
| **对重叠数据的处理** | 差 | **好**（概率分配自然处理重叠） |
| **计算复杂度** | 低 | **较高** |

### 6.3 从硬到软：软分类的优势

在真实数据中，簇与簇之间往往存在**重叠**。一个数据点可能**部分属于**簇A、**部分属于**簇B——比如一个身高175cm的人，既有可能是男生也有可能是女生。

**K-Means的硬分配无法表达这种不确定性**。而**GMM的软分配**通过**概率**自然地表达了这种**不确定性**，这种软分配不仅**更符合真实情况**，还**避免了硬分配中“边界点”的尴尬** —— **边界点**不会被迫完全属于某一个簇，而是**以概率分布在多个簇之间**。

---

> [!note] 总结
> 
> | 概念 | 核心内容 |
> |------|---------|
> | **隐变量困境** | **边际似然** $\log \sum_Z p(X,Z\|\theta)$ 中的“**和的对数**”难以直接优化 |
> | **E步** | 计算 $Q(\theta,\theta^{(t)}) = \mathbb{E}_{Z\|X,\theta^{(t)}}[\log p(X,Z\|\theta)]$ |
> | **M步** | $\theta^{(t+1)} = \arg\max_\theta Q(\theta,\theta^{(t)})$ |
> | **ELBO** | $\log p(X|\theta) \geq \mathbb{E}_q[\log(p(X,Z|\theta)/q(Z))]$，**证据下界** |
> | **收敛性** | EM保证每一步似然函数**不下降**，但只能收敛到**局部最优** |
> | **K-Means** | EM的**硬分配**特例（$\Sigma_k = \sigma^2 I, \sigma^2 \to 0$） |
> | **GMM** | EM的**软分配**经典应用，可处理任意椭圆形状的簇 |
> 
> 核心要点回顾
> 
> 1. **隐变量的困境**：当模型包含**隐变量**时，**边际似然 $\log \sum_Z p(X,Z|\theta)$ 中的“和的对数”无法直接优化**。EM算法通过**迭代**的方式绕开了这个困难。
> 2. **E步与M步**：E步计算完整数据**对数似然**在**隐变量后验分布**下的**期望**（即 $Q$ 函数），M步**最大化**这个期望来更新参数。两步**交替进行**，逐步**逼近最优解**。
> 3. **ELBO与收敛性**：EM算法的数学基础是**Jensen不等式**，它保证**ELBO**始终是边际对数似然的**下界**。每一步**M步都在提升ELBO**，从而**间接提升对数似然**。EM算法**保证收敛**，但只能收敛到**局部最优**。
> 4. **K-Means是EM的特例**：K-Means相当于GMM-EM在**硬分配**极限下的退化版本。K-Means做**硬分类**（每个点100%属于一个簇），GMM做**软分类**（每个点以概率属于各簇）。软分类能更好地处理**簇重叠**和**不确定性**。
> 5. **EM算法的本质**：EM与其说是一种具体的算法，不如说是一种**解决问题的框架**。它不限定具体的模型——可以是**GMM、HMM、LDA**等——只要模型包含**隐变量**，都可以用**EM框架**来求解。