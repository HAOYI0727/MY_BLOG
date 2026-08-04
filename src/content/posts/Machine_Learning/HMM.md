---
title: Hidden Markov Model (HMM) —— 隐马尔可夫模型
published: 2025-07-30
description: 系统讲解隐马尔可夫模型（HMM）的完整数学原理与三大核心算法：从马尔可夫链到双重随机过程的演进出发，定义HMM的五元组参数（Q, V, π, A, B）；详细推导估值问题、解码问题和学习问题；并通过词性标注、语音识别等经典应用场景展示HMM的实践价值。
cover: "/assets/images/posts/hmm.png"
coverInContent: false
tags: [隐马尔可夫模型, HMM, 前向算法, 序列建模, 机器学习]
category: Machine_Learning
draft: false
---

# Hidden Markov Model (HMM) —— 隐马尔可夫模型

## 引言

上一篇文章中，我们系统推导了 **EM 算法**的完整数学框架——它通过“**E 步计算隐变量后验、M 步最大化期望似然**”的迭代策略，为含**隐变量**的概率模型提供了**通用的参数估计方法论**。EM 算法是一个“**框架**”，而非具体模型。现在，我们将走进这个框架最经典、最具影响力的具体实现之一 —— **隐马尔可夫模型（Hidden Markov Model, HMM）**。

HMM 所面对的数据与前面所有模型都不同。线性回归、逻辑回归、SVM、决策树乃至 GMM，处理的都是**独立同分布**的样本 —— **样本之间没有顺序依赖**。但现实世界中大量数据是以**序列**形式存在的：语音是一串声学帧，文本是一串单词，DNA 是一串碱基。这些序列的共同特征在于**时序依赖 —— 当前位置的状态与前一位置的状态密切相关**。HMM 正是为捕捉这种依赖而设计的**双重随机过程**：底层是一个不可观测的**马尔可夫链**（状态随时间转移），上层是在每个状态下生成**观测值**的发射过程。

本文将从 HMM 的**五元组参数** $(Q, V, \pi, A, B)$ 出发，系统讲解其**三大核心问题**及对应的经典算法：**估值问题**（前向算法，计算观测序列概率）、**解码问题**（维特比算法，推断最优隐藏状态序列）、**学习问题**（Baum-Welch 算法，即 EM 在 HMM 中的具体实现）。这三个问题覆盖了从“**已知模型做推断**”到“**从数据中学习模型**”的完整链路，也是语音识别、词性标注、中文分词等应用的算法基石。

> [!note]
> 
> 至此，我们走完了本系列 15 篇文章的完整旅程——从**线性回归**的解析基石出发，历经**逻辑回归、KNN、朴素贝叶斯**的判别与生成范式，**SVM 与核技巧**的几何极致，**决策树与集成学习**的树模型家族（随机森林、AdaBoost、GBM、XGBoost、LightGBM），再到**无监督与概率模型**的降维（PCA）、聚类（K-Means）、隐变量推断（EM）与时序建模（HMM）。
> 
> 这条路径从“**有标签**”走向“**无标签**”，从“**独立同分布**”走向“**时序依赖**”，从“**参数化假设**”走向“**非参数灵活**”，构建了一个覆盖机器学习核心领域的全景认知框架。

---

## 一、从马尔可夫链到隐马尔可夫模型

### 1.1 马尔可夫链

在进入HMM之前，先回顾一下**马尔可夫链（Markov Chain）** 。

马尔可夫链描述的是一个**状态序列**，其中每个状态都**直接可见**。它的核心假设是**马尔可夫性（Markov Property）：未来状态只依赖于当前状态，与过去无关。**

用数学语言表达：

$$
P(q_{t+1} \mid q_t, q_{t-1}, ..., q_1) = P(q_{t+1} \mid q_t)
$$

一个马尔可夫链由**三个要素**决定：

- **状态集合** $Q = \{q_1, q_2, ..., q_N\}$
- **初始状态分布** $\pi = \{\pi_i\}$，其中 $\pi_i = P(q_1 = i)$
- **状态转移矩阵** $A = \{a_{ij}\}$，其中 $a_{ij} = P(q_{t+1} = j \mid q_t = i)$

### 1.2 隐马尔可夫模型

HMM在马尔可夫链的基础上增加了一个层次：**状态本身是不可见的（隐藏的）** ，只能观察到**与状态相关的观测值**。

HMM是一个**双重随机过程**：

1. **底层**：一个**隐藏的马尔可夫链**，**状态不可见**，按**转移概率**在状态间**跳转**
2. **上层**：在每个隐藏状态下，按**发射概率**生成一个**可观测的输出**

一个经典的例子：**天气与海藻**。我们看不见天气（隐藏状态：晴天、多云、雨天），但能看见海藻的湿度（观测值：干、湿润、潮湿）。海藻的湿度由当天的天气决定。

---

## 二、HMM的五元组参数

### 2.1 形式化定义

一个HMM由五个要素构成：

$$
\boxed{\lambda = (Q, V, \pi, A, B)}
$$

其中：

1. **隐藏状态集合** $Q = \{q_1, q_2, ..., q_N\}$ —— **所有可能的状态**。$N$ 是状态的数量。

2. **观测集合** $V = \{v_1, v_2, ..., v_M\}$ —— **所有可能的观测值**。$M$ 是观测值的数量。注意**状态数 $N$ 和观测数 $M$ 不一定相等**。

3. **初始状态概率分布** $\pi = \{\pi_i\}$ —— $\pi_i = P(q_1 = i), \quad 1 \leq i \leq N$，表示**在初始时刻 $t=1$ 时处于状态 $i$ 的概率**。

4. **状态转移概率矩阵** $A = \{a_{ij}\}$ —— $a_{ij} = P(q_{t+1} = j \mid q_t = i), \quad 1 \leq i, j \leq N$，表示**从状态 $i$ 转移到状态 $j$ 的概率**。每一行之**和为1**：$\sum_{j=1}^{N} a_{ij} = 1$。

5. **观测概率矩阵（发射概率）** $B = \{b_j(k)\}$ —— $b_j(k) = P(o_t = v_k \mid q_t = j), \quad 1 \leq j \leq N, \quad 1 \leq k \leq M$，表示**在状态 $j$ 下观测到 $v_k$ 的概率**。每一行之**和为1**：$\sum_{k=1}^{M} b_j(k) = 1$。

### 2.2 生成过程

HMM生成一个观测序列的过程如下：

1. **根据初始分布 $\pi$ 选择初始隐藏状态 $q_1$**
2. **根据发射概率 $B$，从状态 $q_1$ 生成观测值 $o_1$**
3. **根据转移概率 $A$，从状态 $q_t$ 转移到 $q_{t+1}$**
4. **重复步骤2-3，直到生成完整的观测序列 $O = (o_1, o_2, ..., o_T)$**

**代码实现**：

```python
import numpy as np

class HMM:
    """隐马尔可夫模型的基础类"""
    
    def __init__(self, n_states, n_obs):
        """
        n_states: 隐藏状态数量 N
        n_obs: 观测值数量 M
        """
        self.n_states = n_states
        self.n_obs = n_obs
        
        # 初始化参数（随机）
        self.pi = np.ones(n_states) / n_states  # 初始状态分布
        self.A = np.ones((n_states, n_states)) / n_states  # 转移矩阵
        self.B = np.ones((n_states, n_obs)) / n_obs  # 发射矩阵
        
    def generate(self, T):
        """生成一个长度为T的观测序列"""
        states = np.zeros(T, dtype=int)
        observations = np.zeros(T, dtype=int)
        
        # 步骤1：选择初始状态
        states[0] = np.random.choice(self.n_states, p=self.pi)
        # 步骤2：生成第一个观测
        observations[0] = np.random.choice(self.n_obs, p=self.B[states[0]])
        
        # 步骤3-4：迭代生成
        for t in range(1, T):
            states[t] = np.random.choice(self.n_states, p=self.A[states[t-1]])
            observations[t] = np.random.choice(self.n_obs, p=self.B[states[t]])
        
        return states, observations

# 示例：创建一个简单的HMM并生成序列
hmm = HMM(n_states=3, n_obs=4)
states, obs = hmm.generate(T=10)
print(f"隐藏状态序列: {states}")
print(f"观测序列: {obs}")
```

---

## 三、HMM的三大经典问题

HMM有三个经典问题，覆盖了从“已知模型做推断”到“从数据中学习模型”的完整链路。

### 3.1 问题一：估值问题（Evaluation）

**问题描述**：给定模型 $\lambda = (A, B, \pi)$ 和观测序列 $O = (o_1, o_2, ..., o_T)$，计算观测序列出现的概率 $P(O \mid \lambda)$。

**应用场景**：**语音识别**中，有**多个HMM模型**（每个词一个模型），给定一段**语音信号（观测序列）**，计算它在每个模型下的概率，**概率最大的模型对应的词就是识别结果**。

**解决算法**：

**1. 直接计算法（不可行）**

最朴素的想法是**穷举所有可能的状态序列** $I = (i_1, i_2, ..., i_T)$，计算**联合概率再求和**：

$$
P(O \mid \lambda) = \sum_{I} P(O, I \mid \lambda) = \sum_{I} \pi_{i_1} \prod_{t=2}^{T} a_{i_{t-1}i_t} \prod_{t=1}^{T} b_{i_t}(o_t)
$$

但时间复杂度是 **$O(T N^T)$** —— 状态数 $N$ 的 $T$ 次方，完全不可行。

**2. 前向算法（Forward Algorithm）**

前向算法用**动态规划**将复杂度降到 $O(T N^2)$。

**定义前向变量 $\alpha_t(i)$：在时刻 $t$，隐藏状态为 $i$，且观测到 $o_1, o_2, ..., o_t$ 的联合概率**：

$$
\alpha_t(i) = P(o_1, o_2, ..., o_t, q_t = i \mid \lambda)
$$

**初始化**（$t=1$）：

$$
\alpha_1(i) = \pi_i \cdot b_i(o_1), \quad 1 \leq i \leq N
$$

**递推**（$t = 2, 3, ..., T$）：

$$
\alpha_t(j) = \left[ \sum_{i=1}^{N} \alpha_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(o_t), \quad 1 \leq j \leq N
$$

**终止**：

$$
P(O \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)
$$

**代码实现**：

```python
def forward(self, obs):
    """前向算法：计算 P(O | lambda)"""
    T = len(obs)
    alpha = np.zeros((T, self.n_states))
    
    # 初始化
    for i in range(self.n_states):
        alpha[0, i] = self.pi[i] * self.B[i, obs[0]]
    
    # 递推
    for t in range(1, T):
        for j in range(self.n_states):
            alpha[t, j] = np.sum(alpha[t-1, :] * self.A[:, j]) * self.B[j, obs[t]]
    
    # 终止
    return np.sum(alpha[T-1, :])

# 将方法添加到HMM类中
HMM.forward = forward

# 测试
hmm = HMM(n_states=3, n_obs=4)
# 设置已知参数（便于验证）
hmm.pi = np.array([0.5, 0.3, 0.2])
hmm.A = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
hmm.B = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.4, 0.3, 0.2], [0.2, 0.2, 0.3, 0.3]])
obs = [0, 2, 1, 3]
prob = hmm.forward(obs)
print(f"观测序列 {obs} 的概率: {prob:.6f}")
```

**3. 后向算法（Backward Algorithm）**

后向算法是**从后往前递推**。

**定义后向变量 $\beta_t(i)$：在时刻 $t$ 状态为 $i$ 的条件下，观测到 $o_{t+1}, o_{t+2}, ..., o_T$ 的概率**：

$$
\beta_t(i) = P(o_{t+1}, o_{t+2}, ..., o_T \mid q_t = i, \lambda)
$$

**初始化**（$t=T$）：

$$
\beta_T(i) = 1, \quad 1 \leq i \leq N
$$

**递推**（$t = T-1, T-2, ..., 1$）：

$$
\beta_t(i) = \sum_{j=1}^{N} a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)
$$

**终止**：

$$
P(O \mid \lambda) = \sum_{i=1}^{N} \pi_i \cdot b_i(o_1) \cdot \beta_1(i)
$$

前向和后向算法可以结合使用，在Baum-Welch算法中两者缺一不可。

**代码实现**：

```python
def backward(self, obs):
    """后向算法"""
    T = len(obs)
    beta = np.zeros((T, self.n_states))
    
    # 初始化
    beta[T-1, :] = 1
    
    # 递推（从后往前）
    for t in range(T-2, -1, -1):
        for i in range(self.n_states):
            beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs[t+1]] * beta[t+1, :])
    
    # 终止
    return np.sum(self.pi * self.B[:, obs[0]] * beta[0, :])

HMM.backward = backward
```

### 3.2 问题二：解码问题（Decoding）

**问题描述**：给定模型 $\lambda = (A, B, \pi)$ 和观测序列 $O = (o_1, o_2, ..., o_T)$，找到**最有可能产生该观测序列的隐藏状态序列** $I^* = (i_1^*, i_2^*, ..., i_T^*)$。

$$
I^* = \arg\max_{I} P(I \mid O, \lambda)
$$

**应用场景**：**词性标注**中，给定一个句子（**观测序列是单词**），找出每个单词最可能的词性（**隐藏状态是词性标签**）。

**解决算法**：

**1. 维特比算法（Viterbi Algorithm）**

维特比算法是**动态规划**在HMM解码问题上的经典应用。它的核心思想是**在时刻 $t$ 到达状态 $i$ 的所有路径中，只保留概率最大的一条**——因为最优的全局路径一定由**局部最优路径**组成。

**定义 $\delta_t(i)$：在时刻 $t$，到达状态 $i$ 的所有路径中，概率最大的那条路径的概率**：

$$
\delta_t(i) = \max_{i_1, i_2, ..., i_{t-1}} P(i_1, i_2, ..., i_{t-1}, i_t = i, o_1, ..., o_t \mid \lambda)
$$

同时**定义 $\psi_t(i)$：记录到达状态 $i$ 的最优路径中，前一时刻的状态是什么（用于回溯）**。

**初始化**（$t=1$）：

$$
\delta_1(i) = \pi_i \cdot b_i(o_1), \quad \psi_1(i) = 0
$$

**递推**（$t = 2, 3, ..., T$）：

$$
\delta_t(j) = \max_{1 \leq i \leq N} \left[ \delta_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(o_t), \quad \psi_t(j) = \arg\max_{1 \leq i \leq N} \left[ \delta_{t-1}(i) \cdot a_{ij} \right]
$$

**终止**：

$$
P^* = \max_{1 \leq i \leq N} \delta_T(i), \quad i_T^* = \arg\max_{1 \leq i \leq N} \delta_T(i)
$$

**回溯**（从 $T-1$ 到 $1$）：

$$
i_t^* = \psi_{t+1}(i_{t+1}^*)
$$

**代码实现**：

```python
def viterbi(self, obs):
    """维特比算法：解码最可能的隐藏状态序列"""
    T = len(obs)
    delta = np.zeros((T, self.n_states))
    psi = np.zeros((T, self.n_states), dtype=int)
    
    # 初始化
    for i in range(self.n_states):
        delta[0, i] = self.pi[i] * self.B[i, obs[0]]
        psi[0, i] = 0
    
    # 递推
    for t in range(1, T):
        for j in range(self.n_states):
            # 找到使 delta[t-1][i] * A[i][j] 最大的 i
            max_prob = -1
            max_idx = 0
            for i in range(self.n_states):
                prob = delta[t-1, i] * self.A[i, j]
                if prob > max_prob:
                    max_prob = prob
                    max_idx = i
            delta[t, j] = max_prob * self.B[j, obs[t]]
            psi[t, j] = max_idx
    
    # 终止
    best_last_state = np.argmax(delta[T-1, :])
    best_path = [best_last_state]
    
    # 回溯
    for t in range(T-1, 0, -1):
        best_path.insert(0, psi[t, best_path[0]])
    
    return best_path

HMM.viterbi = viterbi

# 测试维特比算法
obs = [0, 2, 1, 3]
best_states = hmm.viterbi(obs)
print(f"观测序列: {obs}")
print(f"最可能的状态序列: {best_states}")
```

---

### 3.3 问题三：学习问题（Learning）

**问题描述**：给定观测序列 $O = (o_1, o_2, ..., o_T)$（或多个观测序列），**估计模型参数 $\lambda = (A, B, \pi)$，使得 $P(O \mid \lambda)$ 最大**。

**应用场景**：用大量已标注或未标注的语料，训练一个HMM用于**词性标注**。

**解决算法**：

**1. 有监督学习：极大似然估计**

如果训练数据中**既有观测序列，也有对应的隐藏状态序列**（即有标注数据），学习问题很简单 —— **直接用频率估计概率**：

$$
\pi_i = \frac{\text{以状态 } i \text{ 开头的序列数}}{\text{总序列数}}
$$

$$
a_{ij} = \frac{\text{从状态 } i \text{ 转移到状态 } j \text{ 的次数}}{\text{从状态 } i \text{ 转移的总次数}}
$$

$$
b_j(k) = \frac{\text{在状态 } j \text{ 下观测到 } v_k \text{ 的次数}}{\text{在状态 } j \text{ 下观测的总次数}}
$$

**2. 无监督学习：Baum-Welch算法（EM在HMM中的实现）**

但在大多数情况下，我们**只有观测序列，没有隐藏状态序列**。这就是**EM算法**大显身手的地方——**Baum-Welch算法**正是EM算法在HMM中的具体实现。

Baum-Welch算法通过**迭代**的方式，**在E步“猜测”隐藏状态，在M步更新模型参数**。

**E步：计算期望**

利用**前向变量 $\alpha_t(i)$ 和后向变量 $\beta_t(i)$**，定义两个关键的**期望统计量**：

**$\gamma_t(i)$：在时刻 $t$ 处于状态 $i$ 的概率（给定整个观测序列）**：

$$
\gamma_t(i) = P(q_t = i \mid O, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j) \cdot \beta_t(j)}
$$

**$\xi_t(i, j)$：在时刻 $t$ 处于状态 $i$、时刻 $t+1$ 处于状态 $j$ 的概率**：

$$
\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid O, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{\sum_{i'=1}^{N} \sum_{j'=1}^{N} \alpha_t(i') \cdot a_{i'j'} \cdot b_{j'}(o_{t+1}) \cdot \beta_{t+1}(j')}
$$

**M步：最大化更新参数**

有了这些期望统计量，就可以**更新模型参数**：

$$
\pi_i^{(new)} = \gamma_1(i)
$$

$$
a_{ij}^{(new)} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

$$
b_j(k)^{(new)} = \frac{\sum_{t=1}^{T} \gamma_t(j) \cdot \mathbb{I}(o_t = v_k)}{\sum_{t=1}^{T} \gamma_t(j)}
$$

**代码实现**：

```python
def baum_welch(self, obs, n_iter=100, tol=1e-6):
    """
    Baum-Welch算法：无监督学习HMM参数
    这是EM算法在HMM中的具体实现
    """
    T = len(obs)
    
    for iteration in range(n_iter):
        # ---- E步：计算前向和后向概率 ----
        # 前向
        alpha = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self.B[i, obs[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1, :] * self.A[:, j]) * self.B[j, obs[t]]
        
        # 后向
        beta = np.zeros((T, self.n_states))
        beta[T-1, :] = 1
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs[t+1]] * beta[t+1, :])
        
        # 计算gamma和xi
        gamma = np.zeros((T, self.n_states))
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T):
            denominator = np.sum(alpha[t, :] * beta[t, :])
            for i in range(self.n_states):
                gamma[t, i] = alpha[t, i] * beta[t, i] / denominator
        
        for t in range(T-1):
            denominator = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    denominator += alpha[t, i] * self.A[i, j] * self.B[j, obs[t+1]] * beta[t+1, j]
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                   self.B[j, obs[t+1]] * beta[t+1, j]) / denominator
        
        # ---- M步：更新参数 ----
        # 更新pi
        pi_new = gamma[0, :]
        
        # 更新A
        A_new = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                A_new[i, j] = numerator / denominator if denominator > 0 else 0
        
        # 更新B
        B_new = np.zeros((self.n_states, self.n_obs))
        for j in range(self.n_states):
            for k in range(self.n_obs):
                numerator = np.sum(gamma[t, j] for t in range(T) if obs[t] == k)
                denominator = np.sum(gamma[:, j])
                B_new[j, k] = numerator / denominator if denominator > 0 else 0
        
        # 检查收敛
        if (np.max(np.abs(self.pi - pi_new)) < tol and 
            np.max(np.abs(self.A - A_new)) < tol and 
            np.max(np.abs(self.B - B_new)) < tol):
            print(f"Baum-Welch收敛于第 {iteration+1} 轮迭代")
            break
        
        self.pi, self.A, self.B = pi_new, A_new, B_new
    
    return self

HMM.baum_welch = baum_welch

# 测试：从一个随机HMM生成数据，然后用Baum-Welch恢复参数
true_hmm = HMM(n_states=3, n_obs=4)
true_hmm.pi = np.array([0.6, 0.3, 0.1])
true_hmm.A = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
true_hmm.B = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.5, 0.3, 0.1], [0.1, 0.1, 0.4, 0.4]])

# 生成观测序列
_, obs_seq = true_hmm.generate(T=500)

# 用Baum-Welch学习（从随机初始化开始）
learnt_hmm = HMM(n_states=3, n_obs=4)
learnt_hmm.baum_welch(obs_seq, n_iter=50)

print("\n学习后的参数与真实参数的对比：")
print(f"真实 pi: {true_hmm.pi}")
print(f"学习 pi: {learnt_hmm.pi}")
```

---

## 四、应用场景

### 4.1 词性标注（POS Tagging）

**词性标注**是HMM在NLP中最经典的应用之一。

**问题设定**：
- **隐藏状态：词性标签**（名词、动词、形容词等）
- **观测值：句子中的单词**
- **转移概率**：$P(\text{词性}_t \mid \text{词性}_{t-1})$——某种词性后面通常跟什么词性
- **发射概率**：$P(\text{单词}_t \mid \text{词性}_t)$——某种词性下出现某个单词的概率

给定一个句子 $O = (w_1, w_2, ..., w_T)$，用**维特比算法**找到最可能的词性序列 $I = (t_1, t_2, ..., t_T)$。

### 4.2 语音识别

HMM最早和最成功的应用领域就是**语音识别**。

**问题设定**：
- **隐藏状态：音素**（语音的基本单元）
- **观测值：声学特征向量**（MFCC等）
- 每个单词或音素对应一个**HMM模型**
- 识别时，计算语音信号在每个模型下的概率（**前向算法**），选择**概率最大的模型**

HMM之所以在语音识别中如此成功，是因为语音信号具有**时序特性 —— 前后帧之间存在依赖关系**，而HMM天然适合建模这种依赖。

### 4.3 其他应用

- **中文分词**：将“状态”视为字在词中的位置（词首、词中、词尾、单字词）
- **命名实体识别**：识别文本中的人名、地名、组织名
- **生物信息学**：DNA序列中的基因预测
- **手写识别**：在线手写字符识别

---

## 五、完整代码实现：天气预测

```python
import numpy as np
import matplotlib.pyplot as plt

# 创建一个天气-海藻的HMM示例
# 隐藏状态：0=晴天, 1=多云, 2=雨天
# 观测值：0=海藻干, 1=海藻湿润, 2=海藻潮湿

weather_hmm = HMM(n_states=3, n_obs=3)

# 初始状态分布：假设第一天天气不确定
weather_hmm.pi = np.array([0.4, 0.3, 0.3])

# 状态转移矩阵：天气变化规律
weather_hmm.A = np.array([
    [0.6, 0.3, 0.1],  # 晴天 -> 晴天0.6, 多云0.3, 雨天0.1
    [0.3, 0.4, 0.3],  # 多云 -> 晴天0.3, 多云0.4, 雨天0.3
    [0.1, 0.3, 0.6]   # 雨天 -> 晴天0.1, 多云0.3, 雨天0.6
])

# 发射概率矩阵：不同天气下海藻的湿度
weather_hmm.B = np.array([
    [0.7, 0.2, 0.1],  # 晴天：干0.7, 湿润0.2, 潮湿0.1
    [0.2, 0.6, 0.2],  # 多云：干0.2, 湿润0.6, 潮湿0.2
    [0.1, 0.3, 0.6]   # 雨天：干0.1, 湿润0.3, 潮湿0.6
])

# 观测序列：连续5天的海藻湿度观测值
obs_sequence = [0, 1, 2, 1, 0]  # 干, 湿润, 潮湿, 湿润, 干

# 1. 估值问题：计算观测序列的概率
prob = weather_hmm.forward(obs_sequence)
print(f"观测序列 {obs_sequence} 的概率: {prob:.6f}")

# 2. 解码问题：推断最可能的天气序列
best_weather = weather_hmm.viterbi(obs_sequence)
weather_names = ['晴天', '多云', '雨天']
print(f"观测序列: {obs_sequence}")
print(f"最可能的天气序列: {[weather_names[s] for s in best_weather]}")

# 3. 学习问题：从观测序列中学习参数
# 生成更多数据用于训练
_, train_obs = weather_hmm.generate(T=1000)

learnt_hmm = HMM(n_states=3, n_obs=3)
learnt_hmm.baum_welch(train_obs, n_iter=30)

print("\n学习前后的参数对比（部分）:")
print(f"真实转移矩阵:\n{weather_hmm.A}")
print(f"学习转移矩阵:\n{learnt_hmm.A}")
```

---

> [!note] 总结
> 
> | 概念 | 核心内容 |
> |------|---------|
> | **HMM五元组** | $\lambda = (Q, V, \pi, A, B)$：**状态集、观测集、初始分布、转移矩阵、发射矩阵** |
> | **估值问题** | 计算 $P(O \mid \lambda)$，用**前向算法**（$O(TN^2)$）代替直接计算（$O(TN^T)$） |
> | **解码问题** | 找**最可能的状态序列** $I^*$，用**维特比算法**（动态规划） |
> |**学习问题** | **从观测序列估计参数**，**Baum-Welch算法** = EM算法在HMM中的实现 |
> | **Baum-Welch** | **E步：计算 $\gamma_t(i)$ 和 $\xi_t(i,j)$；M步：更新 $\pi, A, B$** |
> | **核心假设** | **一阶马尔可夫性**（状态只依赖前一时刻）+ **观测独立性**（观测只依赖当前状态） |
> 
> 核心要点回顾
> 
> 1. **HMM是一个双重随机过程**：底层是看不见的**马尔可夫链（状态转移）**，上层是**根据状态生成观测值的过程（发射）**。**五元组 $(Q, V, \pi, A, B)$** 完整描述了这个模型。
> 2. **三大核心问题**覆盖了HMM的全部使用场景：**估值**（前向算法）计算观测序列的概率，用于**模型匹配**；**解码**（维特比算法）推断最可能的状态序列，用于**序列标注**；**学习**（Baum-Welch算法）**从数据中估计参数**，用于**模型训练**。
> 3. **前向算法**用动态规划将估值问题的复杂度从 $O(TN^T)$ 降到 $O(TN^2)$。**后向算法**是从后往前递推，两者结合使用可以计算**各种期望统计量**。
> 4. **维特比算法**是动态规划在解码问题上的经典应用，通过“保**留到达每个状态的最优路径**”来高效地找到**全局最优状态序列**。
> 5. **Baum-Welch算法**是EM算法在HMM中的具体实现。E步用**前向后向算法**计算**隐状态的期望**（$\gamma_t(i)$ 和 $\xi_t(i,j)$），M步用这些期望**更新参数**。它保证每一步都**提升似然函数**，但**只能收敛到局部最优**。
> 6. **HMM的假设**是两个“朴素”但实用的假设：**一阶马尔可夫性**（未来只依赖当前）和**观测独立性**（观测只依赖当前状态）。这些假设虽然简化了模型，但在**语音识别、词性标注**等任务中效果极好。