---
title: 强化学习基础 -- 策略梯度、优势函数、重要性采样与KL散度惩罚
published: 2026-07-02
description: 系统梳理策略梯度、优势函数、重要性采样与KL散度惩罚的核心概念，串联从VPG到PPO的技术演进逻辑，并探讨On-Policy与Off-Policy的权衡及数据漂移问题。
cover: "/assets/images/posts/Basic_Knowledge.png"
coverInContent: false
tags: [强化学习, 基础知识]
category: AI_Alignment
draft: false
---

# 强化学习基础 -- 策略梯度、优势函数、重要性采样与KL散度惩罚

> "理解PPO等强化学习算法之前，必须先理解它试图解决什么问题，以及它站在了哪些巨人的肩膀上。"

## 一、策略梯度（Policy Gradient）

### 1.1 从价值方法到策略方法

强化学习的核心目标是找到一个**最优策略**——即**智能体在给定状态下应该采取什么行动**。

早期的**价值方法**（如Q-learning、DQN）占据了主导地位：**先估计每个状态-动作对的价值函数$Q(s,a)$，再从中推导出策略（通常是$\epsilon$-贪婪策略）**。

但价值方法存在几个根本性的局限：
1. **连续动作空间的困境**：如果动作空间是**连续**的（如机器人关节角度），$\max_a Q(s,a)$**无法通过穷举计算**，需要通过**复杂的优化过程**求解。
2. **策略表达能力的限制**：确定性策略$\pi(s) = \arg\max_a Q(s,a)$无法表达**随机策略**——而在部分可观测环境或需要探索的场景中，随机策略可能是最优的。
3. **价值函数对策略的直接优化并不等价**：最优价值函数$\max_\pi Q^\pi$的优化目标与策略的最终性能之间隔着一层**间接关系**。

近期的**策略梯度方法**则**直接对策略进行参数化**，然后**沿着提升期望累积奖励的方向更新参数**。设策略为$\pi_\theta(a|s)$，其中$\theta$是参数（如神经网络的权重），目标是**最大化期望累积奖励**：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

其中$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$是**轨迹**，$R(\tau)$是**折扣累积奖励**。

### 1.2 策略梯度定理的完整推导

**策略梯度定理**（Policy Gradient Theorem，Sutton et al., 1999）是这一切的数学基石。它告诉我们：**即使$J(\theta)$对$\theta$的依赖通过轨迹分布$\tau \sim \pi_\theta$间接体现，梯度仍然有一个简洁的表达式**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

**公式的直觉含义**：

- $\nabla_\theta \log \pi_\theta(a_t|s_t)$：这是**得分函数（Score Function）** ，指示了参数$\theta$朝哪个方向移动，才能让动作$a_t$在状态$s_t$下变得**更可能**
- $R(\tau)$：这是**权重**，决定了**参数调整的幅度**——回报越高，调整幅度越大
- 整个式子意味着：**"执行一个动作，观察回报，然后朝着让这个动作更可能发生的方向调整参数，调整幅度与回报的大小成正比"**
  
**具体的数学推导**：

首先，将期望展开为对所有可能轨迹的积分：

$$\nabla_\theta J(\theta) = \nabla_\theta \int_\tau p_\theta(\tau) R(\tau) d\tau$$

其中$p_\theta(\tau)$是策略$\pi_\theta$下轨迹$\tau$的概率密度。将梯度算子移入积分（在正则性条件下）：

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) d\tau$$

利用对数和导数技巧：$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$，得到：

$$\nabla_\theta J(\theta) = \int_\tau p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log p_\theta(\tau) R(\tau)\right]$$

轨迹概率$p_\theta(\tau)$的分解（MDP的无记忆性）：

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$$

取对数：

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log p(s_{t+1}|s_t, a_t)$$

**注意：只有中间项$\log \pi_\theta(a_t|s_t)$依赖于$\theta$！环境动态$p(s_{t+1}|s_t, a_t)$和初始状态分布$p(s_0)$都不依赖$\theta$。**

因此：

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

代入，即得：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

这个公式的直观含义是：**对每个动作，用它的对数概率梯度乘以该轨迹的总回报，然后取期望**。

### 1.3 REINFORCE算法与蒙特卡洛估计

实际中，我们无法精确计算这个期望，而是通过**蒙特卡洛采样**来估计——**让智能体与环境互动若干轮，收集轨迹，然后用样本均值近似梯度**。这就是经典的**REINFORCE算法**（Williams, 1992）：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot R(\tau_i)$$

**python伪代码示例**：

```python
def reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99):
    """
    最朴素的策略梯度算法。
    核心思想：用蒙特卡洛采样估计期望梯度，直接沿梯度方向更新。
    """
    for episode in range(num_episodes):
        # ----- 阶段1: 采样一条完整轨迹 -----
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            action = policy.sample(state)      # 从当前策略采样
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        # ----- 阶段2: 计算折扣累积回报 -----
        discounted_rewards = []
        cumulative = 0
        for t in range(len(rewards) - 1, -1, -1):
            cumulative = rewards[t] + gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        
        # ----- 阶段3: 策略梯度更新 -----
        # 对轨迹中的每一步，计算梯度并累加
        total_gradient = 0
        for t in range(len(states)):
            # 策略梯度定理：∇θ log πθ(a|s) * R(τ)
            grad = policy.compute_gradient(states[t], actions[t])
            total_gradient += grad * discounted_rewards[t]
        
        # 沿梯度方向更新参数
        policy.params += learning_rate * total_gradient
```

REINFORCE的核心优势是**无偏性**——只要采样足够多，梯度估计会收敛到真实梯度。但它面临两个致命问题：

- **高方差（High Variance）。** 梯度估计中使用的总回报$R(\tau)$是一个**累积量**，噪声极大。同一个状态下执行同一个动作，后续的**随机性（环境转移、策略采样）**可能导致完全不同的回报。这使得**梯度估计的方差**高得令人难以接受，**训练曲线剧烈震荡**。
- **采样效率低下（Sample Inefficiency）。** 策略梯度本质上是**On-Policy**的：每次更新参数后，策略变了，之前采集的数据就不再有效（因为梯度期望是在当前策略下计算的）。这意味着**每更新一次参数就要重新与环境互动一轮，极其耗时**。

这两个问题催生了两个核心工具：
- **优势函数**（Advantage Function）→ **降低方差**
- **重要性采样**（Importance Sampling）→ **实现Off-Policy复用数据**

---

## 二、优势函数（Advantage Function）

### 2.1 使用优势函数的原因

回想策略梯度的基本形式：$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$。这里用总回报$R$作为动作好坏的度量。

但问题在于：**一个动作的绝对回报并不能说明它本身的好坏**。假如某个状态无论如何都能获得回报（环境本身就很"慷慨"），那么所有动作的$R$都是正的，梯度会让所有动作的概率都增加——这显然不合理。我们需要的是**相对评估：这个动作比"平均水平"好多少？**

### 2.2 优势函数的定义与偏差-方差权衡

**优势函数$A^\pi(s,a)$** 精确地回答了这个问题：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

其中：
- $Q^\pi(s,a) = \mathbb{E}_{\tau \sim \pi}[\sum_{t=0}^\infty \gamma^t r_t | s_0=s, a_0=a]$：**在状态$s$执行动作$a$后续的期望总回报**
- $V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s,a)]$：**在状态$s$按照策略$\pi$行动的期望总回报**

优势函数衡量的是一个动作**相对于策略平均水平的额外价值**：
- 若$A>0$，说明这个动作优于平均水平，应该**被鼓励**
- 若$A<0$，则劣于平均水平，应该**被抑制**
- 若$A=0$，与平均水平持平，无特殊倾向

将优势函数代入策略梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s,a)]$$

**为什么能替换？** 因为对任意只依赖于状态$s$的基线$b(s)$：

$$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = 0$$

$V^\pi(s)$只依赖于状态，因此减去它不改变梯度的期望，但能**显著降低方差**——这就是所谓的"**带基线的策略梯度**"（Policy Gradient with Baseline）。

### 2.3 广义优势估计（GAE）

实践中，我们无法直接获得真实的$A^\pi$，需要从数据中估计。**广义优势估计（GAE, Generalized Advantage Estimation）** 是由Schulman等人（2016）提出的标准方法，它用TD残差的折扣和来估计优势：

$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

其中$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$是**TD误差（Temporal Difference Error）** 。

**GAE的精妙之处**：通过参数$\lambda \in [0,1]$在**偏差（Bias）** 和**方差（Variance）** 之间做连续调节：

| $\lambda$值 | 估计方式 | 偏差 | 方差 | 适用场景 |
|------------|---------|------|------|---------|
| $\lambda=0$ | **单步TD优势**：$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | **低**（依赖价值函数近似） | **低** | 价值函数准确时 |
| $\lambda=1$ | **蒙特卡洛优势**：$A_t = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$ | **无偏**（无价值函数展开误差） | **极高** | 轨迹短、方差可控时 |
| $\lambda=0.95$（典型） | 中间插值 | 中等 | 中等 | **大多数场景的最佳平衡** |

**具体推导过程**：

1. **TD误差的定义**
   - 对于给定的价值函数$V$，TD误差定义为：$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
   - 这个量衡量的是：**实际观测到的即时奖励加上下一状态的估计价值，与当前状态估计价值的差值**。如果$\delta_t^V > 0$，说明当前状态的实际表现比预期更好。

2. **从TD误差到优势估计**
    - $k$步优势估计定义为：$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$$
    - 当$k \to \infty$时，$$\hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V = Q(s_t,a_t) - V(s_t) = A(s_t,a_t)$$，这是**无偏的蒙特卡洛估计**。
    - 当$k=1$时，$$\hat{A}_t^{(1)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$，这是**高偏差但低方差的单步TD估计**。

3. **GAE的插值**
   - GAE通过参数$\lambda$对上述所有$k$步估计进行指数加权平均：$$A_t^{GAE(\gamma,\lambda)} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}$$
   - 代入$\hat{A}_t^{(k)}$的定义，可以化简为：$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$
   - 这个简洁的递归形式是GAE在实际实现中的标准形式。

**python伪代码示例**
```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    计算GAE优势函数。
    
    Args:
        rewards: [T] 每一步的即时奖励
        values: [T+1] 每一步的价值估计（最后一步为0或V(s_terminal)）
        gamma: 折扣因子
        lam: GAE的λ参数
    
    Returns:
        advantages: [T] 每一步的优势估计
        targets: [T] 每一步的目标价值（用于更新Critic）
    """
    T = len(rewards)
    advantages = [0] * T
    gae = 0
    
    # 逆序计算（从后向前），确保TD误差正确累积
    for t in range(T - 1, -1, -1):
        # TD误差：δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t+1] - values[t]
        
        # GAE递推：A_t = δ_t + γλA_{t+1}
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    
    # 目标价值 = 优势 + 当前价值估计
    targets = [advantages[t] + values[t] for t in range(T)]
    
    return advantages, targets
```

在LLM对齐实践中，GAE通常与训练的价值网络$V_\phi$配合使用（如PPO），其中$\lambda$取0.95~0.97，$\gamma$取1.0（因为LLM对话是episodic任务，无需折扣）。GRPO和其衍生算法（DAPO、GSPO）则完全摒弃了GAE，改用组内归一化奖励作为优势估计。

---

## 三、重要性采样（Importance Sampling）

### 3.1 核心思想与数学形式

重要性采样是一个源自统计学的通用技术。其核心问题是：**如果我们想计算$\mathbb{E}_{x \sim p}[f(x)]$，但无法从$p$中采样，只能从另一个分布$q$中采样，怎么办？**

答案是做一个数学变换：

$$\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$$

也就是说，从$q$中采样的每个样本，乘上一个**重要性权重**$p(x)/q(x)$，就可以**无偏地估计$p$分布下的期望**。

### 3.2 在强化学习中的应用

在策略梯度中，我们原本需要从当前策略$\pi_\theta$中采样来计算梯度。但如果引入重要性采样，就可以用**旧策略**$\pi_{\theta_{\text{old}}}$采集的数据来更新**新策略**$\pi_\theta$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_{\theta_{\text{old}}}}(s,a) \right]$$

这就把**On-Policy**转化为了**Off-Policy**：**数据来自旧策略（行为策略），但用来更新新策略（目标策略）**。

**更简洁的表示**：

**令$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$为重要性比率**。则目标函数可写为：$$L^{IS}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ r_t(\theta) \cdot A^{\pi_{\theta_{\text{old}}}}(s,a) \right]$$

**最大化这个目标，就等于在旧数据上"重新加权"地优化新策略**。

### 3.3 重要性采样的方差分析

虽然上述等式在数学上成立，但实际使用中有一个巨大的陷阱：**当$p$和$q$的分布差异过大时，重要性权重的方差会爆炸。**

具体地，考虑重要性权重$w(x) = p(x)/q(x)$。在$q$分布下，$w(x)$的方差为：

$$\text{Var}_q(w(x)) = \int \left(\frac{p(x)}{q(x)}\right)^2 q(x) dx - \left(\int \frac{p(x)}{q(x)} q(x) dx\right)^2 = \int \frac{p(x)^2}{q(x)} dx - 1$$

当$q(x)$在$p(x)$较大的区域取值很小时，比值$p(x)/q(x)$会极大，导致方差急剧上升。这就是所谓的**重要性采样的方差爆炸问题**。

在强化学习中，这意味着：**如果新旧策略的差异太大，某些动作在新策略下的概率远高于旧策略，对应的样本会被放大数百倍，完全主导梯度更新**。这正是PPO引入**裁剪机制**和**KL散度约束**的根本动机。

---

## 四、KL散度惩罚

### 4.1 KL散度的定义与性质

**KL散度（Kullback-Leibler Divergence）** 衡量两个概率分布之间的差异：

$$D_{KL}(\pi_{\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s)) = \sum_a \pi_{\text{old}}(a|s) \log \frac{\pi_{\text{old}}(a|s)}{\pi_\theta(a|s)} = \mathbb{E}_{a \sim \pi_{\text{old}}}\left[\log \frac{\pi_{\text{old}}(a|s)}{\pi_\theta(a|s)}\right]$$

**关键性质**：
- $D_{KL}(p \| q) \ge 0$，当且仅当$p=q$时取等号（Gibbs不等式）
- **不对称**：$D_{KL}(p \| q) \neq D_{KL}(q \| p)$，因此它不是一个真正的"距离"
- **能很好地量化"从旧策略变成新策略时，信息损失了多少"**

### 4.2 为什么用KL散度而不是交叉熵？

这是一个常被误解的问题。在监督学习中，交叉熵和KL散度在优化上是等价的——因为标签分布$p_{\text{data}}$固定，**交叉熵的梯度等于KL散度的梯度**：

$$\nabla_\theta H(p_{\text{data}}, p_\theta) = \nabla_\theta [H(p_{\text{data}}) + D_{KL}(p_{\text{data}} \| p_\theta)] = \nabla_\theta D_{KL}(p_{\text{data}} \| p_\theta)$$

但在强化学习中完全不同。PPO中的"旧策略"$\pi_{\text{old}}$是**动态变化的**——每轮更新后它都在变。交叉熵展开为：

$$H(\pi_{\text{old}}, \pi_\theta) = H(\pi_{\text{old}}) + D_{KL}(\pi_{\text{old}} \| \pi_\theta)$$

其中$H(\pi_{\text{old}}) = -\sum_a \pi_{\text{old}}(a|s) \log \pi_{\text{old}}(a|s)$是旧策略的**熵**，这个值随训练过程剧烈波动：
- **训练初期：策略随机性强**，$H(\pi_{\text{old}})$很大，交叉熵阈值被熵项占据，**KL约束几乎不起作用**
- **训练后期：策略趋于确定**，$H(\pi_{\text{old}})$减小，**KL约束突然变强**

这会导致策略更新幅度在训练过程中**非线性变化**，破坏训练的稳定性。**KL散度直接衡量策略分布的变化，不受熵的干扰**，因此是更合适的约束。

### 4.3 KL散度作为信任域约束

TRPO（Trust Region Policy Optimization, Schulman et al., 2015）首次严格证明了：**当新旧策略的KL散度被限制在足够小的范围内时，策略性能的改进是有理论保证的**。一旦KL散度超过某个阈值，价值估计误差会指数级增长。

TRPO将KL散度作为**硬约束**（必须满足），通过共轭梯度法等二阶方法求解：

$$\max_\theta \mathbb{E}_{a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s,a) \right]$$

$$\text{s.t. } D_{KL}(\pi_{\text{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)) \leq \delta$$

但这种方法计算量巨大（需要计算Fisher信息矩阵和共轭梯度迭代）。

PPO则提供了两种更轻量的方案：

**方案一：PPO-Penalty（KL惩罚）**

在目标函数中直接加入KL惩罚项：

$$L^{PPO}(\theta) = L^{CLIP}(\theta) - \beta \cdot D_{KL}(\pi_{\text{old}} \parallel \pi_\theta)$$

其中$\beta$是惩罚系数，可以**自适应调整**：
- 当$D_{KL} > 1.5 \times D_{KL}^{\text{target}}$时，$\beta \leftarrow 2\beta$（加强约束）
- 当$D_{KL} < 0.5 \times D_{KL}^{\text{target}}$时，$\beta \leftarrow \beta/2$（放松约束）

**方案二：PPO-Clip（裁剪）**

直接对重要性比率$r(\theta) = \pi_\theta(a|s)/\pi_{\text{old}}(a|s)$进行裁剪：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A\right)\right]$$

这等效于隐式地限制了策略更新的幅度，但不需要显式计算KL散度。注意：**$\text{clip}(x, 1-\epsilon, 1+\epsilon)$将$x$限制在$[1-\epsilon, 1+\epsilon]$区间内**。

### 4.4 KL散度在RLHF中的角色

在大模型RLHF中，KL散度还有一个特殊作用：**约束模型不偏离SFT阶段的"基础能力"**。

$$\text{Reward}_{\text{final}} = \text{Reward}_{\text{RM}} - \beta \cdot D_{KL}(\pi_\theta \parallel \pi_{\text{SFT}})$$

这个惩罚项**防止模型在追求奖励的过程中"走火入魔"——生成高分但无意义或有害的内容**。$\beta$ 控制对齐目标与能力保持之间的权衡。

---

## 五、On-Policy vs Off-Policy：稳定性与样本效率的权衡

### 5.1 定义与本质区别

| 维度 | On-Policy | Off-Policy |
|------|-----------|------------|
| **交互策略** | 与学习策略相同 | 与学习策略不同 |
| **数据来源** | **当前策略自身** | **其他策略（或过去的自己）** |
| **典型算法** | REINFORCE, TRPO, PPO | Q-learning, DQN, SAC |

**On-Policy（同策略）** ：学习的策略和与环境交互的策略**是同一个**。智能体"**一边实践一边学习**"。每次更新后，**旧的轨迹就"作废"了**。

**Off-Policy（异策略）** ：学习的策略和与环境交互的策略**不是同一个**。智能体"**观察别人（或过去的自己）的行为来学习**"。**一份数据可被多次、多轮使用**。

### 5.2 权衡：稳定性 vs 样本效率

| | On-Policy | Off-Policy |
|--|-----------|------------|
| **稳定性** | ✅ **高**——**梯度估计**直接对应当前策略，优化目标明确 | ❌ **低**——**分布漂移**风险，梯度有偏 |
| **样本效率** | ❌ **低**——每轮更新后数据失效，需**重新采样** | ✅ **高**——**可复用历史数据**，多轮更新 |
| **探索-利用平衡** | 需要**单独设计**探索机制 | 可**分离**探索策略和学习策略 |
| **实现复杂度** | 相对**简单** | 需要**重要性采样校正** |

### 5.3 PPO的定位：本质仍是On-Policy

尽管PPO使用了重要性采样，它本质上仍然是**On-Policy**算法。因为PPO虽然允许用旧策略的数据进行多次更新，但**会严格限制更新的幅度**（通过裁剪或KL约束），确保新策略不会偏离旧策略太远。**当偏差超过阈值时，算法会停止使用这批数据，重新采样**。

换句话说，PPO是在**On-Policy的框架内借用Off-Policy的技术**——**用重要性采样提高数据利用率，但用信任域约束确保策略不会跑偏**。这是PPO能够成为RLHF事实标准的核心原因：它既有On-Policy的稳定性，又有Off-Policy的效率。

---

## 六、数据漂移（Distributional Shift）

### 6.1 什么是数据漂移？

**分布漂移（Distributional Shift）** 是指**训练数据的分布与推理（或部署）时遇到的数据分布不一致**。在强化学习中，这表现为：
- **策略漂移**：训练时评估的策略与实际部署的策略不同
- **环境漂移**：训练环境与部署环境存在差异
- **数据分布漂移**：训练样本的分布与策略实际遇到的状态分布不同

### 6.2 在大模型RLHF中的放大效应

这一问题在LLM的RLHF训练中被**急剧放大**。核心原因在于**训练和推理的解耦**：
1. **训练时**：模型（策略）生成回复（rollout），奖励模型打分，PPO更新参数。这个过程使用的是**当前策略**生成的分布。
2. **推理时**：模型部署后面对的是**真实用户**的prompt分布，可能与训练时的prompt分布大相径庭。

更隐蔽的问题是：**奖励模型本身也会遭遇分布漂移**。奖励模型是在**旧策略的回复分布**上训练的，但当**策略更新**后，新生成的回复可能落在奖励模型从未见过的分布区域，导致奖励估计不准确。这会造成**奖励黑客（Reward Hacking）**——策略学会利用奖励模型的盲点而非真正提升质量。

### 6.3 与重要性采样的区别与联系

数据漂移和重要性采样经常被混淆，但它们解决的问题维度不同：

| | **核心问题** | **解决方案** |
|--|------------|------------|
| **重要性采样** | 如何用分布$q$的数据估计分布$p$的期望 | 乘上权重$p/q$进行数学校正 |
| **数据漂移** | 训练分布与测试/部署分布不一致导致性能下降 | 约束更新幅度、分布鲁棒优化、域自适应 |

在PPO中，两者是**共生关系**：
- 重要性采样**实现了**Off-Policy更新（用旧数据更新新策略）
- KL约束（信任域）**防止了**因策略过快变化导致的数据漂移
- 
### 6.4 缓解策略

| 策略 | 说明 |
|------|------|
| **信任域约束** | TRPO/PPO的**KL约束和裁剪机制**，本质上就是为了**防止单步更新过大导致的分布漂移** |
| **自适应KL惩罚** | **动态调整惩罚系数$\beta$**，当检测到KL散度超过阈值时加强约束 |
| **重要性采样校正** | 通过**重要性权重**对Off-Policy数据进行校正，减少分布不匹配带来的偏差 |
| **分布鲁棒优化（DRO）** | 在RLHF中引入分布鲁棒优化，**最小化在最坏情况数据分布下的损失** |
| **训推一致性监控** | 在训练过程中**持续监控推理引擎**（如vLLM）与训练策略之间的**KL散度**，一旦偏离过大则触发回滚或重新采样 |

---

## 七、熵、KL散度与学习率的一体化关系

在大规模RLHF训练中，**熵（Entropy）、KL散度（KL Divergence）与学习率（Learning Rate）**三者之间存在紧密且微妙的耦合关系。

### 7.1 熵：探索能力的晴雨表

策略熵$H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$衡量的是策略的**随机性**。**熵越高，策略越"不确定"，越倾向于探索多样化的动作。**

在RLHF的早期版本中，目标函数通常包含一个**熵奖励项**$c \cdot H(\pi_\theta)$，**通过最大化熵来鼓励探索**。但在**长CoT推理任务**中，过高的熵可能导致模型生成**大量无意义的随机推理路径**，降低训练效率。GRPO及其衍生算法（DAPO、GSPO）通常**移除了显式的熵奖励**，转而依靠算法本身的探索机制。

### 7.2 KL散度：约束安全绳

KL散度$D_{KL}(\pi_\theta \| \pi_{\text{ref}})$衡量的是**当前策略与参考策略的分布偏离程度**。在RLHF中，KL约束的作用是**确保模型不会为了追求高奖励而"忘本"——丢失预训练和SFT阶段获得的语言能力**。

### 7.3 学习率：三者的"调节阀"

**学习率是三者互动中的关键调节变量**：

- **学习率过高**：即使KL约束和裁剪机制存在，模型也会快速偏离参考策略，导致KL散度急剧上升，可能触发训练崩溃。
- **学习率过低**：策略更新过于缓慢，熵可能因自然衰减而下降，模型陷入局部最优。
- **三者之间的恶性循环**：学习率过高→KL散度爆炸→裁剪机制频繁触发→有效学习信号丢失→熵崩塌→策略多样性丧失→训练停滞。

在实践中，学习率的调度策略必须与KL散度的动态监控以及熵的衰减趋势协同设计。一个常见的安全策略是：**当KL散度超过预设阈值时，自动降低学习率**，从而形成一个**负反馈控制回路**。

---

> [!note]
> 概念的完整串联
> 
> 1. **策略梯度**提供了直接优化策略的框架，但面临**高方差**和**低样本效率**两个核心问题。
> 2. **优势函数**通过减去基线$V(s)$来**降低方差**，同时保持**无偏性**——它告诉我们一个动作**相对于平均水平**有多好。**GAE**通过$\lambda$参数在偏差和方差之间提供了连续插值。
> 3. **重要性采样**让我们可以用**旧策略**的数据来更新**新策略**，将On-Policy转化为Off-Policy，**提升了样本效率**——但代价是可能引入**巨大的方差**（当新旧策略差异过大时）。
> 4. **KL散度惩罚**限制了新旧策略的差异，防止重要性采样的**方差爆炸和策略崩溃**——它充当了 **"安全绳"** 。从TRPO的硬约束到PPO的软约束，这是一条从"严谨但昂贵"到"近似但高效"的工程化路径。
> 5. **On-Policy vs Off-Policy**的权衡本质上是**稳定性与样本效率**的权衡。**PPO通过在On-Policy框架内借用Off-Policy技术**，在两者之间找到了一个实用的平衡点，这是其能在RLHF中广泛应用的根本原因。
> 6. **数据漂移**是这一切问题的终极体现——**训练和推理的分布不一致**会导致所有精心设计的机制失效。**信任域约束、自适应惩罚和分布鲁棒优化**是当前主要的应对手段。
> 7. **熵、KL散度与学习率**构成了一个三维耦合系统，三者之间的动态平衡决定了RLHF训练的成败。实践中，这三者必须**协同设计**，而非各自独立调优。
>
> 正如OpenAI Spinning Up文档中所说："从VPG到TRPO再到PPO的技术演进，正是在**弥补样本效率的不足**，同时**保持On-Policy算法的稳定性优势**"。这一演进的背后，正是上述所有概念交织作用的结果。
> 
> **从理论基础到工程实践，每一步都是在"保持理论正确性"和"提升实际可用性"之间寻找平衡**。而理解这些概念之间的内在联系，比记忆单个公式重要得多——这正是从"会用PPO"到"理解PPO为什么work"的关键跃迁。