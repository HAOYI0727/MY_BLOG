---
title: GSPO (Group Sequence Policy Optimization) -- 群组序列策略优化
published: 2026-07-12
description: 系统梳理GSPO群组序列策略优化的核心原理，解析其如何通过序列级重要性比率与长度归一化从根本上修复GRPO的token级采样缺陷，解决MoE专家震荡与长序列方差累积问题，并探讨GSPO的局限与后续演进。
cover: "/assets/images/posts/GSPO.png"
coverInContent: false
tags: [GSPO, GRPO, RLHF, 强化学习, 推理增强, Qwen, 模型对齐]
category: AI_Alignment
draft: false
---

# GSPO (Group Sequence Policy Optimization) -- 群组序列策略优化

> [!important]
> 
> 原论文：[Group Sequence Policy Optimization](https://export.arxiv.org/pdf/2507.18071)
>
> 优秀博客：[A Simple Explanation of GSPO](https://www.adaptive-ml.com/post/a-simple-explanation-of-gspo)、[Paper Review: Group Sequence Policy Optimization](https://andlukyane.com/blog/paper-review-gspo)
> 
> TRL库：[huggingface/trl/gspo_token](https://github.com/huggingface/trl/blob/main/docs/source/gspo_token.md)
> 
> Unsloth集成：[Unsloth Documentation: GSPO Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/gspo-reinforcement-learning)

## 一、引言：从"工程修补"到"算法重构"

在前四篇文章中，我们沿着 **PPO → DPO → GRPO → DAPO** 的脉络，见证了**大模型强化学习对齐**的持续演进。**GRPO**通过"组内相对优势"的设计移除了Critic网络，**DAPO**则通过四项工程优化将GRPO打造成了工业级工具。

然而，即便DAPO已经解决了长CoT场景下的**熵崩塌和样本效率**问题，GRPO框架本身仍然存在一个**根本性的算法设计缺陷**。

2025年7月，阿里Qwen团队发表了**GSPO（Group Sequence Policy Optimization，群组序列策略优化）** 。论文指出：GRPO训练不稳定的根源，在于其**重要性采样比率在算法设计层面存在根本性的误用和失效。**

### 1.1 GRPO的"token级诅咒"：高方差的噪声放大器

要理解GSPO，首先要理解GRPO在**算法设计**层面的根本问题。

回顾**GRPO**的核心机制：

- 对于每个prompt，从**旧策略**中采样$G$个回答，每个回答获得一个**奖励**，**组内归一化**后得到**优势值**$\hat{A}_i$。
- 然后，对于回答中的**每一个token**，计算**重要性比率**：$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$$
- **问题在于：GRPO为每个token独立计算重要性比率，但每个token在采样时只被采样了一次**。

要理解这个问题的严重性，我们需要回到**重要性采样（Importance Sampling）** 的基本理论。
- 重要性采样的核心目的是**用从分布$p$中采样的数据，来估计关于分布$q$的期望**。
- 数学基础是：$$\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}\left[\frac{q(x)}{p(x)} f(x)\right]$$
- 上述公式之所以成立，是因为我们在**完整的样本$x$**上计算**概率密度之比**$\frac{q(x)}{p(x)}$。当$x$是一个**完整的序列**时，$q(x)$和$p(x)$都是**联合概率分布**，重要性比率是**有意义的、可被有效估计的**。

然而，**GRPO的做法是将这个比率分解到每个token上，为每个token独立计算$r_{i,t}(\theta)$**。

- 问题在于**每个token在采样时只被采样了一次**，我们并没有为每个token收集**多个样本**来估计这个比率。根据重要性采样理论，**基于单次样本计算出的权重无法有效校正分布偏差，其梯度估计量会伴随极高的方差**。
- 关键的是，GRPO将"**组内相对比较**"获得的**序列级优势值**$\hat{A}_i$，直接下放给序列中的**每一个token** —— $$\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min(r_{i,t}\hat{A}_i, \text{clip}(r_{i,t})\hat{A}_i)\right]$$ —— 这种强行**将序列级信用分配给每一个独立token**的做法，在统计学上是**缺乏依据**的，这正是**梯度噪声**的根本来源。
- 这种高方差噪声会**随着回答长度的增加而逐步累积**。在一个长度为数千token的长CoT回答中，每个token的重要性比率都存在一定的估计误差，这些误差在序列中**不断叠加**，最终产生巨大的**梯度方差**。
- **裁剪机制**非但没有抑制这种噪声，反而将其**进一步放大**——当某个token的概率比超出了裁剪区间，裁剪会粗暴地截断梯度，但这种"硬处理"反而**加剧了训练的不稳定性**，最终导致**灾难性的、不可逆转的模型崩溃**。

GSPO团队指出，GRPO的根本问题并非简单的"粒度错配"，而是**重要性采样理论的误用 —— 每个token只被采样一次，无法实现有效的分布校正**。基于此诊断，GSPO提出的解法才是**将优化粒度从token级提升到序列级，从而实现奖励单位与优化单位的对齐**。

### 1.2 MoE模型的"专家震荡"：GRPO缺陷的放大镜

GRPO的token级问题在**MoE（混合专家）模型**上被进一步放大。

在MoE模型中，每个token的生成会激活一组特定的**专家**（experts）。GRPO在计算每个token的重要性比率时，会**独立地影响各个专家的激活模式**。MoE模型训练不稳定的根源在于**路由（Routing）的离散性**：训练中**参数更新**会导致专家的**激活模式频繁跳变**，进而导致计算token级重要性比率时所依赖的**条件概率分布发生剧烈波动**，这被称为"**专家震荡**"。

为了抑制这种震荡，此前的研究采用**路由回放（Routing Replay）** 等复杂工程策略——在计算重要性比率时"**回放**"缓存中之前激活的专家路由模式。这种做法虽然能在一定程度上稳定训练，但会产生**额外的内存和通信开销**，并可能**限制MoE模型的实际可用容量**。关键的是，**路由回放并非万灵药**——它只是"治标"而非"治本"。GRPO在MoE上的训练仍然极其不稳定，稍有不慎就会导致训练崩溃。

GSPO之所以能从根本上摆脱对路由回放的依赖，是因为其采用的**序列级联合概率**在数学计算上已经**边缘化掉了具体的专家路由路径**，因此**对单个token的路由变化天然不敏感**。

---

## 二、GSPO的核心原理

GSPO的解决方案直击问题根源：**放弃token级的重要性比率，转而采用序列级的重要性比率**。

### 2.1 序列级重要性比率

在GSPO中，对于整个回答$y_i$，其重要性比率定义为：

$$s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} \right)$$

各符号的含义：
| 符号 | 含义 | 说明 |
|------|------|------|
| $\pi_\theta(y_i \mid x)$ | **当前策略联合概率** | 当前策略$\pi_\theta$生成完整回答$y_i$的概率，即 $\prod_{t=1}^{\|y_i\|}\pi_\theta(y_{i,t} \mid x, y_{i,<t})$ |
| $\pi_{\theta_{old}}(y_i \mid x)$ | **旧策略联合概率** | 采样时使用的旧策略生成完整回答$y_i$的概率 |
| $\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{old}}(y_i \mid x)}$ | **联合概率比** | 整个序列的概率比，未归一化时随长度指数级变化 |
| $\|y_i\|$ | **序列长度** | 回答的token数量（不含prompt），用于长度归一化 |
| $s_i(\theta)$ | **序列级重要性比率** | **每个token的平均概率比的几何平均**，衡量"**平均而言**"新策略相比旧策略更倾向生成这个回答的程度 |

**为什么需要长度归一化？** 如果不做长度归一化，**联合概率比会随序列长度指数级变化**：

$$\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{old}}(y_i \mid x)} = \prod_{t=1}^{|y_i|} r_{i,t}(\theta)$$

- 若平均$r_{i,t} > 1$，长序列的概率比将**爆炸性增长**至无穷
- 若平均$r_{i,t} < 1$，长序列的概率比将**指数级衰减**至0

这会导致**数值不稳定和梯度方差爆炸**。**长度归一化**（开$|y_i|$次方）将$s_i(\theta)$的数值范围统一到合理的尺度上，大幅降低了方差——$s_i(\theta)$的典型值在$[0.5, 2.0]$区间内。

**几何平均**：$s_i(\theta)$衡量的是——**平均而言，当前策略生成这个回答中每个token的概率，相比旧策略提升了多少**。如果$s_i(\theta) > 1$，说明新策略平均而言**更倾向**于生成这个回答；如果$s_i(\theta) < 1$，则相反。

### 2.2 GSPO的优化目标

有了**序列级重要性比率**$s_i(\theta)$，GSPO的优化目标与GRPO在形式上保持了一致，但核心变量从token级换成了**序列级**：

$$\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$

**关键变化**：
- **重要性比率**从$r_{i,t}(\theta)$（**token级**）变为$s_i(\theta)$（**序列级**）
- **裁剪**在**序列**级别进行——整个序列要么被裁剪，要么不被裁剪
- **优势**$\hat{A}_i$仍然是在组内归一化的**序列级奖励**

### 2.3 序列级优化更稳定的理论解释

**层面一：方差消减与梯度优势（统计层面）**

在GRPO中，**每个token的重要性比率独立计算，误差随序列长度线性累积**：

$$\text{Var}(\log r_{i,t}) \approx \sigma^2 \quad \Rightarrow \quad \text{Var}\left(\sum_{t=1}^{|y_i|} \log r_{i,t}\right) \approx |y_i| \cdot \sigma^2$$

而在GSPO中，**$s_i(\theta)$是整个序列的几何平均**：

$$\log s_i(\theta) = \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log r_{i,t}$$

$$\text{Var}(\log s_i(\theta)) = \frac{1}{|y_i|^2} \cdot |y_i| \cdot \sigma^2 = \frac{\sigma^2}{|y_i|}$$

**方差被显著降低——从$O(L)$降至$O(1/L)$**，其中$L$是序列长度。在长CoT场景中（$L \sim 10^4$），这种方差消减是数量级的。从统计学的角度看，GSPO的$s_i(\theta)$是一个**比GRPO的$r_{i,t}$更有效的估计量**，具有更小的均方误差。

**层面二：优化粒度与奖励单位的对齐（算法设计层面）**

**奖励**是在**序列**级别给出的，GSPO的**优化**也是在**序列**级别进行的。这种**粒度对齐**使得梯度信号更加直接、清晰，避免了token级优化中"**为每个token分配序列级奖励**"这种本质上不合理的操作。

**层面三：对MoE的天然友好（架构层面）**

GSPO只关心**序列级别的似然**$\pi_\theta(y_i|x)$，而对**单个token的似然**$\pi_\theta(y_{i,t}|x, y_{i,<t})$不敏感。这意味着，只要整个序列的生成质量不变，**个别token的专家激活模式发生波动并不会显著影响$s_i(\theta)$的计算**。

更精确地说，MoE模型的**路由选择**仅通过**单个token的条件概率$\pi_\theta(y_{i,t} \mid \cdot)$进入损失函数**。由于GSPO的$s_i(\theta)$是token级对数概率的**平均**，**路由波动**对对数概率的影响被长度归一化**平滑化**。因此，**GSPO从根本上消除了对路由回放策略的依赖**。

### 2.4 理论视角：Perplexity-Entropy等价性

后续的理论工作为GSPO提供了更深刻的理解。研究发现，GSPO的**长度归一化重要性比率**$s_i(\theta)$可以等价地表示为：

$$\log s_i(\theta) = \log \frac{\text{Perplexity}_{\theta_{old}}(y_i)}{\text{Perplexity}_{\theta}(y_i)} = -\Delta \text{CrossEntropy}_{\theta \to \theta_{old}}(y_i)$$

这个发现揭示了GSPO与语言模型经典指标之间的深层联系：
- **困惑度（Perplexity）** ：$\text{PPL}(y) = \exp\left(-\frac{1}{|y|}\sum_{t=1}^{|y|}\log \pi(y_t \mid y_{<t})\right)$，衡量模型对序列的"**不确定性**"
- GSPO的$s_i(\theta)$本质上衡量的是**新策略相比旧策略在生成该序列时的困惑度变化**
- 优化$s_i(\theta)$就是在优化**序列级别的困惑度改善**

这一视角将**序列级强化学习与语言模型困惑度最小化**联系了起来，提供了更直观的理解——GSPO本质上是在鼓励模型在**序列级别**上做出更"**自信**"的预测（即**更低的困惑度**），而非在token级别上逐点优化。

---

## 三、GSPO的训练流程

### 3.1 训练流程步骤

1. **组采样（Group Rollout）**：对于每个prompt $x$，从当前策略$\pi_{\theta_{old}}$中**采样$G$个完整回答**$\{y_1, y_2, \ldots, y_G\}$
2. **奖励计算与优势归一化**：**奖励模型**（或规则）对每个完整回答**打分**，得到$\{R_1, R_2, \ldots, R_G\}$；**组内归一化得到优势**：$\hat{A}_i = \frac{R_i - \mu}{\sigma}$
3. **序列级重要性比率计算（关键差异）**
   - **GSPO**：计算整个回答$y_i$的联合概率比，进行长度归一化
   - **GRPO**：逐token计算概率比$r_{i,t}(\theta)$
4. **策略优化**：使用**序列级**$s_i(\theta)$进行裁剪和优化，无需token级损失聚合

### 3.2 核心伪代码

```python
import torch
import numpy as np

# ==================== 超参数配置 ====================
GROUP_SIZE = 16
EPSILON = 0.2
LEARNING_RATE = 1e-6
EPOCHS_PER_ITER = 5

# ==================== 主训练循环 ====================
for iteration in range(total_iterations):
    
    # ---------- Step 1: 组采样 (与GRPO相同) ----------
    groups = []
    for prompt in prompts_batch:
        responses = model_actor.sample(prompt, num_samples=GROUP_SIZE)
        # 计算旧策略的序列级log概率 (用于后续重要性比率)
        # 注意: GSPO只需要序列级log概率，不需要token级
        old_log_probs = model_actor.compute_sequence_log_probs(responses)
        ref_log_probs = model_ref.compute_sequence_log_probs(responses)
        rewards = [model_rm.score(prompt, resp) for resp in responses]
        groups.append({
            "prompt": prompt,
            "responses": responses,
            "old_log_probs": old_log_probs,  # [G] 序列级
            "ref_log_probs": ref_log_probs,  # [G] 序列级
            "rewards": rewards               # [G]
        })
    
    # ---------- Step 2: 奖励归一化 (与GRPO相同) ----------
    for group in groups:
        rewards = group["rewards"]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        group["advantages"] = [(r - mean_r) / std_r for r in rewards]
    
    # ---------- Step 3: 策略更新 (GSPO的核心差异) ----------
    dataset = flatten_experiences(groups)
    
    for epoch in range(EPOCHS_PER_ITER):
        for batch in sample_batches(dataset, BATCH_SIZE):
            # 计算当前策略的序列级log概率
            new_log_probs = model_actor.compute_sequence_log_probs(batch["responses"])
            
            # ---------- GSPO: 序列级重要性比率 ----------
            # 长度归一化: 除以序列长度，取几何平均
            seq_lengths = batch["seq_lengths"]  # [B]
            # s_i(θ) = exp((log π_θ - log π_θ_old) / length)
            log_ratio = new_log_probs - batch["old_log_probs"]  # [B]
            s = torch.exp(log_ratio / seq_lengths)  # [B], 序列级标量
            
            # ---------- GSPO: 序列级裁剪 ----------
            advantages = batch["advantages"]  # [B]
            surr1 = s * advantages
            surr2 = torch.clamp(s, 1 - EPSILON, 1 + EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # ---------- KL散度 (可选) ----------
            # GSPO保留了KL约束，但同样在序列级别计算
            kl = (new_log_probs - batch["ref_log_probs"]) / seq_lengths
            kl_loss = kl.mean()
            
            loss = policy_loss + KL_BETA * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**关键差异**：

| 步骤 | GRPO | GSPO |
|------|------|------|
| **log概率粒度** | token级 (`[B, seq_len]`) | **序列级 (`[B]`)** |
| **重要性比率** | 每个token独立计算 | **整个序列的长度归一化几何平均** |
| **裁剪** | token级硬裁剪 | **序列级硬裁剪** |
| **损失聚合** | token级平均 | **序列级平均** |

### 3.3 实验效果

Qwen团队在Qwen3-30B-A3B-Base微调的冷启动模型上进行了实验：

| 维度 | GRPO | GSPO |
|------|------|------|
| **训练稳定性** | 需要Routing Replay才能收敛 | **无需任何额外策略，天然稳定** |
| **训练效率** | 基线 | **显著更高** |
| **可扩展性** | 性能容易饱和 | 可通过增加算力获得持续**性能提升** |
| **MoE训练** | 极其不稳定，需复杂策略 | 根**本解决稳定性问题** |

在CodeForces任务中，GRPO的最终得分收敛于2000分以下，而GSPO随着训练计算量的增加**持续提升成绩**，展现出更强的可扩展性。

**GSPO所裁剪的token比例比GRPO高两个数量级，但却具有更高的训练效率**。这进一步表明，GRPO的token级优化目标存在噪声大、效率低的问题，而GSPO的序列级优化目标提供了**更可靠、有效的学习信号**。

---

## 四、GSPO解决的问题和创新点

### 4.1 GSPO解决的问题

1. **解决了GRPO中token级重要性采样的根本性缺陷**：GSPO通过将重要性比率从token级提升到**序列级**，**从算法设计层面**解决了GRPO中"**单次采样无法实现有效分布校正**"的根本问题。这不是工程层面的调优，而是**算法层面的修正**。
2. **解决了长序列训练中的方差累积问题**：在长CoT场景中，GRPO的token级误差会**随序列长度线性累积**。GSPO通过**几何平均**将误差相互抵消，**从根本上消除了这种累积效应**。
3. **解决了MoE模型的训练稳定性问题**：GSPO对单个token的似然不敏感，**从根本上消除了MoE训练中的"专家震荡"问题**，彻底摆脱了对**路由回放策略**的依赖。
4. **实现了优化粒度与奖励单位的对齐**：GSPO将"优化什么"与"奖励什么"统一到了**序列级别**，使得梯度信号更加直接、清晰、有效。
5. **简化了RL基础设施**：由于GSPO仅使用序列级别的似然进行优化，它对**精度差异的容忍度更高**。这使得GSPO可以直接使用推理引擎返回的似然进行优化，而**无需使用训练引擎重新计算**——这在partial rollout、多轮RL以及训推分离框架等场景中特别有益。

### 4.2 GSPO创新点

| 创新点 | 说明 |
|--------|------|
| **序列级重要性比率** | **基于整个序列的联合概率比**，而非逐token的概率比 |
| **长度归一化** | **通过几何平均将序列概率比归一化到统一的数值范围**，大幅降低方差 |
| **序列级裁剪** | **裁剪操作在序列级别进行**，而非token级别 |
| **优化粒度对齐** | **优化单位（序列）与奖励单位（序列）完全对齐** |
| **MoE原生友好** | 无需路由回放等复杂 stabilization 策略 |
| **基础设施简化** | 对精度容忍度更高，可直接复用推理引擎的似然 |

---

## 五、GSPO的适应场景、优缺点与局限

### 5.1 适应场景

1. **MoE模型的大规模RL训练**：这是GSPO被设计出来的核心场景，从根本上解决了MoE的专家震荡问题
2. **长序列推理任务**：数学推理（AIME）、代码生成（CodeForces、LiveCodeBench）等需要长CoT的任务
3. **追求训练稳定性的团队**：GSPO**天然稳定**，无需Routing Replay等复杂策略
4. **需要持续RL Scaling的场景**：GSPO可以通过增加算力获得持续的**性能提升**
5. **训推分离的RL系统**：GSPO对精度容忍度高，可直接复用推理引擎的似然，简化基础设施

### 5.2 核心优势

| 优势 | 说明 |
|------|------|
| **算法层面根治不稳定性** | 从根源上解决了GRPO中**重要性采样权重**的设计缺陷 |
| **训练极其稳定** | 无需Routing Replay等额外策略即可**稳定收敛** |
| **MoE原生友好** | 从根本上解决了MoE的**专家震荡**问题 |
| **可扩展性强** | 可通过增加算力获得**持续性能提升** |
| **基础设施友好** | 对**精度**容忍度高，可简化RL系统设计 |
| **优化粒度对齐** | **优化单位与奖励单位完全一致** |
| **已被广泛采用** | 已被HuggingFace TRL v0.20原生支持 |

### 5.3 核心局限

| 局限 | 说明 |
|------|------|
| **牺牲了token级信用分配** | **整个序列共享同一个重要性比率**，无法区分序列中哪些token对最终奖励贡献更大 |
| **硬裁剪的"一刀切"问题** | 当一个序列包含**少数高度离策略**（off-policy）的token时，GSPO会**抑制该整个序列的所有梯度** |
| **可能引发"长度坍塌"** | 研究发现GSPO存在固有的长度偏好，可能导致**回答长度随着训练逐渐坍缩** |
| **对密集模型效果可能有限** | 有观点认为GSPO对MoE模型效果显著，但对**密集（dense）模型**可能效果有限或无明显提升 |
| **序列级粒度可能过粗** | 对于需要细粒度反馈的**复杂推理任务**，序列级优化可能不如token级精细 |
| **硬裁剪固有的学习信号丢失** | 被裁剪区间外的梯度全部丢弃，影响样本效率 |

### 5.4 演进方向

GSPO虽然从根本上解决了GRPO的稳定性问题，但研究者们已经识别出几个关键问题，并催生了后续算法的演进：

**问题一：硬裁剪的"一刀切"问题。** GSPO的序列级硬裁剪意味着，只要序列中有一个token"出格"，整个序列的梯度就被全部抑制。这催生了**SAPO（Soft Adaptive Policy Optimization）**——其核心创新在于引入**连续的、平滑的软门控机制**（Soft Gating）来替代GSPO和GRPO中"截断即丢弃"的硬裁剪（Hard Clipping）。在保持**序列级优化连贯性**的同时，实现了**Token级别的自适应调整**，能够根据每个token的离策略程度动态赋予权重，从而**保留了更多有效的学习信号**。

**问题二：长度坍塌。** 后续研究发现，GSPO的**损失函数本身在数学设计上就存在固有的长度偏好**（Length Bias），这种偏好会在训练中诱发"**响应长度崩溃**（Response Length Collapse）"现象，即模型倾向于生成**越来越短**的回答。这催生了**LUSPO（Length-Unbiased Sequence Policy Optimization）**——其核心目标正是**精准纠正GSPO损失函数中这种固有的长度偏差项**，使梯度信号在数学上对响应长度呈严格的无偏状态，从而从根本上遏制长度坍塌。

**问题三：粒度过粗。** GSPO的序列级粒度可能过粗，无法对序列内部不同部分进行差异化的信用分配。这催生了**SSPO（Sub-sentence-level Policy Optimization）**——将整个回答拆分为若干句子，在**子句级别**进行优化，在序列级的稳定性和token级的精细度之间寻找平衡。

**问题四：Token级与序列级的融合。** 研究者尝试在单个目标函数中同时融合token级和序列级的重要性比率。这催生了**DHPO（Dynamic Hybrid Policy Optimization）**——通过**加权机制**动态结合两种粒度的优势。其核心创新在于在同一个裁剪后的**替代目标函数**（clipped surrogate objective）内部，通过**动态权重机制**精细地融合**token级和序列级**的重要性比率，实现了两种粒度在优化路径上的有机统一，而非外在的线性组合。

**问题五：理论基础深化。** 后续工作从**困惑度-熵等价性**的角度重新审视了GSPO，揭示了其与语言模型经典指标之间的深层联系。

---

> [!note]
> 
> 如果说DAPO回答的是"**如何将GRPO从实验室打磨成工业级工具**"的问题，那么GSPO回答的则是一个更为根本的问题：**GRPO算法设计本身是否存在缺陷？如何从根源上修复它？**
> 
> GSPO通过一个看似简单但影响深远的改动给出了答案：
> 
> 1. **问题诊断**：GRPO的**token级重要性采样**，在"每个token只采样一次"的前提下，无法实现有效的分布校正，反而引入**高方差噪声**——这是对重要性采样理论的误用
> 2. **核心洞察**：既然**奖励**是在**序列**级别给出的，**优化**也应当在**序列**级别进行——**让优化粒度匹配奖励粒度**
> 3. **具体方案**：**用序列的联合概率比的几何平均（长度归一化）替代逐token的概率比**
> 4. **连锁效应**：序列级优化不仅解决了方差累积问题，还**顺便**解决了MoE的**专家震荡**——因为GSPO对单个token的似然不敏感
> 
> 这个改动**从算法设计的层面**，而非工程调优的层面，根治了GRPO的不稳定性。GSPO无需Routing Replay等复杂的策略即可稳定训练MoE模型。
> 
> 从PPO到GRPO到DAPO到GSPO，我们看到了大模型强化学习对齐的一条清晰的演进脉络：
> 
> | 算法 | 核心创新 | 解决的问题 | 遗留/新问题 |
> |------|---------|-----------|------------|
> | **PPO** | 裁剪代理目标 + KL约束 | 策略更新稳定性 | 需要价值网络，资源消耗大 |
> | **GRPO** | 组内相对优势 | 移除价值网络 | token级方差累积，MoE不稳定 |
> | **DAPO** | 四项工程优化 | 熵崩塌、样本效率 | token级根本缺陷未解决 |
> | **GSPO** | 序列级重要性比率 | token级方差、MoE震荡 | 长度坍塌、硬裁剪"一刀切" |
> 
> GSPO的发布标志着**大模型强化学习对齐从"工程修补"进入了"算法重构"的新阶段**。它不仅是Qwen3模型卓越性能的算法基石，也为整个社区提供了一个更稳定、更可扩展的RL训练范式。