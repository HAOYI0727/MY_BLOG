---
title: RLVR (Reinforcement Learning with Verifiable Rewards) —— 可验证奖励强化学习
published: 2026-07-16
description: 系统梳理RLVR（Reinforcement Learning with Verifiable Rewards）的核心思想、数学形式化、主流算法（PPO、GRPO、PACS、ROVER）及训练流程，对比RLHF与RLVR的本质差异，并探讨验证器可靠性、奖励稀疏等核心挑战与前沿解决方案。
cover: "/assets/images/posts/RLVR.png"
coverInContent: false
tags: [RLVR, 强化学习, 后训练, GRPO, 模型对齐]
category: AI_Alignment
draft: false
---

# RLVR (Reinforcement Learning with Verifiable Rewards) —— 可验证奖励强化学习

> [!important]
>
> 原论文：[Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
>
> 优秀博客：[awesome-RLVR](https://github.com/opendilab/awesome-RLVR)、[Reinforcement Learning from Verifiable Rewards](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/)、[Reinforcement Learning with Verifiable Rewards Makes Models Faster, Not Smarter](https://www.promptfoo.dev/blog/rlvr-explained/)


## 引言：后训练时代的范式转移

如果说2024年之前大模型的竞争焦点在于**预训练**（Pre-training），那么2025年之后，主战场已经彻底转向了**后训练**（Post-training）阶段。通过利用**数学、代码**等领域**可验证**的结果作为反馈信号，大模型正在实现推理能力的显著跃升。这种范式，就是本文的主题——**RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习）** 。

RLVR的核心突破在于：不再依赖不可靠或昂贵的人类反馈（即RLHF中的奖励模型），而是利用**客观、可验证的规则**来给予模型奖励。这一转变使得大模型可以在数学、编程等**可自动验证答案**的环境中通过**强化学习**训练，自发地形成类似**推理**的策略：它们学会了**将复杂问题拆解为中间步骤，并掌握了多种来回推敲以解决问题的策略**。

从DeepSeek-R1登顶Nature封面，到OpenAI o1、Kimi k1.5等顶尖模型的大规模应用，RLVR已经成为后训练阶段的事实标准。本文将从数学原理、核心算法、工程实践到前沿进展，为你系统性地拆解这一技术。

## 一、RLVR的核心思想

### 1.1 从RLHF到RLVR

传统的**基于人类反馈的强化学习（RLHF）** 通过训练一个**奖励模型（RM）**来模拟人类偏好，然后使用**PPO**等算法优化策略。然而，RLHF在解决复杂的数学、代码和逻辑推理问题时，**严重依赖大量的人类标注数据**，限制了模型的扩展性。

**RLVR**则走了一条截然不同的路。它用**自动化验证器（Verifier）** 替代了人类标注或神经网络**奖励模型**。验证器根据**预定义的规则**自动检查模型输出的正确性，给出**可验证的奖励信号**。

**RLVR vs RLHF**：RLVR优化**客观正确性**（答案对或错），RLHF优化**主观偏好**（哪个回答更好）。对于**数学推理、代码生成**等有明确正确答案的任务，RLVR是天然的选择。

### 1.2 验证器的类型

在RLVR中，**验证器（Verifier）** 是整套系统的“裁判”。常见的验证器包括：

- **数学与逻辑验证**：检查最终答案是否与标准答案精确匹配
- **代码单元测试**：编译并运行代码，验证功能正确性
- **JSON Schema验证**：确保输出符合指定的结构化格式
- **引用与来源验证**：检查引用是否可解析且支持论点

**验证器输出的奖励通常是二元的**（正确为1，错误为0）。虽然简单，但这种“答对加分、答错扣分”的机制已被证明极其有效。

---

## 二、RLVR的数学形式化

### 2.1 问题建模

标准RLVR中的推理任务可以形式化为一个**有限时域的马尔可夫决策过程（MDP）** 。一个MDP由五元组 $(S, A, P, R, \gamma)$ 定义：

- **状态空间 $S$** ：**当前推理的上下文**，包括问题描述和已生成的推理步骤
- **动作空间 $A$** ：生成**下一个token**或下一个推理步骤
- **状态转移概率 $P(s'|s,a)$** ：在RLVR的推理场景中，状态转移是**确定性的**——给定当前状态和生成的动作，下一个状态是**唯一确定**的
- **奖励函数 $R(s,a)$** ：由**验证器**定义的**奖励信号**
- **折扣因子 $\gamma$** ：通常取1（因为推理任务是有限时域的）

**策略 $\pi_\theta(a|s)$ 是一个语言模型，根据当前状态（prompt + 已生成内容）生成下一个动作（token或推理步骤）**。

### 2.2 奖励函数的设计

RLVR与传统RLHF最核心的区别在于**奖励函数的设计**。
- 在传统RLHF中，奖励函数是通过**人类偏好数据**训练出来的，是**学习的、有噪声的**：$R_{\text{RLHF}}(s,a) = \mathbb{E}[r_{\text{human}}|s,a]$。
- 而在RLVR中，奖励函数是**确定的、可验证的**：$R_{\text{RLVR}}(s,a) = \mathcal{V}(\text{response})$，其中 $\mathcal{V}$ 是验证器函数。

对于一个数学问题 $x$，模型生成完整的推理轨迹 $y = (y_1, y_2, ..., y_T)$，**验证器在最终输出上给出奖励**：

$$R(y) = \begin{cases} 1 & \text{if } y \text{ 正确} \\ 0 & \text{if } y \text{ 错误} \end{cases}$$

这种二元设计**消除了人类标注的主观偏差**，但也带来了新的挑战——**奖励信号稀疏**（奖励只在轨迹结束时给出，**中间步骤没有直接的监督信号**）。模型可能生成长达数千token的推理链，却只在最后得到一个**二元的成功/失败信号**。

为了解决这个问题，研究者们提出了**软奖励（Soft Reward）** 的方案。例如Soft-RLVR框架将每个prompt分解为**原子要求的检查清单**，由LLM验证器**逐项评分**，产生更密集的**部分得分（partial credit）** 信号。

### 2.3 优化目标

RLVR的优化目标是**最大化期望累积奖励**：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} R(s_t, a_t) \right]$$

其中 $\tau = (s_0, a_0, s_1, a_1, ..., s_T)$ 是一条**完整的推理轨迹**。

由于**奖励只在轨迹结束时才非零**（二元奖励的情况），上式可以简化为：

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [R(x, y)]$$

其中 $x$ 是输入问题，$y$ 是模型生成的**完整回答**。

为了稳定训练，通常会加入**KL散度约束**，防止策略偏离参考模型（通常是SFT模型）太远：

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [R(x, y)] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

其中 $\beta$ 是**KL惩罚系数**。

### 2.4 策略梯度

为了优化 $J(\theta)$，使用**策略梯度**方法。标准的策略梯度定理给出：

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ R(x, y) \cdot \nabla_\theta \log \pi_\theta(y|x) \right]$$

- 直观理解：如果一个回答获得了正奖励，我们就增加生成这个回答的概率；如果获得零奖励，就降低其概率。
- 这个简单的形式在实际中会遇到**高方差**的问题。因此，实际实现中通常会引入**基线（baseline）** 来降低方差。

---

## 三、主流RLVR算法

### 3.1 传统方案：PPO

RLVR最早采用的主流算法是**PPO（Proximal Policy Optimization）** 。PPO通过限制每次更新的**策略变化幅度**来保证训练稳定性。

PPO的核心是**裁剪的目标函数**：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是**重要性采样比率**，$\hat{A}_t$ 是**优势函数的估计**。

但PPO在RLVR场景中有一个问题：它需要一个**价值网络（Circle）** 来估计优势函数，这增加了**训练的复杂度和内存开销**。

### 3.2 新一代方案：GRPO

**GRPO（Group Relative Policy Optimization）** 的核心思想是**去掉价值网络，用组内相对比较来代替绝对价值估计**。

具体流程如下：
1. **采样**：对于同一个问题 $x$，模型生成 **$G$ 个不同的回答** $\{y_1, y_2, ..., y_G\}$
2. **验证**：**验证器**检查每个回答的**正确性**，得到奖励 $\{r_1, r_2, ..., r_G\}$，其中 $r_i \in \{0, 1\}$
3. **组内标准化**：计算组内奖励的均值和标准差，**将奖励标准化为优势值**
4. **策略更新**：用标准化后的优势值进行**策略梯度更新**

GRPO的优势函数计算方式：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$$

其中 $$\mu_r = \frac{1}{G}\sum_{i=1}^G r_i , \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_r)^2}$$

这样做的好处是：**无需价值网络**，减少了模型参数和计算开销；**天然的对比信号**，组内比较提供了相对优劣的信息；**更稳定的训练**，标准化处理降低了梯度的方差

> 对GRPO的改进算法：**DAPO（Dynamic Sampling Policy Optimization）** 的核心改进在于**非对称裁剪和动态调整采样**等策略，使模型能够更有效地探索样本。**GSPO（Group Sequence Policy Optimization）** 的优化点是**将优化粒度从token级提升到序列级**，从而实现奖励单位与优化单位的**对齐**。

### 3.3 监督学习方案：PACS

**PACS（imPlicit Actor Critic coupling via a Supervised learning framework）** 是一个新颖的RLVR框架，它**将结果奖励视为可预测的标签**，将RLVR问题重新表述为**监督学习**任务。

具体来说，PACS定义一个**评分函数** $f_\theta(y)$（由策略模型参数化），然后用**交叉熵损失**进行优化：

$$\mathcal{L}_{PACS} = - \mathbb{E}_{y \sim \pi_\theta} [r(y) \cdot \log \sigma(f_\theta(y)) + (1 - r(y)) \cdot \log(1 - \sigma(f_\theta(y)))]$$

梯度分析表明，这种监督式表述**内在地恢复了经典的策略梯度更新**，同时隐式地耦合了**Actor和Critic**的角色。PACS在AIME 2025上的pass@256达到了59.78%，比PPO和GRPO分别提高了13.32和14.36个百分点。

### 3.4 极简方案：ROVER

**ROVER（Random Policy Valuation for Diverse Reasoning）** 提出的观点是标准RLVR可以形式化为**具有确定性状态转移和二元终止奖励的MDP**，其结构比通用强化学习问题简单得多。基于这一洞察，ROVER证明**最优动作可以从固定均匀随机策略的Q函数中恢复**，从而完全绕过了广义策略迭代循环。

ROVER的算法极其简单：**从基于均匀策略Q值的softmax中采样动作**。尽管极度简化，ROVER在多个基准测试上超越了复杂的现有方法，在pass@1上提升了8.2%，在pass@256上提升了16.8%。

## 四、RLVR的算法流程

### 4.1 标准RLVR训练循环

RLVR的训练是一个迭代的四步循环：
1. **采样（Sampling）** —— 对每个prompt，从**策略模型** \( \pi_\theta \) 中**采样**多个候选回答（通常使用**Chain-of-Thought** prompting引导模型生成推理过程）。
2. **验证（Verification）** —— 用**验证器** \( \mathcal{V} \) 检查每个回答的正确性，输出**二进制奖励**。
3. **奖励（Rewarding）** —— **正确**的回答获得 \( r=1 \)，**错误**的获得 \( r=0 \)。
4. **策略更新（Policy Update）** —— 使用**GRPO/PPO**等算法更新策略参数， 奖励**成功**的推理路径。

### 4.2 Python伪代码

以下是一个基于GRPO的简化RLVR训练循环伪代码

```python
import torch
import torch.nn.functional as F

def rlvr_training_step(policy_model, ref_model, prompts, verifier, 
                       num_samples_per_prompt=16, beta=0.1, lr=1e-6):
    """
    单步RLVR训练（基于GRPO）
    Args:
        policy_model: 当前策略模型 π_θ
        ref_model: 参考模型 π_ref（用于KL约束）
        prompts: 输入的prompts列表
        verifier: 可验证奖励函数，返回0/1
        num_samples_per_prompt: 每个prompt采样的回答数 G
        beta: KL惩罚系数
        lr: 学习率
    """
    all_responses = []
    all_rewards = []
    all_log_probs = []
    
    # === 步骤1-2：采样与验证 ===
    for prompt in prompts:
        # 对每个prompt采样G个回答
        responses = []
        rewards = []
        log_probs = []
        
        for _ in range(num_samples_per_prompt):
            # 从策略模型采样
            with torch.no_grad():
                response, log_prob = policy_model.sample(prompt)
            
            # 验证器计算奖励
            reward = verifier.verify(prompt, response)  # 返回0或1
            
            responses.append(response)
            rewards.append(reward)
            log_probs.append(log_prob)
        
        all_responses.extend(responses)
        all_rewards.extend(rewards)
        all_log_probs.extend(log_probs)
    
    # === 步骤3：组内相对优势估计 ===
    advantages = []
    for i in range(0, len(all_rewards), num_samples_per_prompt):
        group_rewards = all_rewards[i:i+num_samples_per_prompt]
        mu = sum(group_rewards) / len(group_rewards)
        sigma = (sum((r - mu) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        sigma = max(sigma, 1e-8)  # 防止除零
        
        for r in group_rewards:
            advantages.append((r - mu) / sigma)
    
    # === 步骤4：策略更新 ===
    # 重新计算当前策略下的log probability（用于重要性采样）
    current_log_probs = []
    for prompt, response in zip(prompts * num_samples_per_prompt, all_responses):
        log_prob = policy_model.log_prob(prompt, response)
        current_log_probs.append(log_prob)
    
    # 重要性采样比
    ratios = torch.exp(torch.tensor(current_log_probs) - torch.tensor(all_log_probs))
    
    # KL惩罚项（计算当前策略与参考策略的KL散度）
    ref_log_probs = []
    for prompt, response in zip(prompts * num_samples_per_prompt, all_responses):
        with torch.no_grad():
            ref_log_prob = ref_model.log_prob(prompt, response)
        ref_log_probs.append(ref_log_prob)
    
    kl_penalty = beta * (torch.tensor(current_log_probs) - torch.tensor(ref_log_probs))
    
    # GRPO损失（带裁剪的策略梯度 + KL惩罚）
    advantages_tensor = torch.tensor(advantages)
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    
    loss = -torch.mean(
        torch.min(ratios * advantages_tensor, clipped_ratios * advantages_tensor) 
        - kl_penalty
    )
    
    # 反向传播与参数更新
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 4.3 验证器设计考量

**验证器是RLVR的“奖励函数”**，其质量直接决定训练效果。验证器设计面临几个核心挑战：
1. **规则验证器的脆弱性**：基于规则的验证器虽然精确，但往往过于“**死板**”。例如，一个规则检查器可能将 \( \frac{12}{36} \) 标记为错误，因为它只识别标准答案 \( \frac{1}{3} \)。这种**假阴性（FN）** 问题在RLVR中普遍存在。
2. **LLM验证器的不可靠性**：用LLM作为验证器虽然更灵活，但容易被表面线索“**欺骗**”。研究表明，当回答以“Let's solve this problem step by step”开头时，GPT-4o等验证器的假阳性率高达35%-66.8%。
3. **应对策略**：研究者提出了多种**纠偏方法**，包括**后向校正**（去偏观察到的二进制奖励以**恢复无偏策略梯度估计**）和**前向校正**（重加权得分函数项使期望更新方向**与干净梯度对齐**）。这些校正方法在GRPO-based RLVR pipeline中均可作为轻量级钩子实现。

---

## 五、RLVR vs RLHF

### 5.1 核心差异对比

| 维度 | RLHF | RLVR |
|------|------|------|
| **奖励信号来源** | **人类标注的主观评分** | **自动化工具的客观验证结果** |
| **奖励信号类型** | 连续值（如1-5分） | **二元信号**（通过/失败） |
| **数据需求** | 百万级标注样本 | 数千条验证用例即可启动 |
| **训练成本** | 标注成本高昂 | 验证自动化，成本极低 |
| **擅长领域** | 创意写作、对话生成等**开放域**任务 | 数学推理、代码生成等**封闭域**任务 |
| **核心风险** | **奖励黑客、主观偏差** | **验证器脆弱性、奖励稀疏** |
| **是否需要奖励模型** | 需要（训练成本高） | **不需要** |
| **可扩展性** | 受限于人力标注 | **高度可扩展** |
| **可审计性** | 低（奖励模型是黑箱） | **高**（每个奖励可追溯） |

### 5.2 奖励信号的本质区别

**RLHF**的奖励信号本质上是**对人类偏好的建模**：$$R_{\text{RLHF}} = \text{RewardModel}(\text{response})$$，这个奖励模型本身是个**神经网络**，会犯错，会被“黑客攻击”。

**RLVR**的奖励信号本质上是**对客观真理的验证**：$$R_{\text{RLVR}} = \text{Verifier}(\text{response})$$，**验证器**可以是确定性的（如编译器），其输出是**可复现、可解释**的。

### 5.3 能力边界的差异

**RLHF**在创意写作等**开放域**任务中表现优异，因为这类任务没有“标准答案”，**人类偏好是唯一的评价标准**。

**RLVR**在数学、编程等**封闭域**任务中能实现接近100%的准确率，因为这些任务**有明确的正确/错误判断标准**。

但RLVR的局限也很明显：**它依赖于验证工具的覆盖范围**。对于没有明确验证标准的任务（如“写一篇有说服力的文章”），RLVR目前还无法直接应用。

### 5.4 从“对齐”到“推理”的范式转移

RLHF和RLVR代表了两种不同的优化方案：
- **RLHF**追求的是**主观质量的优化**——让模型输出更“**好**”（更有帮助、更安全、更符合人类偏好），优化目标是让模型**更像人类**（**对齐人类偏好**）
- **RLVR**追求的是**客观正确性的优化**——让模型输出更“**对**”，优化目标是让模型**更正确**（**最大化任务成功率**）
2025年之前，大模型训练遵循“**预训练+RLHF**”的双阶段范式。RLVR的出现打破了这一格局，将训练从“人类教师指导”转变为“**环境规则驱动**”。

## 六、核心挑战与解决方案

| 核心挑战 / 问题 | 解决方案 / 方法 | 关键结果 / 数据 |
|----------------|----------------|----------------|
| **验证器可靠性**：<br>**假阳性**：LLM 验证器易被表面线索欺骗<br>**假阴性**：规则验证器过于脆弱，对格式敏感 | 探索**噪声奖励下的 RLVR**，通过**校正算法**抵消验证器误差 | - GPT-4o 作为验证器时，若答案以 “Let's solve...” 开头，假阳性率达 **35%–66.8%**<br>- 规则检查器可能将正确分数 `12/36` 误判（与标准答案 `1/3` 格式不同） |
| **奖励稀疏与探索困境**：<br>二元奖励仅在学习结束时提供信号，导致**模型过早收敛（熵崩塌）或无效探索（熵爆炸）**<br>低概率探索 token（“**推理火花**”）逐渐消失 | - | 模型生成数千 token 的推理链，**仅在最后知道对错，信号极度稀疏** |
| **二元奖励稀疏**导致训练效率低 | **Soft-RLVR**：将 prompt 分解为**原子要求的检查清单**，由 LLM 验证器逐项评分，提供**密集的部分得分信号** | 在 IFEval 基准上提升 **+11.1 分** |
| **规则验证器构成根本瓶颈**，无法处理复杂或开放答案 | **生成式验证器**：使用强大 LLM 作为**软性、概率性的奖励模型**，**自动化生成奖励，无需人工标注** | 7B 策略模型借助 7B 生成式奖励模型，**显著超越 72B 的 Qwen2.5-Instruct**，性能超出 **8.6%** |
| 标准 RLVR 依赖**人类标注答案**或手工奖励规范 | **无标签 RLVR**：通过**模型 rollout 的投票**提议候选答案，并用**形式化验证器**（如 Lean）决定正奖励 | 在数学推理基准上达到与**有监督训练相当**的性能，无需人工标签 |


> [!note]
> 
> **RLVR**代表了大模型**后训练**的一次范式转变——**从依赖人类主观偏好的“对齐”，转向基于任务客观正确性的“验证”**。它跳过了昂贵的**奖励模型**训练环节，用**确定性的验证信号**驱动模型探索**复杂的推理路径**。
> 
> DeepSeek-R1等推理模型已经证明了这条路径的可行性。随着**验证器技术**的成熟和**算法理论**的深化，RLVR有望在更广泛的领域发挥作用。
> 
> 当然，RLVR并非RLHF的替代品，而是互补品。对于有**标准答案**的任务，**RLVR**是更优选择；对于**开放性**的创作任务，**RLHF**仍然不可替代。未来的大模型后训练，很可能是两者的深度融合与协同。
