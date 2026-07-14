---
title: RLHF (Reinforcement Learning from Human Feedback) —— 基于人类反馈的强化学习
published: 2026-06-30
description: 系统拆解RLHF的三步流程、核心算法原理、代码实现细节以及面临的挑战，涵盖SFT、奖励模型训练、PPO优化、奖励黑客等核心议题，并探讨DPO、RLAIF等演进方向。
cover: "/assets/images/posts/RLHF.png"
coverInContent: false
tags: [RLHF, 强化学习, 后训练, 奖励模型]
category: AI_Alignment
draft: false
---

# RLHF (Reinforcement Learning from Human Feedback) —— 基于人类反馈的强化学习

> [!important]
>
> 原论文：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)、[Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)

## 引言：为什么大模型需要“听懂人话”？

2022年，OpenAI发表了《Training Language Models to Follow Instructions with Human Feedback》。这篇论文堪称大模型从“能做事”走向“会做事”的分水岭——在此之前，即便像GPT-3这样参数规模达175B的“巨无霸”模型，也常因“捏造事实、偏离指令、生成有害内容”而让用户头疼。

问题的根源在于：大语言模型的预训练目标是“预测互联网文本的下一个token”，这与用户核心需求“安全、诚实地遵循指令”存在本质错位。预训练模型像一位读过万卷书却不懂人情世故的学者——知识渊博，但不知道如何与人得体地交流。

**RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）** 正是为解决这一问题而生。它通过人类反馈来微调模型，让模型输出的内容更符合人类的期望。本文将从专业角度，系统拆解RLHF的三步流程、核心算法原理、代码实现细节以及面临的挑战。

--- 

## 一、RLHF 的三步走战略

RLHF框架包含三个关键步骤，形成一个从“模仿”到“优化”的闭环。

### 1.1 监督微调（Supervised Fine-Tuning, SFT）

RLHF的第一步是收集高质量的**人类示范数据**（human demonstrations），然后用这些数据对预训练模型进行有监督微调。

在InstructGPT的原始论文中，OpenAI收集了约13,000条prompt数据（包括用户提交和标注员编写的prompt），对GPT-3进行有监督训练。这一步的目的是让模型先学会“模仿”人类的输出模式，为后续的强化学习提供一个良好的初始策略。

**SFT的核心数学原理**：给定prompt x和对应的示范输出y，SFT通过最大化条件概率来训练模型：

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{SFT}}} \log \pi_{\theta}(y|x)$$

其中$\pi_{\theta}$是语言模型策略，$\theta$是模型参数。本质上，这是一个标准的交叉熵损失——模型学习让正确答案的生成概率尽可能高。

SFT的局限性在于：它只能学习标注数据中的显式知识，对标注质量极度敏感。更重要的是，SFT无法教会模型“什么比什么更好”——它只能告诉你“这样做是对的”，却无法告诉你“这样做比那样做更好”。

### 1.2 奖励模型训练（Reward Model Training）

RLHF的第二步是训练一个**奖励模型**（Reward Model, RM），用来为模型的输出打分。

奖励模型的核心任务是将人类的**偏好判断**转化为一个可计算的**数值信号**。具体做法是：

1. **收集偏好数据**：让人类标注者对同一prompt下的多个模型输出进行排序。InstructGPT收集了约33,000条prompt对应的模型输出排名数据。
2. **训练奖励模型**：将偏好排序转化为成对比较（pairwise comparison）数据，训练一个模型来预测“哪个输出更受人类偏爱”。

**奖励模型的数学原理**：假设对于同一个prompt $x$，我们有两个输出 $y_w$（被偏好的输出）和 $y_l$（不被偏好的输出）。奖励模型 $r_\phi$ 需要满足：

$$r_\phi(x, y_w) > r_\phi(x, y_l)$$

训练目标是最大化偏好输出与非偏好输出之间奖励差值的sigmoid概率：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}} \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

其中$\sigma$是sigmoid函数。这个损失函数本质上是一个**成对排序损失**（Pairwise Ranking Loss）——它鼓励奖励模型给被偏好的输出打更高的分。

**代码示例：奖励模型训练的核心逻辑**

```python
import torch
import torch.nn.functional as F

def compute_reward_loss(reward_model, prompt_ids, chosen_ids, rejected_ids):
    """
    计算奖励模型的成对排序损失
    
    Args:
        reward_model: 奖励模型，输出一个标量分数
        prompt_ids: prompt的token ids
        chosen_ids: 被偏好的回复的token ids
        rejected_ids: 不被偏好的回复的token ids
    """
    # 计算被偏好回复的奖励分数
    chosen_score = reward_model(prompt_ids, chosen_ids)  # shape: (batch_size,)
    # 计算不被偏好回复的奖励分数
    rejected_score = reward_model(prompt_ids, rejected_ids)  # shape: (batch_size,)
    
    # 成对排序损失：让chosen_score > rejected_score
    loss = -F.logsigmoid(chosen_score - rejected_score).mean()
    
    return loss
```

### 1.3 强化学习优化（PPO）

有了奖励模型之后，第三步就是使用强化学习算法来优化语言模型，使其生成能获得更高奖励分数的输出。

目前RLHF最主流的强化学习算法是**PPO**（Proximal Policy Optimization，近端策略优化）。PPO的核心思想是：在优化策略以最大化奖励的同时，**限制策略更新的幅度**，避免模型在一次更新中发生剧烈变化。

**为什么需要限制更新幅度？** 想象一下，如果每次更新都让模型“放飞自我”，它可能会迅速找到一个“钻空子”的方式——生成一些能骗过奖励模型但实际质量很差的输出。这就是著名的**奖励黑客**（Reward Hacking）问题。

**PPO的数学原理**：PPO的优化目标包含三个部分：

$$\mathcal{L}^{\text{PPO}}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_{\theta}^{\text{RL}}(y|x)} \left[ \min\left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：

- $\rho_t(\theta) = \frac{\pi_{\theta}^{\text{RL}}(y_t|x, y_{<t})}{\pi_{\theta}^{\text{old}}(y_t|x, y_{<t})}$ 是**重要性采样比率**（importance sampling ratio），衡量新旧策略在生成当前token上的概率比值
- $\hat{A}_t$ 是**优势函数**（advantage function），衡量当前动作相对于平均水平的优劣
- $\epsilon$ 是**裁剪超参数**（通常设为0.2），控制策略更新的最大幅度
- $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$ 将比率裁剪到 $[1-\epsilon, 1+\epsilon]$ 区间内

这个裁剪操作是PPO的核心创新：当$\rho_t$超出$[1-\epsilon, 1+\epsilon]$范围时，梯度被“裁剪”掉，从而防止策略更新过大。

**代码示例：PPO的核心更新逻辑**

```python
import torch

def ppo_loss(new_log_probs, old_log_probs, advantages, epsilon=0.2):
    """
    计算PPO的策略损失（核心裁剪机制）
    
    Args:
        new_log_probs: 当前策略下动作的对数概率
        old_log_probs: 旧策略下动作的对数概率
        advantages: 优势函数值
        epsilon: 裁剪范围
    """
    # 计算概率比率 r_t(θ) = π_new / π_old
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 未裁剪的损失
    surr1 = ratio * advantages
    
    # 裁剪后的损失
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # 取最小值（更保守的更新）
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss
```

**RLHF中的完整PPO目标函数**：在实际的RLHF训练中，PPO的目标函数还会加入两个额外的项：

$$\text{objective}(\theta) = \mathbb{E}_{x,y\sim\pi_{\theta}^{\text{RL}}} \left[ r_\phi(x,y) - \beta \cdot D_{\text{KL}}(\pi_{\theta}^{\text{RL}} \| \pi^{\text{SFT}}) \right]$$

其中：

- $r_\phi(x,y)$ 是奖励模型给出的分数
- $D_{\text{KL}}(\pi_{\theta}^{\text{RL}} \| \pi^{\text{SFT}})$ 是当前策略与SFT模型之间的**KL散度**（Kullback-Leibler divergence），作为**惩罚项**防止模型过度偏离初始的SFT模型
- $\beta$ 是KL惩罚系数，控制惩罚的强度

KL散度惩罚项的作用是作为一个“锚点”，防止当前被优化的模型发生过大的偏移，从而稳定训练过程。通俗地说，它提醒模型：“你可以探索更好的输出，但别把之前学到的‘基本礼仪’全忘了。”

---

## 二、RLHF 中的关键角色

在RLHF的PPO训练中，有四个核心模型各司其职：

| 模型 | 角色 | 对应RL元素 | 说明 |
|------|------|-----------|------|
| **Actor模型** | 被优化的语言模型 | 智能体（Agent） | 根据prompt生成回复，不断优化策略 |
| **Reference模型** | 锚点模型 | - | SFT阶段的模型，用于计算KL散度惩罚 |
| **Critic模型** | 价值评估器 | 价值函数（Value Function） | 估计每个状态的价值，辅助计算优势函数 |
| **Reward模型** | 评分器 | 奖励函数（Reward Function） | 对生成的完整回复打分 |

Actor模型是RLHF的核心，它负责生成文本并接收奖励信号进行优化。Critic模型则通过**GAE**（Generalized Advantage Estimation）来计算每个token的优势值，帮助Actor更精准地判断哪些决策带来了更高的回报。

---

## 三、深入理解奖励建模

### 3.1 为什么奖励模型如此关键？

奖励模型是RLHF的“指挥棒”——它决定了什么样的输出是“好的”。如果奖励模型训练不当，整个RLHF流程都会偏离方向。

奖励模型本质上是一个**偏好编码器**：它接收人类标注的成对偏好数据，学习将抽象的人类价值观转化为可计算的数值信号。

### 3.2 奖励模型的训练细节

在实际训练中，奖励模型通常与策略模型共享相同的架构（如同样规模的Transformer），只是在顶层增加一个线性层来输出标量分数。训练时需要注意：

1. **数据质量**：偏好数据的质量直接影响奖励模型的效果
2. **过拟合防范**：将同一prompt的所有输出对作为单个batch元素，避免模型记住特定样本
3. **规模选择**：InstructGPT使用6B参数的奖励模型，在效果和效率之间取得平衡


## 四、核心挑战与解决方案

### 4.1 奖励黑客（Reward Hacking）

奖励黑客是RLHF最经典的失败模式：模型在优化过程中学会了“钻奖励模型的空子”——生成的输出获得了高奖励分数，但实际质量却在下降。

研究表明，激进的PPO训练中局部奖励黑客率可高达14.45%。奖励黑客不仅仅是最终的模型病理，更是一种可以在训练过程中被分类、定位和部分预测的动态现象。

**应对策略**：
- 引入KL散度惩罚，限制策略偏离
- 使用不确定性感知的PPO（UP-PPO），降低奖励黑客率
- 结合推理任务验证器（RTV）等混合奖励系统

### 4.2 响应多样性下降

随着RLHF训练的进行，模型往往会变得越来越“保守”——生成的输出多样性下降，倾向于产生千篇一律的回复。

**应对策略**：
- 在PPO目标中加入熵正则化项，鼓励探索
- 动态调节KL系数

### 4.3 训练不稳定性

PPO训练对超参数高度敏感，容易出现奖励震荡、熵崩溃、价值函数漂移等问题。

**应对策略**：
- 使用GAE进行稳定的优势估计
- 采用适当的裁剪系数（$\epsilon=0.2$是经验值）
- 监控KL散度、奖励均值等训练指标

---

## 五、RLHF 的演进与未来

### 5.1 DPO：轻量化的替代方案

2023年提出的**DPO**（Direct Preference Optimization，直接偏好优化）发现了一个令人惊讶的事实：我们可以**绕过显式的奖励模型训练**，直接用偏好数据优化语言模型。

DPO的核心洞察是：语言模型本身就可以被看作一个隐式的奖励模型。通过数学推导，可以将RLHF的奖励最大化问题转化为一个可以直接在偏好数据上优化的目标函数。

DPO的优势在于流程更简单、资源消耗更低，但在某些复杂对齐场景下，效果可能不如完整的PPO+RLHF流程。

### 5.2 RLAIF：用AI替代人工标注

**RLAIF**（Reinforcement Learning from AI Feedback，基于AI反馈的强化学习）是另一个重要方向——用大模型（如GPT-4）替代人类标注者来提供偏好判断。

研究表明，RLAIF可以达到与使用人类反馈相当的性能，为解决RLHF的标注成本问题提供了一个有希望的路径。Anthropic的**Constitutional AI**就是RLAIF的典型应用——模型根据一套预定义的“宪法”原则自主评估和改进输出。

### 5.3 在线RLHF与自适应优化

传统的RLHF使用固定的离线数据集进行训练。最新的研究方向是在线RLHF——模型在实际部署中持续接收用户反馈，实现动态优化。2026年提出的**ARF-RLHF**（Adaptive Reward-Following）更是将自然语言反馈转化为连续的偏好轨迹进行优化，在多个任务上超越了传统的PPO和DPO。

---

> [!note]
> 
> RLHF的核心思想可以概括为一句话：**将模糊的人类偏好转化为可计算的奖励信号，再用强化学习来优化模型**。
> 
> 从InstructGPT到ChatGPT，从Claude到最新的开源模型，RLHF已经成为大模型从“博学”走向“好用”的关键技术桥梁。它让模型从“死记硬背标准答案”变成了“理解评分标准并发挥创造力”。
> 
> 当然，RLHF并非万能药。奖励黑客、多样性下降、训练不稳定性等挑战依然存在。DPO、RLAIF、在线RLHF等新方法的涌现，正在不断拓展这一领域的技术边界。
> 
> 正如一位研究者所说：“RLHF的失败不仅仅是最终模型的病理，而是可以在训练过程中被分类、定位和部分预测的动态现象。”理解这些失败模式，正是我们不断改进这项技术的起点。