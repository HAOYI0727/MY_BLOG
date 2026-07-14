---
title: SFT (Supervised Fine-Tuning)—— 监督微调
published: 2026-06-28
description: 深入剖析监督微调（SFT）的核心原理，涵盖最大似然估计与交叉熵损失的数学推导、Loss Masking 代码实现、灾难性遗忘与不完全学习现象（ILP）等系统挑战，以及数据质量、分层学习率等最佳实践。全面理解 SFT 作为 LLM 后训练基石的定位、优势与演进方向。
cover: "/assets/images/posts/SFT.png"
coverInContent: false
tags: [SFT, 微调, 后训练]
category: AI_Alignment
draft: false
---

# SFT (Supervised Fine-Tuning)—— 监督微调

## 引言

如果你训练过大语言模型，一定经历过这样的时刻：花了几天时间做监督微调（Supervised Fine-Tuning, SFT），loss 降得很漂亮，benchmark 分数也在涨，但把训练集中的样本重新喂给模型，总有一批样本它死活答不对。这不是你的错——这是 SFT 范式中一个长期被忽视的系统性缺陷。

但这并不意味着 SFT 不值得深入理解。恰恰相反，SFT 是目前几乎所有大语言模型（LLM）应用落地的核心环节，是连接预训练模型与具体任务的桥梁。从 ChatGPT 到 Claude，从开源社区到工业界，SFT 都是模型后训练（post-training）的第一步。

本文将带你从头理解 SFT 的数学原理、优化目标、代码实现，以及它面临的最新挑战。无论你是刚入门的大模型开发者，还是想深入理解技术本质的研究者，希望这篇文章都能给你带来启发。

---

## 一、什么是监督微调？

### 1.1 从预训练到微调

预训练模型（如 GPT 系列）通过海量无监督数据学习了通用的语言表征——它知道如何“接话”，但不知道如何“回答问题”。

举个例子：你给一个预训练模型输入“中国的首都是”，它能自然地补全“北京”。但如果你问它“中国的首都是哪里？”，它可能继续补全“是北京”而不是给出一个完整的回答。预训练模型学会了语言建模，但没有学会“指令遵循”。

SFT 要做的就是：**用大量（指令，回答）配对数据，教会模型“当用户给出指令时，应该如何回应”**。

### 1.2 SFT 在 LLM 训练流程中的位置

在典型的 LLM 后训练流程中，SFT 处于承上启下的位置：

```
预训练（Pre-training）→ SFT（监督微调）→ RLHF / DPO（偏好对齐）
```

- **预训练**：模型学习通用知识（海量文本，无监督）
- **SFT**：模型学习指令遵循（人工标注的问答对，有监督）
- **RLHF/DPO**：模型学习人类偏好（偏好数据，强化学习/偏好优化）

没有 SFT，后续的 RLHF 会非常困难，因为模型根本不知道“什么是回答”。

---

## 二、SFT 的数学原理

### 2.1 核心思想：最大似然估计（MLE）

SFT 的核心目标可以一句话概括：**让模型在给定输入 x 时，生成目标输出 y 的概率尽可能大**。

假设我们有一个专家数据集：

$$D = \{(x_1, y_1^*), (x_2, y_2^*), ..., (x_N, y_N^*)\}$$

其中 $x_i$ 是输入（如用户的问题），$y_i^*$ 是专家给出的标准回答。

我们训练一个模型 $\pi_\theta(y|x)$，它表示在参数 $\theta$ 下，给定输入 $x$ 生成输出 $y$ 的概率。

最朴素的想法是：**让模型在专家答案上的概率最大化**。数学上，这可以写成：

$$\max_{\theta} \prod_{(x, y^*) \in D} \pi_\theta(y^*|x)$$

这就是**最大似然估计（Maximum Likelihood Estimation, MLE）** 。

### 2.2 为什么要取对数？

直接优化乘积有一个严重的问题：每个 $\pi_\theta(y^*|x)$ 都是一个介于 0 和 1 之间的概率，当样本量很大时，这些概率的乘积会趋近于 0，导致**数值下溢**（计算机无法精确表示如此小的数）。

更不用说，乘积的求导计算非常复杂。

解决方法很简单：**取对数**。

因为对数函数是单调递增的（如果 $a > b$，则 $\log a > \log b$），最大化似然等价于最大化对数似然：

$$\log \mathcal{L}(\theta) = \sum_{(x, y^*) \in D} \log \pi_\theta(y^*|x)$$

乘积变成了求和，数值稳定性大大提高，求导也变得更简单。

### 2.3 从最大化到最小化：负对数似然（NLL）

在机器学习中，我们习惯**最小化损失函数**而不是最大化目标函数。所以把符号取反：

$$L_{\text{SFT}}(\theta) = -\sum_{(x, y^*) \in D} \log \pi_\theta(y^*|x)$$

这就是**负对数似然（Negative Log-Likelihood, NLL）** 损失函数。

写成期望形式：

$$L_{\text{SFT}}(\theta) = \mathbb{E}_{(x, y^*) \sim D}\left[-\log \pi_\theta(y^*|x)\right]$$

**最大化似然 ⇔ 最大化对数似然 ⇔ 最小化负对数似然**。

### 2.4 与交叉熵损失的关系

你可能已经发现了——这个式子就是**交叉熵损失**（Cross-Entropy Loss）。

对于语言模型，输出是一个 token 序列 $y = (y_1, y_2, ..., y_T)$，模型在每个位置预测下一个 token 的概率分布。根据概率链式法则：

$$\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t | x, y_{<t})$$

代入负对数似然：

$$L_{\text{SFT}}(\theta) = -\sum_{(x, y^*) \in D} \sum_{t=1}^{T} \log \pi_\theta(y_t^* | x, y_{<t}^*)$$

这就是**逐 token 的交叉熵损失之和**。

### 2.5 梯度的直观理解

对参数 $\theta$ 求导：

$$\nabla_\theta L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y^*) \sim D}\left[\nabla_\theta \log \pi_\theta(y^*|x)\right]$$

这个梯度告诉模型：**朝着让专家答案概率增加的方向更新参数**。

在统一的梯度优化范式下，SFT 的梯度系数（Gradient Coefficient）恒为 1——没有奖励加权，没有偏好调制，就是纯粹的“模仿”。

---

## 三、SFT 的代码实现

理论讲完了，来看看代码。以下是一个简化的 SFT 训练核心逻辑。

### 3.1 Loss Masking：只计算回答部分的损失

SFT 中一个容易被忽视的关键细节是：**我们只希望在“回答”部分计算损失，而不希望在“提问”部分计算损失**。

为什么要这样做？

因为 SFT 的目标是让模型学会“回答问题”，而不是“背诵提问”。如果提问部分也参与损失计算，模型可能会学会机械地复述问题，而不是理解并回答它。

在 PyTorch 中，`CrossEntropyLoss` 有一个 `ignore_index` 参数（默认值为 -100），我们可以把属于 Prompt 和 Padding 部分的 token 全部设为 -100，这样它们就不会产生梯度。

### 3.2 核心代码

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, tokens, loss_mask):
    """
    计算 SFT 损失
    
    Args:
        logits: 模型输出，形状 (batch_size, seq_len, vocab_size)
        tokens: 输入 token，形状 (batch_size, seq_len)
        loss_mask: 掩码，1 表示需要计算损失的位置（回答部分），
                   0 表示忽略（提问部分），形状 (batch_size, seq_len)
    
    Returns:
        仅在回答 token 上计算的平均交叉熵损失
    """
    # Shift: 用位置 t 的 logits 预测位置 t+1 的 token
    logits = logits[:, :-1, :]  # 去掉最后一个位置
    targets = tokens[:, 1:]      # 去掉第一个位置
    mask = loss_mask[:, 1:].to(logits.dtype)
    
    # 计算交叉熵（不自动求平均）
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1).long(),
        reduction="none"
    )
    
    # 重塑为原始形状并应用掩码
    ce = ce.view(targets.shape) * mask
    
    # 仅在回答 token 上求平均
    return ce.sum() / mask.sum().clamp(min=1.0)
```

这段代码改编自开源 SFT 实现。核心逻辑是：

1. **Shift**：用位置 $t$ 的 logits 预测位置 $t+1$ 的 token（自回归语言模型的标准做法）
2. **Mask**：只对回答部分的 token 计算损失
3. **平均**：在回答 token 的数量上求平均，而不是在整个序列长度上

### 3.3 完整训练循环示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# 准备数据（简化版）
def prepare_sft_data(instruction, response):
    """构造 SFT 训练数据，返回 tokens 和 loss_mask"""
    # 构造对话格式
    full_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    
    # 构造 loss_mask：只在 assistant 回答部分计算损失
    # （实际实现中需要根据 token 位置精确构造，此处仅为示意）
    loss_mask = torch.ones_like(tokens)
    # ... 将 prompt 部分的 mask 设为 0
    
    return tokens, loss_mask

# 训练循环
optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        logits = model(batch["tokens"])  # 前向传播
        loss = sft_loss(logits, batch["tokens"], batch["loss_mask"])
        
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        optimizer.zero_grad()
```

### 3.4 一个常见的坑

SFT 训练中最容易写错的代码是什么？**Prompt Masking**。

很多初学者会把整个序列（包括提问）的损失都算进去，导致模型学会的是“如何复述问题+回答”而不是“如何回答问题”。正确的做法是**只对 assistant 的回复部分计算损失**。

---

## 四、SFT 的优势与局限

### 4.1 SFT 的优势

**1. 简单直接**

SFT 本质上就是标准的监督学习——给模型看“问题-答案”对，让它模仿。不需要设计奖励模型，不需要做策略梯度采样，不需要处理不稳定的训练过程。

**2. 训练快速稳定**

交叉熵损失提供了清晰、一致的监督信号，梯度稳定，收敛快。

**3. 数据效率高**

相比从头训练，SFT 仅需千级到万级标注样本即可达到较好效果。在低数据场景下（<1000 个标注样本），仅靠 SFT 就能取得主导性能。

### 4.2 SFT 的局限

**1. 灾难性遗忘（Catastrophic Forgetting）**

SFT 在帮助模型获得新技能的同时，往往会导致模型遗忘预训练阶段学到的一些通用能力。研究发现，SFT 后的模型在上下文感知能力上会出现显著下降。

**2. 不完全学习现象（Incomplete Learning Phenomenon, ILP）**

这是 ACL 2026 上最新揭示的一个系统性缺陷：即使训练 loss 已经收敛、benchmark 分数在涨，模型中仍有 **15.3%±2.1%** 的训练样本处于“未学习”状态。这些样本在训练集上重新测试时，模型就是答不对。

ILP 不是过拟合、不是灾难性遗忘、不是数据噪声，而是一个独立的新问题。研究团队将其归因为五类根因：预训练知识缺失、预训练知识冲突、数据内部矛盾、左侧遗忘、优化不足。

**3. 泛化能力有限**

相比强化学习（RL）微调，SFT 在新输入上的泛化能力较弱，尤其是当训练集和测试集的分布差异较大时。

**4. 只能模仿，不能超越**

SFT 本质上是“模仿专家”，模型的上限被训练数据的质量所限制。它无法像 RL 那样通过探索发现超越人类标注者的解决方案。

---

## 五、SFT 的最新研究方向

### 5.1 缓解灾难性遗忘

研究者提出了多种方案来减轻 SFT 导致的灾难性遗忘：

- **LP-SFT**（Local-Preserving SFT）：通过保持预训练模型的熵结构来保护已有能力
- **InfoSFT**：通过信息感知的 token 加权，将学习信号集中在信息量最大的 token 上
- **RAFT**：数据精炼与自适应蒸馏，在领域微调中恢复部分通用能力

### 5.2 重新思考 SFT 的目标函数

传统的 SFT 使用负对数似然（NLL）作为目标函数，但研究者发现 NLL 可能是 SFT 泛化能力受限的原因之一。

新的研究开始探索：
- **概率视角的目标函数**：超越对数似然
- **Q-target 框架**：将 SFT 监督分解为“对观测 token 的依赖强度”和“剩余概率质量的分配方式”两个维度
- **f-散度目标**：在优化过程中保留 KL 散度项，更有效地约束模型更新

### 5.3 SFT 与 RL 的统一视角

最新的理论研究揭示了 SFT 与偏好学习（如 DPO）之间的深层联系：**SFT 本质上是隐式奖励学习的一个特例**。

SFT 的梯度更新实际上等价于一种带有“缺陷奖励机制”的强化学习——这个缺陷会导致模型过分关注那些它本不确定的样本，从而引发训练不稳定和泛化能力差的问题。

这一视角为改进 SFT 提供了新思路：通过调整学习率或修改目标函数，可以在 SFT 阶段就引入类似 RL 的正则化效果。

---

## 六、实践建议

### 6.1 数据质量 > 数据数量

在 SFT 中，数据质量比数量更重要。高质量标注数据需要满足：

- **覆盖度**：涵盖所有关键业务场景
- **平衡性**：正负样本比例控制在合理范围
- **一致性**：标注规范明确，标注者之间的一致性高（Kappa > 0.75）

### 6.2 分层学习率

不同层的参数应该使用不同的学习率：

- **底层参数**（词嵌入层）：1e-5 量级（微调）
- **中间层**：1e-4 量级
- **顶层任务头**：1e-3 量级（强化学习）

### 6.3 早停与验证

使用早停机制避免过拟合：

```python
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,      # 连续3次验证不提升则停止
    early_stopping_threshold=0.001  # 最小改进阈值
)
```

### 6.4 诊断不完全学习

如果你的模型在训练集上表现不佳，可以尝试：

1. **打乱数据顺序 + 自适应 epoch 停止**：这是零成本的优化方案
2. **使用 MC 转换检测**：将开放生成问题转化为选择题，用多轮采样判断模型是否真正学会了
3. **针对不同根因采取不同干预**：知识缺失→知识增强，数据矛盾→数据清洗

---

> [!note]
> 
> SFT 看似简单——不就是监督学习吗？但深入了解后你会发现，它的数学根基（最大似然估计）、工程实现（loss masking）、以及它面临的挑战（灾难性遗忘、不完全学习），每一个层面都值得认真对待。
> 
> SFT 是 LLM 后训练的基石。理解 SFT，不仅是为了会用，更是为了理解整个 LLM 对齐（Alignment）技术栈的起点。从 SFT 到 RLHF，从模仿到探索，这是一条仍在快速演进的技术路线。