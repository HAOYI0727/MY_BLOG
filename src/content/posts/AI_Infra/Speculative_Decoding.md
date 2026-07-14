---
title: Speculative Decoding —— LLM推理优化技术（一）
published: 2026-04-12
description: 系统讲解推测解码的核心原理与变体方法：从自回归生成的“串行诅咒”出发，剖析草稿-验证机制如何用小模型快速生成候选token、大模型并行验证；深入推导拒绝采样保证输出分布与原模型一致的数学证明；分析加速比公式及接受率对性能的决定性影响；并介绍Medusa、EAGLE、Lookahead Decoding等变体方法及其适用场景。
cover: "/assets/images/posts/speculative_decoding.png"
coverInContent: false
tags: [推测解码, 拒绝采样, 推理优化]
category: AI_Infra
draft: false
---

# Speculative Decoding —— LLM推理优化技术（一）

## 引言：自回归生成的“串行诅咒”

大语言模型生成文本时，每生成一个token都要跑一次完整的前向传播——这个过程**本质上是串行的**。对于70B参数的模型，一次前向传播就要加载140GB的权重（FP16）。生成一个1000 token的回复，模型权重要从显存里**搬进搬出1000次**。

这就是自回归生成的“串行诅咒”：$K$个token需要$K$次串行的模型调用。

但仔细想想：**所有的token都同样难预测吗？** 显然不是。在一个句子里，“the cat is on the ___”后面跟“mat”几乎是确定的。这些“容易”的token，能不能用一个小模型快速猜出来，然后让大模型一次性全部验证？

**推测解码（Speculative Decoding）** 的核心思想正是如此：

> 用一个小而快的**草稿模型（Draft Model）** 快速生成一串候选token，然后用大**目标模型（Target Model）** 在一次前向传播中并行验证这串token，决定接受哪些、拒绝哪些。

如果草稿模型猜得准，一次大模型前向传播就能“白赚”好几个token——推理速度提升**2-3倍**，且**输出分布与原模型完全一致**。

---

## 一、核心思想：草稿-验证（Draft-then-Verify）

### 1.1 为什么能加速？

推测解码的加速逻辑建立在一个关键洞察上：**大模型并行验证多个token的时间，约等于串行生成一个token的时间**。

原因在于，Transformer解码阶段的瓶颈是**内存带宽**而非计算。模型权重从显存加载到计算单元需要时间，但一旦权重加载完毕，计算多个token的logits和计算一个token的logits，边际计算成本很小。

### 1.2 工作流程

推测解码的每一轮包含三步：

```
Step 1: 草稿生成（Draft）
        小模型（Draft Model）自回归生成 K 个候选token
        [t1, t2, t3, ..., tK]

Step 2: 并行验证（Verify）
        大模型（Target Model）一次前向传播，计算这 K 个token的logits
        同时额外计算下一个token的分布（用于可能的修正）

Step 3: 接受/拒绝（Accept/Reject）
        逐token比较两个模型的概率分布，决定接受哪些、拒绝哪个
        如果第 j 个token被拒绝，用大模型的分布重新采样替换
        然后以被拒绝的位置为起点，进入下一轮
```

### 1.3 草稿模型的选择

草稿模型需要在**速度**和**准确性**之间权衡：

| 草稿模型类型 | 速度 | 接受率 | 典型应用 |
|------------|------|--------|---------|
| 同架构小模型（如7B→1.5B） | 快 | 中高 | 通用场景 |
| N-gram模型 | 极快 | 较低 | 结构化文本 |
| 自推测（Self-Speculation） | 无额外模型开销 | 中 | 不想部署两个模型 |
| 微调专用草稿模型 | 快 | 高 | 特定领域数据 |

理想情况下，草稿模型应该与目标模型使用**相同的tokenizer**。

---

## 二、数学原理：拒绝采样保证分布一致性

推测解码最精妙的部分在于：**它通过一个数学上严格的拒绝采样（Rejection Sampling）机制，保证最终输出分布与直接使用目标模型自回归生成完全一致**。

### 2.1 单token的接受概率

设目标模型的概率分布为 $p(x)$，草稿模型的概率分布为 $q(x)$。对于草稿模型采样出的候选token $x \sim q$，我们以如下概率接受它：

$$\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

如果接受，直接输出 $x$。如果拒绝，则从以下分布中重新采样：

$$p_{\text{resample}}(x') = \frac{\max(0, p(x') - q(x'))}{\sum_{y} \max(0, p(y) - q(y))}$$

### 2.2 为什么这能保证分布一致？

我们来证明：经过上述接受/拒绝过程后，最终输出的分布恰好等于 $p$。

设最终输出为 $Y$。有两种情况：

1. **接受**：$Y = x$，其中 $x \sim q$，概率为 $\alpha(x)$
2. **拒绝并重采样**：从修正分布中采样 $Y = x'$

完整写出：

$$\begin{aligned}
P(Y = x) &= \underbrace{q(x) \cdot \alpha(x)}_{\text{接受}} + \underbrace{P(\text{拒绝}) \cdot p_{\text{resample}}(x)}_{\text{拒绝后重采样}} \\
&= q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + P(\text{拒绝}) \cdot \frac{\max(0, p(x) - q(x))}{Z}
\end{aligned}$$

其中 $Z = \sum_y \max(0, p(y) - q(y))$。

注意到 $q(x) \cdot \min(1, p(x)/q(x)) = \min(q(x), p(x))$。

而 $P(\text{拒绝}) = \sum_y q(y) \cdot (1 - \alpha(y)) = \sum_y \max(0, q(y) - p(y)) = Z$（因为 $\sum p = \sum q = 1$，两边差值相等）。

因此：

$$P(Y = x) = \min(q(x), p(x)) + Z \cdot \frac{\max(0, p(x) - q(x))}{Z} = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$

**证毕**。这个证明的关键在于：拒绝采样的修正项恰好补上了 $p$ 比 $q$ 多出的概率质量。

### 2.3 多token的扩展

对于长度为 $K$ 的草稿序列 $x_1, x_2, ..., x_K$，算法逐token进行上述拒绝采样：

1. 对 $x_1$ 做接受/拒绝判断
2. 如果 $x_1$ 被接受，继续判断 $x_2$
3. 一旦某个 $x_j$ 被拒绝，丢弃 $x_j$ 及之后所有草稿token，从修正分布中重新采样 $x_j$，本轮结束
4. 如果所有 $K$ 个token都被接受，额外从目标模型采样一个token（作为“奖励”）

### 2.4 接受率与草稿模型质量

**接受率（Acceptance Rate）** $\alpha$ 是推测解码最关键的性能指标——它衡量草稿模型的预测与目标模型有多一致。

对于单token，期望接受率为：

$$\mathbb{E}[\alpha] = \sum_x q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = 1 - \frac{1}{2} \sum_x |p(x) - q(x)| = 1 - D_{\text{TV}}(p, q)$$

其中 $D_{\text{TV}}$ 是总变差距离（Total Variation Distance）。**接受率越高，加速效果越好**。

实践中，同一家族的模型（如LLaMA 7B做草稿、LLaMA 70B做目标）通常能获得**0.6-0.8**的接受率，草稿长度 $K=4$ 时可实现约 **2倍** 的加速。

---

## 三、加速比分析

### 3.1 理论加速比公式

设：
- $t_{\text{target}}$ = 目标模型一次前向传播的时间
- $t_{\text{draft}}$ = 草稿模型生成一个token的时间
- $K$ = 每轮草稿长度
- $\alpha$ = 单token平均接受率

每轮中，草稿模型串行生成 $K$ 个token，耗时 $K \cdot t_{\text{draft}}$。目标模型做一次并行验证，耗时 $t_{\text{target}}$。

每轮平均生成的token数为：

$$\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

（这个公式的推导：$1 + \alpha + \alpha^2 + ... + \alpha^K$，其中额外的1是全部接受后的“奖励”token。）

因此，**理论加速比**为：

$$\text{Speedup} = \frac{\mathbb{E}[\text{tokens per round}] \cdot t_{\text{target}}}{K \cdot t_{\text{draft}} + t_{\text{target}}} = \frac{\frac{1 - \alpha^{K+1}}{1 - \alpha}}{\frac{K \cdot t_{\text{draft}}}{t_{\text{target}}} + 1}$$

### 3.2 数值示例

假设 $t_{\text{draft}} / t_{\text{target}} = 0.1$（草稿模型比目标模型快10倍），不同 $K$ 和 $\alpha$ 下的加速比：

| K | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|---|-------|-------|-------|-------|
| 3 | 1.38× | 1.78× | 2.00× | 2.25× |
| 5 | 1.38× | 1.96× | 2.42× | 3.00× |
| 7 | 1.31× | 1.99× | 2.64× | 3.55× |

**关键洞察**：
- 接受率 $\alpha$ 对加速比的影响是**决定性的**
- 存在最优草稿长度 $K$：太短浪费验证能力，太长后面token的接受率递减
- 实践中 $K=3$ 到 $K=6$ 通常是最优区间

---

## 四、代码示例：PyTorch实现推测解码

下面是一个简化的PyTorch实现，展示推测解码的核心逻辑：

```python
import torch
import torch.nn.functional as F

def speculative_decode(
    target_model,      # 大模型
    draft_model,       # 小模型（草稿模型）
    input_ids,         # 输入token ids, shape: (1, seq_len)
    max_tokens: int,   # 最大生成token数
    K: int = 5,        # 每轮草稿长度
    temperature: float = 1.0,
):
    """
    推测解码主循环
    返回：生成的token序列
    """
    generated = input_ids.clone()
    seq_len = input_ids.shape[1]
    
    while generated.shape[1] - seq_len < max_tokens:
        # ============ Step 1: 草稿模型生成 K 个token ============
        draft_tokens = []
        draft_logits_list = []
        
        # 用当前已生成序列作为context
        context = generated
        
        for _ in range(K):
            with torch.no_grad():
                # 草稿模型前向传播
                logits = draft_model(context)
                # 取最后一个token的logits
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(next_token.item())
                draft_logits_list.append(next_logits)
                
                # 拼接到context，继续生成下一个草稿token
                context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
        
        # ============ Step 2: 目标模型并行验证 ============
        # 将草稿token拼接到原始输入后面
        draft_ids = torch.tensor(draft_tokens, device=input_ids.device).unsqueeze(0)
        full_input = torch.cat([generated, draft_ids], dim=1)
        
        with torch.no_grad():
            target_logits = target_model(full_input)
        
        # 提取每个草稿token位置的目标模型logits
        target_logits_list = []
        for i in range(K):
            pos = generated.shape[1] + i  # 草稿token在完整序列中的位置
            target_logits_list.append(target_logits[0, pos, :] / temperature)
        
        # ============ Step 3: 接受/拒绝 ============
        accepted_count = 0
        
        for i in range(K):
            draft_prob = F.softmax(draft_logits_list[i], dim=-1)
            target_prob = F.softmax(target_logits_list[i], dim=-1)
            
            draft_token = draft_tokens[i]
            
            # 计算接受概率
            p_target = target_prob[draft_token].item()
            p_draft = draft_prob[draft_token].item()
            
            # 核心公式：α = min(1, p_target / p_draft)
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))
            
            # 随机决定是否接受
            if torch.rand(1).item() < accept_prob:
                # 接受这个草稿token
                generated = torch.cat([generated, 
                                       torch.tensor([[draft_token]], device=input_ids.device)], 
                                      dim=1)
                accepted_count += 1
            else:
                # 拒绝：从修正分布中重新采样
                # 修正分布: max(0, p_target - p_draft) 归一化
                corrected = torch.clamp(target_prob - draft_prob, min=0)
                if corrected.sum() > 0:
                    corrected = corrected / corrected.sum()
                    new_token = torch.multinomial(corrected, num_samples=1)
                else:
                    # 兜底：直接从目标分布采样
                    new_token = torch.multinomial(target_prob, num_samples=1)
                
                generated = torch.cat([generated, new_token], dim=1)
                break  # 遇到拒绝就停止本轮
        else:
            # 所有K个token都被接受 → 额外采样一个"奖励"token
            # 用目标模型在最后一个位置的下一个token分布采样
            last_pos = full_input.shape[1] - 1
            next_logits = target_logits[0, last_pos, :] / temperature
            next_prob = F.softmax(next_logits, dim=-1)
            bonus_token = torch.multinomial(next_prob, num_samples=1)
            generated = torch.cat([generated, bonus_token], dim=1)
        
        # 检查是否达到最大长度或遇到EOS
        if generated.shape[1] - seq_len >= max_tokens:
            break
    
    return generated
```

**代码关键点解读**：

1. **草稿生成阶段**：小模型自回归生成K个token，每一步都依赖上一步的输出
2. **并行验证阶段**：大模型一次前向传播处理包含所有草稿token的完整序列
3. **接受/拒绝逻辑**：逐token比较 $p_{\text{target}}$ 和 $p_{\text{draft}}$，接受概率为 $\min(1, p_{\text{target}}/p_{\text{draft}})$
4. **修正采样**：拒绝时从 $\max(0, p_{\text{target}} - p_{\text{draft}})$ 中重采样，保证分布一致性

**vLLM中的配置示例**：在实际生产环境中，vLLM等推理框架已原生支持推测解码：

```python
from vllm import LLM, SamplingParams

# 配置推测解码
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",  # 目标模型
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",  # 草稿模型
    num_speculative_tokens=5,  # K=5
)

# 正常使用即可，推测解码在底层自动进行
outputs = llm.generate(prompts, sampling_params)
```

---

## 五、变体方法

### 5.1 Medusa：多头预测

Medusa的核心思想是：**不引入额外的草稿模型，而是在目标模型上添加多个并行的预测头（Decoding Heads）** 。

```
传统推测解码：目标模型 + 草稿模型（两个独立模型）
Medusa：目标模型 + 多个预测头（共享主干网络）
```

每个Medusa头预测不同偏移位置的token：
- Head 0：预测下一个token（同原始输出层）
- Head 1：预测下下个token
- Head 2：预测下下下个token
- ...

这些头的预测结果通过**树形注意力（Tree Attention）** 组合成多个候选路径，然后用典型接受方案选择最长的有效前缀。

Medusa的优势：
- **无需额外模型**：避免了部署两个大模型的开销
- **参数高效训练**：只训练新增的预测头
- **采样场景更优**：放宽了分布匹配要求

Medusa在Vicuna-7B上可实现约 **2倍** 的加速。

### 5.2 EAGLE：特征不确定性建模

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）基于一个关键洞察：**草稿生成的不确定性主要来自特征（feature）层面，而非token采样**。

EAGLE在目标模型上添加一个**轻量级的Transformer解码器层**作为草稿生成模块。这个模块利用目标模型的中间特征来预测后续token，比完整的小模型更高效。

EAGLE-3可实现 **2-6倍** 的推理加速。

### 5.3 Lookahead Decoding：N-gram模式匹配

Lookahead Decoding是一种**无需训练**的推测解码方法。它通过检测输入中的**N-gram模式**来预测后续token：

1. 扫描输入上下文中的重复N-gram模式
2. 基于历史模式生成候选token序列
3. 用目标模型验证这些候选

这种方法特别适合**代码生成**、**结构化文本**等具有强重复模式的场景，可实现 **2-4倍** 加速。

### 5.4 方法对比

| 方法 | 是否需要训练 | 额外模型开销 | 加速比 | 适用场景 |
|------|------------|------------|--------|---------|
| 标准推测解码 | 否 | 需要部署草稿模型 | 2-3× | 通用 |
| Medusa | 是（仅训练头） | 极小（额外头） | ~2× | 不想部署双模型 |
| EAGLE | 是（轻量层） | 小（单层） | 2-6× | 追求极致加速 |
| Lookahead | 否 | 无 | 2-4× | 结构化/重复文本 |

---

## 六、总结：一张图看懂推测解码

```
┌─────────────────────────────────────────────────────────────────────┐
│                          传统自回归解码                               │
│    Step 1: [大模型] → token1  Step 2: [大模型] → token2  Step 3: ...  │
│    每步都要加载140GB权重（70B模型），内存带宽成为瓶颈                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     推测解码（Draft-then-Verify）                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 草稿阶段（串行，小模型，快）                                      │   │
│  │ [小模型] → t1 → [小模型] → t2 → [小模型] → t3 → ... → tK       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 验证阶段（并行，大模型，一次前向传播）                             │   │
│  │ [大模型] 并行计算 t1, t2, ..., tK 的分布                        │   │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 接受/拒绝（拒绝采样保证分布一致）                                 │   │
│  │ α = min(1, p_target / p_draft)                              │   │
│  │ 接受 → 保留token  拒绝 → 从修正分布重采样                        │   │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│      效果：大模型一次前向传播 → 平均获得 1+α+α²+...+α^K 个token           │
│      典型加速比：2-3×                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**核心要点**：
1. **推测解码用一个快模型“猜”、一个慢模型“验”** ，利用解码阶段的内存带宽瓶颈特性，让大模型一次前向传播验证多个token
2. **拒绝采样机制保证了数学上的无损**：输出分布与原模型完全一致
3. **接受率是加速比的决定性因素**：$\text{Speedup} = \frac{(1-\alpha^{K+1})/(1-\alpha)}{K \cdot t_{\text{draft}}/t_{\text{target}} + 1}$
4. **不存在银弹**：Medusa无需额外模型但需要训练，EAGLE加速比更高但增加系统复杂度，Lookahead零训练但依赖数据模式