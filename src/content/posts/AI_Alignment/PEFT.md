---
title: PEFT (Parameter-Efficient Fine-Tuning) —— 参数高效微调
published: 2026-06-26
description: 系统解析PEFT技术体系，涵盖Adapter Tuning、Prefix Tuning、Prompt Tuning、LoRA、QLoRA、AdaLoRA、IA3等核心方法的数学原理、代码实现与选型指南，探讨低秩假设、矩阵分解、参数重参数化等深刻思想，以及前沿进展与未来方向。
cover: "/assets/images/posts/PEFT.png"
coverInContent: false
tags: [PEFT, LoRA, 后训练, 模型微调]
category: AI_Alignment
draft: false
---

# PEFT (Parameter-Efficient Fine-Tuning) —— 参数高效微调

> [!important]
>
> 原论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)、[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)、[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
>
> 优秀博客：[Understanding Parameter-Efficient LLM Finetuning: Prompt Tuning And Prefix Tuning](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient)、[LoRA fine-tuning Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)、[HuggingFace LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
>
> PEFT库：[HuggingFace PEFT](https://github.com/huggingface/peft)

## 引言：大模型微调的困境

近年来，大语言模型的规模呈指数级增长——从BERT的1亿参数到GPT-3的1750亿参数，再到今天动辄万亿参数的模型——**全量微调**（Full Fine-Tuning）的成本已高到令人望而却步。以GPT-3 175B为例，单次全参数微调需要TB级显存和数万GPU小时。与此同时，**绝大多数下游任务并不需要调整模型的全部参数**——我们往往只需要让模型 **“学会”某个特定领域的知识或能力**。

正是在这样的背景下，**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 技术应运而生。PEFT的核心思想是**冻结预训练模型的大部分参数，仅更新一小部分参数或引入少量新增参数**，从而以**极低的计算成本**实现与**全量微调**相当的性能。

本文将从数学原理、方法分类、核心算法和工程实践四个维度，系统性地解析PEFT技术体系。

---

## 一、PEFT的分类体系

根据**参数更新策略**的不同，PEFT方法可分为三大范式：

1. **选择性微调**（Selective Fine-Tuning）：**仅更新模型中的部分已有参数，其余参数全部冻结**。代表性方法包括**BitFit**（仅微调**偏置项**）和**DiffPruning**（通过**可微分掩码**选择重要参数）。

2. **附加式微调**（Additive Fine-Tuning）：**在模型中新增可训练的参数模块，原始模型参数完全冻结**。这是目前最主流的一类方法，包括：
   - **Adapter Tuning**：在**Transformer层间**插入小型**适配器**模块
   - **Prefix Tuning**：在**输入序列前**添加可训练的**前缀向量**
   - **Prompt Tuning**：在输入层添加可训练的**软提示**
   - **LoRA：在权重矩阵旁添加低秩分解矩阵**

3. **重参数化微调**（Reparameterized Fine-Tuning）：通过**数学变换**将参数更新重参数化为**更紧凑**的形式。LoRA本质上也是一种重参数化方法。

---

## 二、Adapter Tuning：最早的PEFT范式

**Adapter Tuning**是PEFT领域最早的经典方法之一。其核心思路是**在Transformer的每一层中插入小型神经网络模块（称为Adapter），训练时仅更新这些Adapter的参数**。

一个典型的Adapter模块是一个**瓶颈结构 —— 先通过降维层将特征维度压缩，再通过升维层恢复**：

```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model: int, reduction_factor: int = 16):
        super().__init__()
        self.down = nn.Linear(d_model, d_model // reduction_factor)
        self.up = nn.Linear(d_model // reduction_factor, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # 残差连接：原始输出 + Adapter输出
        return x + self.up(self.activation(self.down(x)))
```

Adapter Tuning的优势在于**结构简单、易于实现**，但缺点是**引入了额外的推理延迟**——每个Adapter模块都增加了前向传播的计算量。以7B模型为例，Adapter方法通常需要新增约3%-5%的参数。

---

## 三、Prefix Tuning与Prompt Tuning：用“提示”替代“调参”

### 3.1 Prefix Tuning

**Prefix Tuning**的核心思想是在**每一层的输入序列前**添加可训练的**前缀向量（Prefix Vectors）**。这些前缀向量本质上是**连续的虚拟令牌**，在训练过程中被优化以**引导模型生成合适的输出**。

Prefix Tuning的形式为 `[PREFIX; x; y]`，其中PREFIX是可训练的**前缀参数**。训练时，原始模型参数完全冻结，**仅更新前缀向量**。

### 3.2 Prompt Tuning

**Prompt Tuning**是Prefix Tuning的**轻量版**——仅在**输入层**添加可训练的**提示向量（soft prompts）**，而非每一层。因此，Prompt Tuning的参数规模**更小**——以7B模型为例，新增参数占比通常小于0.1%。

```python
# Prompt Tuning的简化示意
class PromptTuning(nn.Module):
    def __init__(self, num_virtual_tokens: int, embed_dim: int):
        super().__init__()
        # 可训练的软提示嵌入
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_virtual_tokens, embed_dim)
        )
        
    def forward(self, input_embeds):
        # 将软提示拼接到输入嵌入前面
        return torch.cat([self.prompt_embeddings, input_embeds], dim=0)
```

Prefix Tuning和Prompt Tuning的共同优势是**参数效率极高**——仅需更新0.1%以下的参数即可获得不错的性能。但缺点也同样明显：**性能上限有限**，在复杂任务上往往不及LoRA等方法。

---

## 四、LoRA：低秩适配的典范

**低秩适配（Low-Rank Adaptation, LoRA）** 是目前最流行、应用最广泛的PEFT方法。

### 4.1 关键假设

LoRA的核心洞察基于一个关键假设：**模型在适应下游任务时，权重更新矩阵 $\Delta W$ 本质上是低秩的**。也就是说，模型适应新任务时，权重的变化并不需要满秩矩阵来描述，**核心变化可以通过低秩矩阵来近似**。

这个发现意味着如果微调时的权重更新本质上是**低秩**的，那可以**用两个小矩阵的乘积来近似这个更新**，而不是去更新整个庞大的权重矩阵。

在原论文中，LoRA主要应用于Transformer的**注意力权重矩阵（特别是 $W_q$ 和 $W_v$）**。实验表明，**在查询（Query）和值（Value）投影上应用LoRA效果最好**，而在键（Key）投影上应用的效果相对有限。

### 4.2 数学原理

对于预训练模型中的一个**线性层**，其原始计算为：

$$y = W_0 x$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 是**预训练的权重矩阵**。

全参数微调会**直接更新** $W_0$，而**LoRA**的做法是**冻结 $W_0$**，转而学习一个**更新矩阵** $\Delta W \in \mathbb{R}^{d \times k}$，使得微调后的输出为：

$$y = W_0 x + \Delta W x$$

**LoRA**的核心创新在于**将 $\Delta W$ 分解为两个低秩矩阵的乘积**：

$$\Delta W = B \cdot A$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，且**秩 $r \ll \min(d, k)$**。

这样一来，原本需要学习 **$d \times k$ 个参数**，现在只需要学习 **$r \times (d + k)$ 个参数**。参数量从 $O(dk)$ 降至 $O(r(d+k))$。

**前向传播**的计算变为：

$$y = W_0 x + \Delta W x = W_0 x + B A x$$

**在训练过程中，$W_0$ 被冻结（不计算梯度），仅更新 $A$ 和 $B$**。

引入**缩放系数**后，完整的前向计算为：

$$y = W_0 x + \frac{\alpha}{r} \cdot B A x$$

其中 $\alpha$ 是**缩放因子**，用于控制LoRA更新的**强度**。实际更新幅度为 $\alpha/r$。在实际调参中，**$\alpha$ 通常设置为 $r$ 的2倍**。这个缩放机制允许我们在改变 $r$ 时保持**更新幅度的相对稳定**。


### 4.3 梯度传播

反向传播时，梯度仅流向低秩矩阵：

$$\frac{\partial \mathcal{L}}{\partial A} = \left(\frac{\partial \mathcal{L}}{\partial \Delta W}\right)^T \cdot B^T , \quad \frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial \Delta W} \cdot A^T$$

这种设计使得**基础模型参数的梯度恒为零**，反向传播的计算复杂度从 $O(d^2)$ 降至 $O(dr)$。

### 4.4 初始化策略

LoRA采用**精心设计的初始化策略**以保证训练稳定性：
- 矩阵 $A$ 使用**随机高斯分布初始化**
- 矩阵 $B$ 使用**全零初始化**
- 这样，在训练开始时 $\Delta W = BA = 0$，模型的行为与原始预训练模型完全一致，**避免了训练初期的剧烈扰动**。

### 4.5 秩的选择策略

秩 $r$ 是LoRA**最核心的超参数**，需要在**表达能力**与**计算效率**之间权衡：

| 秩的范围 | 适用场景 | 特点 |
|---------|---------|------|
| $r \leq 8$ | **特定任务适配**（风格迁移、领域适配） | 参数极少，泛化性有限 |
| $16 < r < 64$ | **通用任务微调**（指令跟随、问答） | 效率与效果的黄金平衡点 |
| $r \geq 64$ | **接近全参数微调效果** | 参数效率下降 |

实验表明，在LLaMA-7B上微调代码生成任务时，$r=16$ 即可达到全参数微调 92% 的效果，而参数量减少了 98%。

---

## 五、LoRA算法代码实现

### 5.1 Python代码实现LoRA

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank  # 缩放系数
        
        # 冻结原始权重（不可训练）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), 
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features), 
            requires_grad=False
        )
        
        # 可训练的低秩矩阵
        # A矩阵：随机初始化
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # B矩阵：零初始化，确保训练初期 ΔW = 0
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, x):
        # 原始输出（冻结权重）
        base_output = x @ self.weight.T + self.bias
        # LoRA增量
        # 注意：实际计算顺序是 x @ A^T @ B^T
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_output + lora_output
```

### 5.2 使用HuggingFace PEFT库

在实际工程中，我们通常使用HuggingFace的PEFT库，它封装了LoRA及其变体的完整实现：

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# 配置LoRA
peft_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # 指定要应用LoRA的模块
)

# 包装模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 输出：trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193
```

### 5.3 推理时的权重合并

在部署阶段，可以将LoRA权重合并回原始模型，消除额外的计算开销：

```python
# 加载LoRA适配器
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "path/to/lora-adapter")
# 合并权重
model = model.merge_and_unload()  # 现在 ΔW 已合并到 W0 中
```

---

## 六、LoRA的改进与优化

### 6.1 QLoRA：当量化遇上LoRA

**QLoRA（Quantized Low-Rank Adaptation）** 是LoRA的**量化增强版**，其核心思想用公式概括为**QLoRA = 4-bit量化基座 + 全精度LoRA适配器**

QLoRA将**预训练模型**的权重从32位或16位浮点数量化为**4位精度**，同时**保持LoRA适配器为全精度**。这种“双重优化”策略带来了**显著的显存节省**——QLoRA可以在单张48GB显存的GPU上微调65B参数的模型。

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config,
    device_map="auto",
)

# 应用LoRA（适配器保持全精度）
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, lora_config)
```

QLoRA的工程价值在于**极大地降低了微调大模型的硬件门槛**，使得在消费级GPU上进行百亿参数模型的微调成为可能。

### 6.2 AdaLoRA：自适应秩分配

标准LoRA在所有层和所有模块上**均匀分配**参数预算。然而，不同层、不同模块对微调的重要性是不同的。**AdaLoRA（Adaptive LoRA）** 通过**奇异值分解（SVD）** 参数化增量更新，并根据各权重矩阵的重要性**动态分配**秩预算。

AdaLoRA采用**三阶段训练调度**，在训练过程中**动态调整各模块的秩**，从而在相同参数预算下获得更好的性能。

### 6.3 IA3：通过抑制与放大内部激活进行微调

**IA3（Infused Adapter by Inhibiting and Amplifying Inner Activations）** 是一种极简的PEFT方法，通过在注意力机制的**Key和Value**以及前馈模块中**注入可学习的缩放向量**来调整激活值。

IA3的新增参数极少，但性能在某些场景下会显著下降。它代表了PEFT在**极端参数效率**方向上的探索。

---

## 七、方法对比与选型指南

| 方法 | 新增参数占比（7B模型） | 推理延迟 | 适用场景 |
|------|----------------------|---------|---------|
| **Adapter** | 3% ~ 5% | +5% ~ 15% | 需要结构解耦的场景 |
| **Prefix Tuning** | 0.1% ~0.3% | +10% ~ 30%显存 | 低数据量场景 |
| **Prompt Tuning** | <0.1% | 轻微增加 | 极简适配，参数最少 |
| **LoRA** | 0.5% ~ 2% | 无额外延迟（可融合） | **通用首选**，效果与效率最佳平衡 |
| **QLoRA** | 0.5% ~ 2% | 无额外延迟 | **显存受限**场景，大模型微调 |

---

> [!note]
> 
> PEFT技术彻底改变了我们与大模型交互的方式——从“重新训练整个模型”到“**轻量级适配**”，从“需要数百张GPU”到“单张消费级显卡即可完成”。**LoRA及其衍生方法**（如QLoRA、AdaLoRA）已成为这一领域的基石。
> 
> 理解PEFT不仅仅是学会调用几个API——它背后蕴含着**低秩假设**、**矩阵分解**、**参数重参数化**等深刻的数学思想。当你下次微调一个大模型时，不妨想一想：你真正需要更新的是哪些参数？有多少参数是可以“借”而不是“买”的？
> 
> 这或许正是PEFT带给我们最宝贵的思维方式——**在资源有限的世界里，学会用最小的代价撬动最大的价值**。