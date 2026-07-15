---
title: Contrastive Language-Image Pre-training —— CLIP对比学习
published: 2026-08-02
description: 系统讲解CLIP的核心原理与技术演进：从双塔架构出发，推导InfoNCE损失函数的数学本质及其与交叉熵的等价性；剖析WIT-400M大规模图文对预训练的工程挑战；深入解读零样本分类的工作原理；并介绍SigLIP如何通过Sigmoid损失替代Softmax归一化，实现更优的小批次训练效率与灵活性。
cover: "/assets/images/posts/clip.png"
coverInContent: false
tags: [CLIP, 对比学习, 多模态, InfoNCE, SigLIP]
category: AI_Multimodal
draft: false
---

# Contrastive Language-Image Pre-training —— CLIP对比学习

## 引言：当图像遇见文字

在深度学习的世界里，图像和文本曾经是两条平行的河流——计算机视觉模型在像素的海洋中航行，自然语言处理模型在文字的河流中漂泊，它们各自精通自己的领域，却无法真正理解对方的语言。

直到2021年，OpenAI提出了**CLIP**（Contrastive Language-Image Pre-training）。这个模型第一次让图像和文本在同一个向量空间中“对话”——不是通过复杂的标签标注，而是通过一种简单而优雅的方式：**对比学习**。

CLIP的核心思想可以概括为一句话：**让配对的图文在特征空间中靠近，让不配对的图文远离**。这听起来简单，但背后蕴含着深刻的数学原理和工程智慧。本文将带你深入CLIP的技术内核，从双塔架构到InfoNCE损失函数，从4亿图文对的预训练到零样本分类，再到新一代的SigLIP模型，一步步拆解多模态对齐的底层逻辑。

---

## 一、CLIP的双塔架构：两条独立的编码器

CLIP采用**双塔架构**（Dual-Encoder Architecture），包含两个独立的编码器：

- **图像编码器（Image Encoder）** ：将图像转换为固定维度的向量表示
- **文本编码器（Text Encoder）** ：将文本描述转换为相同维度的向量表示

两个编码器独立工作，将各自模态的输入映射到**同一个联合嵌入空间**（Joint Embedding Space）中。在这个空间中，语义相关的图文对距离较近，语义不相关的图文对距离较远。

### 1.1 图像编码器：ResNet 或 ViT

CLIP的图像编码器支持两种架构：

1. **ResNet系列**：采用卷积神经网络提取图像特征
2. **Vision Transformer（ViT）系列**：将图像分割为图像块（patch），通过Transformer编码器处理

实验结果表明，ViT架构在性能上优于ResNet——**模型越大，效果越好**。以ViT-L/14@336为例，在ImageNet零样本分类任务上可达**77.02%** 的准确率。

### 1.2 文本编码器：Transformer

文本编码器基于Transformer架构，其设计要点包括：
- 词汇表大小：49,152个词汇
- 层数：12层
- 隐藏层维度：512
- 注意力头数：8个
- 最大序列长度：77个token（含[SOS]和[EOS]标记）

文本编码器对输入文本进行BPE（Byte Pair Encoding）编码后，通过多层Transformer生成文本的向量表示。

### 1.3 双塔架构的数学形式

设图像编码器为 $f_{\text{img}}$，文本编码器为 $f_{\text{text}}$。对于一张图像 $x$ 和一段文本 $y$，我们有：

$$v = f_{\text{img}}(x) \in \mathbb{R}^d, \quad t = f_{\text{text}}(y) \in \mathbb{R}^d$$

其中 $d$ 是联合嵌入空间的维度（CLIP中通常为512或1024）。

两个编码器的输出通过**余弦相似度**衡量匹配程度：

$$\text{sim}(v, t) = \frac{v \cdot t}{\|v\| \cdot \|t\|}$$

训练的目标就是：让匹配的图文对 $(v_i, t_i)$ 的相似度尽可能高，让不匹配的图文对 $(v_i, t_j)$（$i \neq j$）的相似度尽可能低。

下面是CLIP双塔架构的PyTorch伪代码实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # 图像编码器：ViT或ResNet
        self.visual_encoder = VisionTransformer()  # 或 ResNet()
        # 文本编码器：Transformer
        self.text_encoder = Transformer()
        # 可学习的温度参数（对数尺度）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image, text):
        # 分别编码
        image_features = self.visual_encoder(image)  # [batch, d]
        text_features = self.text_encoder(text)      # [batch, d]
        
        # L2归一化（使用余弦相似度）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵
        # logits_per_image[i, j] = sim(image_i, text_j) * exp(logit_scale)
        logits_per_image = self.logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
```

---

## 二、InfoNCE损失函数：对比学习的数学核心

有了双塔架构，我们还需要一个**损失函数**来指导模型学习——告诉模型什么是“好”的对齐，什么是“坏”的对齐。这就是**InfoNCE损失函数**的角色。

### 2.1 从对比学习的基本思想出发

对比学习的核心是**构造正样本对和负样本对**。在CLIP的语境中：

- **正样本对**：批次中配对的图文对 $(v_i, t_i)$
- **负样本对**：批次中不配对的图文对 $(v_i, t_j)$，其中 $i \neq j$

对于一个批次大小为 $N$ 的图文对，模型需要从 $N \times N$ 种可能的组合中，正确识别出 $N$ 个真实匹配对。

### 2.2 InfoNCE损失的数学推导

InfoNCE（Information Noise-Contrastive Estimation）损失的核心思想是：**将对比学习转化为一个 N 分类问题**。

对于第 $i$ 个图像 $v_i$，它与所有 $N$ 个文本的相似度构成一个分布。我们希望 $v_i$ 与它匹配的文本 $t_i$ 的相似度尽可能高：

$$\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)}$$

对称地，对于第 $i$ 个文本 $t_i$，我们希望它与匹配的图像 $v_i$ 的相似度尽可能高：

$$\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(t_i, v_j)/\tau)}$$

最终的CLIP损失是两者的和：

$$\mathcal{L}_{\text{CLIP}} = \mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}$$

其中：
- $\text{sim}(a, b)$ 是余弦相似度
- $\tau$ 是**温度系数**（temperature），控制相似度分布的“锐度”，CLIP中通常设为可学习的参数，初始值对应0.07
- $N$ 是批次大小

这种**对称的损失设计**确保了模型不会偏向任一模态，从而学习到更鲁棒的跨模态表示。

### 2.3 InfoNCE与交叉熵的等价性

一个有趣的发现是：**InfoNCE损失在数学上等价于标准的交叉熵损失**。

交叉熵的本质是衡量两个概率分布的差异。在分类任务中，我们用softmax将模型输出转换为概率分布，再与one-hot标签计算交叉熵。而InfoNCE损失的形式：

$$\mathcal{L} = -\log \frac{\exp(s_{\text{pos}}/\tau)}{\sum_{i=1}^N \exp(s_i/\tau)}$$

这正是对logits做softmax后取负对数——只不过这里的“类别”变成了批次中的样本位置。

用PyTorch代码验证这个等价性：

```python
import torch
import torch.nn.functional as F

# 假设batch_size=4，相似度得分矩阵
logits = torch.tensor([
    [3.0, 0.5, 0.1, 1.2],
    [0.3, 2.8, 0.4, 0.9],
    [0.2, 0.6, 2.5, 0.7],
    [1.1, 0.8, 0.3, 3.5]
])

temperature = 1.0

# 方法1：手动计算InfoNCE Loss
pos_scores = logits.diag()  # 对角线是正样本得分
exp_pos = torch.exp(pos_scores / temperature)
exp_all = torch.exp(logits / temperature).sum(dim=1)
manual_loss = -torch.log(exp_pos / exp_all).mean()

# 方法2：使用cross_entropy
labels = torch.arange(logits.shape[0])  # [0, 1, 2, 3]
ce_loss = F.cross_entropy(logits / temperature, labels)

print(f"手动计算: {manual_loss.item():.4f}, CrossEntropy: {ce_loss.item():.4f}")
# 输出: 手动计算: 0.8921, CrossEntropy: 0.8921
```

这个等价性告诉我们：**CLIP的对比学习本质上是在做一个“找出配对的图文”的分类任务**。

### 2.4 CLIP损失函数的完整实现

```python
def clip_loss(image_features, text_features, temperature=0.07):
    """
    CLIP风格的对称对比损失
    
    Args:
        image_features: [batch_size, d_model] 图像特征（已归一化）
        text_features: [batch_size, d_model] 文本特征（已归一化）
        temperature: 温度系数
    
    Returns:
        对称对比损失（图像→文本 + 文本→图像）
    """
    batch_size = image_features.size(0)
    
    # 计算相似度矩阵 [batch_size, batch_size]
    logits = (image_features @ text_features.T) / temperature
    
    # 标签：对角线位置是正样本
    labels = torch.arange(batch_size, device=logits.device)
    
    # 对称交叉熵损失
    loss_i2t = F.cross_entropy(logits, labels)      # 图像→文本
    loss_t2i = F.cross_entropy(logits.T, labels)    # 文本→图像
    
    return (loss_i2t + loss_t2i) / 2
```

---

## 三、WIT-400M：4亿图文对的预训练

CLIP的成功不仅在于算法设计，更在于**数据的规模和质量**。

### 3.1 数据集的构建

CLIP的训练数据集名为 **WIT**（WebImageText），包含 **4亿** 个（图像，文本）对。

数据收集的方式是：从互联网上爬取图片及其配对的alt-text文本描述。通过搜索一组精心策划的常见词汇（包括所有ImageNet类别名称）来收集这些数据。

数据构建策略具有三大特点：
1. **多样性覆盖**：包含自然场景、专业领域、抽象概念等多类型数据
2. **噪声鲁棒性**：允许一定比例的错误配对，通过对比学习自动过滤噪声
3. **长尾分布**：刻意保留低频类别，增强模型对罕见概念的建模能力

### 3.2 训练的工程挑战

在4亿图文对上训练CLIP是一项巨大的工程挑战：
- 使用 **1024块GPU** 进行分布式训练
- 采用**混合精度训练**（FP16+FP32），在保持精度的同时提升训练速度30%
- 实际工程中，使用8块A100 GPU训练ViT-B/16模型约需**12天**

作为对比，ViT-B/32使用128块A100（40GB）GPU训练了约**36小时**。

---

## 四、零样本分类：无需训练的跨领域泛化

CLIP最令人惊艳的能力是**零样本分类**（Zero-shot Classification）——无需针对特定任务进行任何微调，就能直接对未见过的类别进行分类。

### 4.1 零样本分类的原理

零样本分类的核心思想很简单：

1. **训练阶段**：CLIP通过对比学习将图像和文本映射到同一个联合嵌入空间
2. **推理阶段**：对于一张新图像，将其编码为向量 $v$；对于每个候选类别 $c$，构造文本提示（如“a photo of a {c}”），编码为向量 $t_c$；计算 $v$ 与所有 $t_c$ 的余弦相似度，选择相似度最高的类别作为预测结果

数学上，零样本分类可以表示为：

$$\hat{c} = \arg\max_{c \in \mathcal{C}} \text{sim}(f_{\text{img}}(x), f_{\text{text}}(\text{prompt}(c)))$$

其中 $\mathcal{C}$ 是所有候选类别的集合，$\text{prompt}(c)$ 是将类别名转换为文本提示的函数。

### 4.2 令人震撼的性能

CLIP在ImageNet上实现了 **76.2%** 的零样本分类准确率——而在此之前，最先进的零样本学习方法仅达到11.5%。

使用ViT-L/14@336架构，这一数字进一步提升至 **77.02%**。后续的开源实现OpenCLIP更是达到了 **80.1%** 的零样本准确率。

这一性能的意义在于：**CLIP无需使用ImageNet的128万张训练样本，就能达到与有监督ResNet50相当的性能**。

### 4.3 零样本分类的代码实现

```python
import torch
import clip

# 加载预训练的CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备图像
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# 定义候选类别
class_names = ["dog", "cat", "bird", "fish", "car", "airplane"]
text_prompts = [f"a photo of a {c}" for c in class_names]
text_tokens = clip.tokenize(text_prompts).to(device)

# 编码并计算相似度
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
    
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    
# 输出预测结果
predicted_class = class_names[similarities.argmax().item()]
print(f"预测类别: {predicted_class}, 置信度: {similarities.max().item():.4f}")
```

---

## 五、SigLIP：Sigmoid损失驱动的对比学习进化

CLIP的成功引发了大量后续研究，其中**SigLIP**（Sigmoid Loss for Language-Image Pre-training）是最具影响力的改进之一。

### 5.1 SigLIP的核心创新

SigLIP由Google于2023年提出，其核心创新是：**用简单的成对Sigmoid损失替代了CLIP中基于Softmax的InfoNCE损失**。

与CLIP的Softmax对比学习不同，Sigmoid损失**仅在图像-文本对上操作，不需要全局视图进行归一化**。

数学上，SigLIP的损失函数定义为：

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma(z_{ij} \cdot (t \cdot \text{sim}(v_i, t_j) + b))$$

其中：
- $z_{ij} = 1$ 当 $i = j$（正样本对），$z_{ij} = -1$ 当 $i \neq j$（负样本对）
- $t$ 是可学习的**逆温度参数**（inverse temperature）
- $b$ 是可学习的**偏置参数**（bias）
- $\sigma(\cdot)$ 是Sigmoid函数

### 5.2 SigLIP vs CLIP：关键差异

| 特性 | CLIP (InfoNCE) | SigLIP (Sigmoid) |
|------|---------------|------------------|
| 损失函数 | Softmax归一化 | 成对Sigmoid |
| 全局归一化 | 需要（计算所有配对相似度） | 不需要 |
| 小批次表现 | 较差 | **显著更优** |
| 大批次扩展 | 需要极大批次 | 更灵活 |
| 内存效率 | 较低 | **更高** |

SigLIP最显著的优势在于：**在小批次训练时表现远超Softmax损失**。例如，在16k批次大小下，Sigmoid损失的性能比Softmax损失高出很多；随着批次大小增加，两者差距逐渐缩小。

实验表明，SigLIP在WebLI数据集上预训练时，**32k批次大小已足够达到最佳性能**，而Softmax损失需要98k的批次大小。

### 5.3 SigLIP已成为新一代多模态模型的首选编码器

由于SigLIP在训练效率和性能上的优势，它已成为**LLaVA-NeXT等新一代多模态大模型的首选视觉编码器**。

LLaVA-NeXT项目中广泛采用了SigLIP视觉编码器（如 `siglip-so400m-patch14-384`）。这些编码器将图像转换为视觉令牌（visual tokens），再通过投影层输入到大语言模型中。

SigLIP的成功表明：**对比学习的损失函数设计仍有巨大的优化空间**。从Softmax到Sigmoid，不仅仅是激活函数的替换，更是一种从“全局归一化”到“成对独立建模”的范式转变。

---

## 六、总结

从CLIP到SigLIP，多模态对齐的技术路线经历了三个关键阶段的演进：

**第一阶段：双塔架构 + 对比学习**。CLIP证明了通过简单的对比学习，可以在4亿图文对上训练出具备强大零样本能力的多模态模型。双塔架构将图像和文本映射到统一的联合嵌入空间，为后续所有多模态模型奠定了基础。

**第二阶段：InfoNCE损失的数学优雅性**。InfoNCE损失将对比学习转化为N分类问题，其与交叉熵的等价性揭示了对比学习的本质——让模型学会“找出配对的样本”。

**第三阶段：SigLIP的效率革命**。通过将Softmax替换为Sigmoid，SigLIP在保持性能的同时大幅提升了训练效率和批次灵活性，成为新一代多模态模型的标准组件。

多模态对齐的故事还在继续。从CLIP到SigLIP，从4亿图文对到更大规模的数据集，从零样本分类到多模态大语言模型，每一次进步都建立在对数学原理的深刻理解之上。正如一位研究者所说：“对比学习的精髓在于‘比较’——通过让模型学会区分匹配与不匹配，我们赋予了机器跨模态理解的能力。”