---
title: Vision Transformer —— ViT视觉Transformer
published: 2026-08-04
description: 系统讲解视觉Transformer（ViT）的核心机制与技术演进：从Patch Embedding与位置嵌入的基础原理出发，对比ViT与CNN在归纳偏置上的根本差异及DeiT通过知识蒸馏弥补数据效率的尝试；深入剖析Swin Transformer的窗口注意力与层级化特征表示、可变形注意力的动态聚焦机制；系统梳理高分辨率适配的三条技术路线——AnyRes/LLaVA-UHD的切图融合策略、NaViT/ViTAR的原生动态分辨率方案、以及RoPE/插值/LookHere等位置编码外推方法。
cover: "/assets/images/posts/vit.png"
coverInContent: false
tags: [Transformer, ViT, 多模态, 归纳偏置, Swin Transformer, 位置编码外推]
category: AI_Multimodal
draft: false
---

# Vision Transformer —— ViT视觉Transformer

## 引言：当Transformer“看见”图像

2020年，Google的一篇论文《An Image is Worth 16x16 Words》彻底改变了计算机视觉的格局。这篇论文提出了**Vision Transformer（ViT）** ，首次证明了一个纯Transformer架构可以在不依赖卷积层的情况下，在图像分类任务上达到甚至超越CNN的性能。

这个结论在当时是反直觉的。CNN之所以在视觉领域统治了十余年，靠的是其精心设计的**归纳偏置**（Inductive Bias）——局部感受野、权值共享、平移不变性。这些先验知识让CNN能够在有限数据上高效学习。而ViT抛弃了所有这些设计，它把图像切成一块块“视觉单词”，直接扔给Transformer去处理。

那么，ViT是如何工作的？它凭什么能打败CNN？当面对高分辨率图像时，它又遇到了什么新的挑战？本文将带你从零开始，一步步拆解视觉Transformer的完整技术图谱。

---

## 一、ViT基础机制：把图像切成“单词”

### 1.1 Patch Embedding：图像分块与线性投影

ViT处理图像的第一步，是将二维图像转化为一维的token序列。这个过程分为两个关键步骤：

**第一步：图像分块（Patch Partition）**

将输入图像（如 $224 \times 224 \times 3$）划分为固定大小的非重叠图像块（patch）。最常见的设置是 $16 \times 16$ 像素，这样一幅 $224 \times 224$ 的图像会被切成：

$$N = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196 \text{ 个图像块}$$

每个图像块展平为一个向量，维度为 $16 \times 16 \times 3 = 768$。

**第二步：线性投影（Linear Projection）**

通过一个可训练的线性层（即全连接层），将每个展平的图像块映射到 $D$ 维的嵌入空间：

$$\mathbf{x}_p^i = \text{Linear}(\text{Flatten}(\text{Patch}_i)) \in \mathbb{R}^D$$

$D$ 是Transformer的隐藏维度（ViT-Base中 $D=768$，ViT-Large中 $D=1024$）。

这一步等价于使用一个卷积核大小为 $16 \times 16$、步长为 $16$ 的卷积操作：

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 14*14=196
        
        # 使用卷积实现patch embedding：kernel_size=patch_size, stride=patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.proj(x)           # [B, 768, 14, 14]
        x = x.flatten(2)           # [B, 768, 196]
        x = x.transpose(1, 2)      # [B, 196, 768]
        return x
```

### 1.2 位置嵌入（Position Embedding）

Transformer的自注意力机制是**置换不变**（Permutation Invariant）的——它对输入序列的顺序不敏感。但对于图像来说，空间位置信息至关重要：左上角的图像块和右下角的图像块，即使内容相同，语义也完全不同。

因此，ViT需要为每个token添加位置信息。ViT采用**可学习的1D位置编码**（Learnable Positional Encoding）：

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1; \mathbf{x}_p^2; ...; \mathbf{x}_p^N] + \mathbf{E}_{\text{pos}}$$

其中 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 是一个可学习的参数矩阵，与token嵌入相加后输入Transformer。

ViT还会在序列开头添加一个特殊的 **[CLS] token**（分类令牌），其最终输出状态作为图像的全局表示用于分类。

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # [CLS] token：可学习的分类令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 位置编码：可学习的1D位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer编码器（12层）
        self.encoder = TransformerEncoder(embed_dim, num_layers=12)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]
        
        # 拼接[CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 经过Transformer编码器
        x = self.encoder(x)
        
        # 取[CLS] token的输出进行分类
        return self.head(x[:, 0])
```

---

## 二、归纳偏置：ViT vs CNN

### 2.1 什么是归纳偏置？

**归纳偏置**（Inductive Bias）是模型在学习之前，其结构本身已经带有的一些先验假设。这些先验帮助模型在有限数据上更高效地学习。

CNN的归纳偏置主要体现在两个方面：

1. **局部性（Locality）** ：图像中相邻像素的相关性更强，CNN通过卷积核只关注局部区域
2. **平移不变性（Translation Invariance）** ：一个特征无论在图像哪个位置出现，CNN都能用同样的卷积核检测到

这些先验让CNN在ImageNet等数据集上表现出色，但也限制了模型的灵活性——CNN很难学习到跨越远距离像素的全局依赖关系。

### 2.2 ViT：没有先验，用数据“喂”出能力

ViT完全抛弃了CNN的归纳偏置。它的自注意力机制从一开始就能让任意两个图像块直接交互，捕捉全局依赖。但这种“自由”是有代价的：

**当训练数据不足时，ViT的表现不如CNN**。因为没有结构先验，模型不知道该优先关注什么，只能从数据中“硬学”。

**但当数据量足够大时，ViT的表现会超过CNN**。ViT原论文的核心结论是：**当拥有足够多的数据进行预训练时，ViT的表现就会超过CNN，突破Transformer缺少归纳偏置的限制**。

这个“足够多”的门槛大约是 **1亿张以上** 的图像。ViT在JFT-300M（3亿张图像）上预训练后，在ImageNet上达到了88.55%的准确率，超过了当时的SOTA CNN。

### 2.3 弥补归纳偏置的尝试：DeiT与知识蒸馏

ViT对大规模数据的依赖催生了一系列改进工作。其中最具代表性的是 **DeiT**（Data-efficient Image Transformers），它通过**知识蒸馏**（Knowledge Distillation）从CNN教师模型中学习归纳偏置，使得ViT能够在 **仅使用ImageNet-1k**（128万张）数据的情况下达到与CNN相当的性能。

DeiT的核心思想是：让ViT学生模型同时学习真实标签和CNN教师模型的软标签（soft labels），从而“继承”CNN的归纳偏置。

---

## 三、ViT变体：从Swin到可变形注意力

原始的ViT在处理高分辨率图像时存在两个核心问题：

1. **计算复杂度**：全局自注意力的复杂度为 $O(N^2)$，$N$ 是token数量。对于高分辨率图像，$N$ 急剧增加，导致计算量爆炸
2. **多尺度特征缺失**：视觉任务（如目标检测、分割）需要同时捕捉局部细节和全局上下文，而ViT的单一尺度特征表示难以满足

### 3.1 Swin Transformer：窗口注意力 + 层级结构

**Swin Transformer**（Shifted Window Transformer）由微软亚洲研究院提出，通过两项核心创新解决了上述问题。

**窗口多头自注意力（W-MSA）**：Swin将特征图划分为不重叠的局部窗口（如 $7 \times 7$），每个窗口内独立计算自注意力。

计算复杂度的对比：

| 机制 | 计算复杂度 | 显存占用（224×224） |
|------|-----------|-------------------|
| 全局注意力 | $O(H^2W^2 C)$ | 极高（>24GB） |
| 窗口注意力 | $O(HW M^2 C)$，$M=7$ | 可控（<8GB） |

对于 $224 \times 224$ 的图像，全局注意力需计算 $50,176$ 个token之间的关系（$224 \times 224$），而窗口化后仅需计算每个 $7 \times 7$ 窗口内 $49$ 个token的关系。

```python
def window_partition(x, window_size):
    """
    将特征图划分为窗口
    x: [B, H, W, C]
    window_size: M
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, 
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)
```

**移位窗口（Shifted Window）**：纯窗口化会导致**窗口间信息孤立**——不同窗口的token无法交互。Swin通过**移位窗口**解决这个问题：在相邻层中，窗口位置偏移 $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ 像素。

具体实现为两层交替：
- **偶数层**：使用常规窗口划分
- **奇数层**：窗口向右下移动 $\lfloor M/2 \rfloor$ 像素

这样，原本属于不同窗口的token会进入同一窗口，实现跨窗口信息传递。

**层级化特征表示**：Swin还借鉴了CNN的层级设计，通过 **Patch Merging** 逐步下采样：

- **Stage 1**：$4 \times 4$ patch划分，输出 $56 \times 56 \times 96$ 的特征图
- **Stage 2**：$2 \times 2$ 窗口合并（类似stride=2卷积），通道数翻倍
- **Stage 3-4**：重复窗口合并，最终分辨率降低32倍

这种设计构建了四级特征金字塔，既能捕捉局部细节，又能提取全局语义。

### 3.2 可变形注意力（Deformable Attention）

**可变形注意力**（Deformable Attention）是另一种处理非规则输入的重要创新。

标准自注意力中，每个query需要关注所有key，计算量大且会引入无关区域的噪声。可变形注意力的核心思想是：**以数据相关的方式选择自注意力中key的位置**，使得注意力模块能够聚焦于相关区域。

具体来说，对于每个query，可变形注意力只关注**参考点附近的一小部分固定数量的key**。这些参考点的位置是通过**偏移学习**（Offset Learning）动态确定的——模型学习预测每个采样点应该偏移多少。

**Deformable Attention Transformer（DAT）** 将可变形注意力模块与金字塔架构结合，构建了一个强大的视觉主干网络。DAT的优势在于：
- **数据依赖**：注意力模式根据输入内容动态调整
- **计算高效**：只关注关键区域，避免全局计算
- **适应性强**：能处理非正方形输入和任意形状的目标

---

## 四、高分辨率适配：从AnyRes到动态ViT

ViT的输入分辨率通常是固定的（如 $224 \times 224$ 或 $336 \times 336$）。当遇到高分辨率图像（如4K照片、文档扫描）时，简单的**Resize**会导致严重的信息丢失——报表上的小字会变成马赛克。

如何让ViT高效处理高分辨率图像？以下是三条主流技术路线。

### 4.1 AnyRes / LLaVA-UHD：切图 + 独立编码 + 融合

**AnyRes**（Any Resolution）技术由LLaVA-NeXT引入，核心思想模仿了人类的阅读习惯：**先看全貌，再看细节**。

**AnyRes的工作流程**：
1. **选择最佳分辨率**：从预定义的网格候选点（grid pinpoints）中选择最适配原图宽高比的分辨率
2. **Resize与Padding**：将图像调整到该分辨率
3. **切分为子图**：将图像切分为多个crop（子图），每个子图的大小接近ViT的预训练尺寸
4. **独立编码**：每个子图 + 原图的缩略图，分别通过ViT编码
5. **拼接融合**：将所有子图的token按空间位置拼接，送入LLM

```python
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    计算AnyRes的图像网格形状
    image_size: (width, height)
    grid_pinpoints: 预定义的网格候选点列表，如 [(1,1), (1,2), (2,2), ...]
    """
    best_resolution = None
    best_error = float('inf')
    
    for n_rows, n_cols in grid_pinpoints:
        # 计算该网格对应的分辨率
        target_width = n_cols * patch_size
        target_height = n_rows * patch_size
        # 计算宽高比误差
        error = abs(target_width / target_height - image_size[0] / image_size[1])
        if error < best_error:
            best_error = error
            best_resolution = (target_width, target_height)
    
    return best_resolution
```

**LLaVA-UHD**进一步优化了这个思路，包含三个关键组件：

1. **图像模块化策略**：将原始分辨率图像切分为更小的可变大小切片，实现高效编码
2. **压缩模块**：进一步压缩视觉编码器输出的图像token，减少计算量
3. **空间模式**：组织切片token的空间位置，供LLM理解

LLaVA-UHD的实验结果令人印象深刻：
- 在LLaVA-1.5（$336 \times 336$）基础上，支持 **6倍大** 的分辨率（$672 \times 1088$）
- 推理计算量仅增加 **94%**
- 在TextVQA上准确率提升 **6.4%**
- 在8张A100上训练仅需 **23小时**（与LLaVA-1.5的26小时相当）

### 4.2 原生动态分辨率：NaViT与ViTAR

AnyRes虽然有效，但仍然依赖于预定义的网格和resize操作。更根本的解决方案是让ViT**原生支持任意分辨率**。

**NaViT**（Native Resolution ViT）在训练过程中使用**序列打包**（Sequence Packing），将不同分辨率的图像打包到同一个批次中训练。这使得模型能够处理任意分辨率和宽高比的输入。

**ViTAR**（Vision Transformer with Any Resolution）则通过**模糊位置编码**（Fuzzy Positional Encoding）解决了分辨率泛化问题。

ViTAR的核心创新是：**为任意尺寸的patch生成连续的位置表示**，而不是依赖固定尺寸的位置编码表。这样，模型在训练时不会过拟合到特定的分辨率。

ViTAR的性能表现：
- 在 $1120 \times 1120$ 分辨率下达到 **83.3%** 的Top-1准确率
- 在 $4032 \times 4032$ 的超高分辨率下仍保持 **80.4%** 的准确率
- 同时显著降低了计算成本

### 4.3 位置编码外推：RoPE与插值策略

位置编码是限制ViT分辨率泛化的关键瓶颈。当推理时的分辨率高于训练时，位置编码会面临“没见过”的位置索引。

**2D-RoPE：旋转位置编码的外推能力**

**RoPE**（Rotary Position Embedding）最初为语言模型设计，在**长度外推**（Length Extrapolation）方面表现出色。研究者将RoPE扩展到2D视觉数据，发现 **2D-RoPE** 在ViT上同样展现了卓越的外推性能：

- 在 $224 \times 224$ 上训练，在 $1024 \times 1024$ 上测试时，2D-RoPE仍能保持较好的性能
- Swin Transformer结合RoPE后，在多分辨率推理中表现优异

RoPE的核心机制是：**在每个自注意力层中，通过旋转查询和键向量来注入位置信息**。与绝对位置编码不同，RoPE编码的是**相对位置**，因此对序列长度的变化更鲁棒。

**插值策略**

另一种常用的外推方法是**位置编码插值**（Positional Encoding Interpolation）：
- 当推理分辨率提高时，将原有的位置编码通过**双线性插值**扩展到新的尺寸
- 这种方法在Swin Transformer的微调中被广泛使用：在 $224 \times 224$ 上预训练，微调时通过插值适配到 $640 \times 640$

**ResFormer：多分辨率训练**

**ResFormer**采用了另一种思路：在训练过程中，对同一张图像的不同分辨率副本进行操作，并施加**尺度一致性损失**来促进跨尺度的信息交互。

更重要的是，ResFormer提出了**全局-局部位置嵌入**策略，能够根据输入尺寸平滑调整位置编码。实验结果表明，ResFormer在从 $96$ 到 $640$ 的广泛分辨率范围内都表现良好。

**LookHere：超越RoPE**

**LookHere**是2024年提出的最新位置编码方法。它的核心思想是：**限制每个注意力头只关注固定方向的视野**，使用2D注意力掩码让不同的头“看向”不同的方向。

LookHere的优势：
- 提供平移等变性（Translation Equivariance）
- 确保注意力头的多样性
- 限制外推时注意力头面临的分布偏移

在外推场景下（$224 \times 224$ 训练，$1024 \times 1024$ 测试），LookHere在ImageNet上的表现**超越了2D-RoPE达21.7%**。

---

## 五、总结

视觉Transformer从2020年诞生至今，经历了快速的演进：

| 阶段 | 代表模型 | 核心创新 | 分辨率处理 |
|------|---------|---------|-----------|
| 奠基 | ViT | Patch Embedding + Transformer | 固定分辨率 |
| 效率优化 | Swin Transformer | 窗口注意力 + 层级结构 | 固定分辨率，计算高效 |
| 数据效率 | DeiT | 知识蒸馏 | 固定分辨率，减少数据需求 |
| 高分辨率 | AnyRes/LLaVA-UHD | 切图 + 独立编码 + 融合 | 动态网格切分 |
| 原生动态 | NaViT/ViTAR | 序列打包/模糊位置编码 | 任意原生分辨率 |
| 位置外推 | LookHere | 定向注意力掩码 | 训练低、推理高 |

这条演进路线揭示了一个清晰的趋势：**视觉Transformer正在从“固定分辨率的囚徒”走向“任意分辨率的自由”** 。

这种自由的意义不仅仅是技术上的突破。在实际应用中，高分辨率意味着更多的细节——文档中的小字、遥感图像中的细微目标、医疗影像中的病理特征。让ViT能够高效处理任意分辨率的图像，正在为OCR、自动驾驶、医学影像分析等领域带来革命性的变化。

正如ViT原论文的标题所说：“**An Image is Worth 16x16 Words**”。而今天，我们已经知道——一张图像的价值，远不止16×16个单词能承载。