---
title: Video Audio Multimodal —— 视频理解 & 视频生成 & 音频处理
published: 2026-08-16
description: 系统讲解视频理解与生成、音频处理的核心技术：从视频的时空维度出发，剖析3D卷积与时空注意力如何建模动态信息；深入解读VideoMAE的视频Tube掩码策略与90-95%超高掩码比例的时空自监督预训练；剖析Sora视频扩散模型的核心组件及其灵活处理任意分辨率/时长视频的能力；梳理音频处理基础与Whisper的编码器-解码器架构，并介绍VALL-E将语音合成建模为条件语言模型的新范式。
cover: "/assets/images/posts/video_audio_multimodal.png"
coverInContent: false
tags: [视频理解, 视频生成, 音频处理, 3D卷积, 多模态, VideoMAE, Sora, Whisper, VALL-E]
category: AI_Multimodal
draft: false
---

# Video Audio Multimodal —— 视频理解 & 视频生成 & 音频处理

## 引言：当AI开始“看”视频、“听”声音

在前几篇文章中，我们讨论了多模态模型如何理解图像和文本、如何用扩散模型生成图像、如何让机器人“动手”。现在，是时候把视野再拓宽一些了。

视频，是图像的**时间延伸**——一帧帧图像按时间排列，构成了一个包含丰富动态信息的数据形态。音频，是**时间的信号**——声波在时间轴上的变化承载了语言、情绪和环境信息。当AI同时掌握了“看视频”和“听声音”的能力，它才真正开始接近人类感知世界的方式。

本文将对视频理解与生成、音频处理这两个领域做一个**基础性的梳理**。我们将从视频的时空维度出发，介绍3D卷积和时空注意力如何建模动态信息；然后看看VideoMAE如何通过掩码自监督学习视频表征；接着走进Sora的世界，理解视频扩散模型的最基础概念；最后简要介绍Whisper的音频编码方案和VALL-E的语音合成方法。

---

## 一、视频理解的时空维度

### 1.1 视频：不只是图像的集合

一张图像是二维的——有高度和宽度。一段视频是**三维**的——除了高度和宽度，还有**时间**维度。

把视频简单地理解为“一堆图像的集合”是不够的。视频的核心价值在于**动态**：物体在移动、场景在变化、事件在展开。要理解视频，模型必须同时建模两个层面的信息：

- **空间信息**：每一帧图像中，哪些物体在什么位置
- **时间信息**：这些物体在不同帧之间如何运动、如何交互

这就是视频理解中常说的**时空建模**（Spatio-Temporal Modeling）。

### 1.2 3D卷积：在时间维度上“滑动”

处理图像时，我们使用2D卷积——卷积核在高度和宽度两个维度上滑动。处理视频时，一个自然的扩展是**3D卷积**——卷积核在**高度、宽度、时间**三个维度上滑动。

3D卷积核的维度可以表示为 $K_h \times K_w \times K_t$，其中 $K_h$ 和 $K_w$ 是空间尺寸，$K_t$ 是时间深度（即一次覆盖多少帧）。

对于输入视频张量 $X \in \mathbb{R}^{T \times H \times W \times C}$（$T$帧，每帧$H \times W$，$C$个通道），3D卷积的计算为：

$$Y_{t,h,w} = \sum_{i=0}^{K_t-1} \sum_{j=0}^{K_h-1} \sum_{k=0}^{K_w-1} W_{i,j,k} \cdot X_{t+i, h+j, w+k} + b$$

与2D卷积的区别在于：3D卷积的**输出特征图仍然保留了时间维度**，因此可以继续在时间维度上进行更深层的特征提取。

```python
import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    """
    一个简单的3D卷积层示例
    用于提取视频的时空特征
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super().__init__()
        # kernel_size: (时间深度, 高度, 宽度)
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=1
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        # B: batch size, C: 通道数, T: 帧数, H: 高度, W: 宽度
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 示例：输入16帧224x224的视频
model = Simple3DCNN(in_channels=3, out_channels=64)
video = torch.randn(4, 3, 16, 224, 224)  # [B, C, T, H, W]
output = model(video)
print(f"输出形状: {output.shape}")  # [4, 64, 16, 224, 224]
```

3D卷积的代表性工作包括**I3D**（Inflated 3D ConvNet）——将2D卷积核“膨胀”为3D，以及**C3D**等。这些模型在视频动作识别任务上取得了显著成功。

### 1.3 时空注意力：让Transformer“看”视频

随着Transformer在视觉领域的成功，研究者自然想到将其扩展到视频领域。**时空注意力**（Spatio-Temporal Attention）是其中的核心机制。

与图像ViT类似，视频Transformer将视频切分为**时空块**（Spatio-Temporal Patches）——每个块覆盖一个小的空间区域和几帧的时间窗口。这些块被展平后输入Transformer。

时空注意力的核心思想是：**让每个时空块能够“关注”到所有其他时空块**，无论是同一帧中不同位置的块（空间注意力），还是不同帧中同一位置的块（时间注意力）。

一个高效的实现方式是**分解式时空注意力**：
1. **空间注意力**：在同一帧内，不同空间位置之间计算注意力
2. **时间注意力**：在不同帧之间，同一空间位置计算注意力

这种分解方式将 $O((T \times N)^2)$ 的复杂度降低到 $O(T \times N^2 + N \times T^2)$，其中 $T$ 是帧数，$N$ 是每帧的块数。

**UniFormer**（Unified Transformer）是一个典型代表，它**无缝集成了3D卷积和时空自注意力的优势**，在计算效率和准确率之间取得了良好的平衡。

---

## 二、VideoMAE：视频自监督预训练

### 2.1 从MAE到VideoMAE

**MAE**（Masked Autoencoder）是图像自监督学习的一次突破——随机掩码掉图像的大部分patch，让模型从可见patch中重建被掩码的部分。

**VideoMAE**将这个思想扩展到视频领域。VideoMAE的核心洞察是：**视频具有更高的时间冗余性，因此可以采用比图像更高的掩码比例**。

对于图像，MAE的掩码比例约为75%。对于视频，VideoMAE采用了**90%-95%** 的极高掩码比例。视频内容在时间维度上存在大量重复信息（相邻帧之间变化很小），这使得即使掩码掉绝大多数内容，模型仍然有足够的线索进行重建。

### 2.2 Video Tube Masking：定制化的视频掩码策略

VideoMAE提出了一种名为 **Video Tube Masking** 的掩码策略。

与图像MAE随机掩码单个patch不同，Video Tube Masking在**时间维度上也进行掩码**——如果一个patch在某一帧被掩码，它在相邻帧的对应位置也被掩码，形成一个“时空管状”的掩码区域。

这种设计迫使模型**利用时间上下文来理解视频内容**——它不能简单地通过参考相邻帧来“作弊”重建被掩码的patch，而必须真正理解视频的时空结构。

```python
import torch
import numpy as np

def video_tube_masking(video_tokens, mask_ratio=0.9, tube_size=4):
    """
    视频Tube掩码的简化实现
    
    Args:
        video_tokens: [T, N, D] T帧，每帧N个patch
        mask_ratio: 掩码比例（VideoMAE使用90-95%）
        tube_size: 时间管道的长度（掩码连续几帧）
    """
    T, N, D = video_tokens.shape
    
    # 1. 随机选择要掩码的空间位置（在单帧上采样）
    num_mask = int(N * mask_ratio)
    mask_indices = np.random.choice(N, num_mask, replace=False)
    
    # 2. 在时间维度上扩展为tube
    # 对每个被选中的空间位置，掩码连续tube_size帧
    mask = torch.zeros(T, N, dtype=torch.bool)
    for idx in mask_indices:
        # 随机选择起始帧
        start_t = np.random.randint(0, T - tube_size + 1)
        mask[start_t:start_t + tube_size, idx] = True
    
    # 3. 应用掩码（被掩码的位置用可学习的掩码token替代）
    masked_tokens = video_tokens.clone()
    masked_tokens[mask] = 0  # 实际应用中用特殊的[MASK] token
    
    return masked_tokens, mask
```

### 2.3 VideoMAE的关键发现

VideoMAE的研究有三个重要发现：

1. **超高掩码比例有效**：90%-95%的掩码比例仍然能取得优秀的性能，这得益于视频的时间冗余性

2. **小数据集上同样有效**：VideoMAE在仅有**3000-4000个视频**的小数据集上也能取得令人印象深刻的结果——这在传统监督学习中几乎是不可能的

3. **数据质量重于数量**：预训练数据与目标数据集之间的**领域偏移**（Domain Shift）是一个重要因素

在性能上，VideoMAE使用标准的ViT骨干网络，在Kinetics-400上达到了**87.4%** 的准确率，在Something-Something V2上达到**75.4%**。

---

## 三、Sora与视频扩散模型

### 3.1 从图像扩散到视频扩散

我们在第四篇中详细讨论了扩散模型（DDPM）的原理：通过逐步向数据添加噪声、再学习逆向去噪的过程，从纯噪声中生成图像。

**视频扩散模型**将这一思想从2D扩展到3D——生成的对象从单张图像变成了**一帧帧连续的图像序列**。核心挑战在于：不仅要保证每一帧的图像质量，还要保证**帧与帧之间的时间一致性**——物体不能突然消失、运动要平滑自然。

### 3.2 Patchify与Spacetime Patches

Sora的核心技术之一是 **Patchify**——将视频分割为**时空块**（Spacetime Patches）。

具体来说：
1. 视频通过VAE编码器压缩到**低维潜在空间**
2. 在潜在空间中，视频被切割为**时空Patch**——每个Patch覆盖一个小的空间区域和几帧的时间窗口
3. 这些Patch被**展平为一维Token序列**，输入到Transformer中

这种设计的精妙之处在于：**Patchify让模型能够处理任意分辨率、任意时长、任意宽高比的视频**。在推理时，只需在指定大小的网格中随机初始化这些Patches，就可以控制生成视频的尺寸。

```python
class SpacetimePatchEmbed(nn.Module):
    """
    时空Patch嵌入（简化版）
    将视频切分为时空块并线性投影
    """
    def __init__(self, patch_size_t=4, patch_size_h=16, patch_size_w=16, 
                 in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size_t = patch_size_t
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w
        
        # 使用3D卷积实现时空Patch嵌入
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(patch_size_t, patch_size_h, patch_size_w),
            stride=(patch_size_t, patch_size_h, patch_size_w)
        )
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, embed_dim, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# 示例：输入8帧256x256的视频
patch_embed = SpacetimePatchEmbed(patch_size_t=2, patch_size_h=16, patch_size_w=16)
video = torch.randn(2, 3, 8, 256, 256)
tokens = patch_embed(video)
print(f"Token序列: {tokens.shape}")  # [2, (8/2)*(256/16)*(256/16)=1024, 768]
```

### 3.3 DiT：扩散Transformer

Sora的另一个核心技术是 **DiT（Diffusion Transformer）** ——用Transformer架构替代传统扩散模型中的U-Net作为去噪网络的主干。

传统扩散模型（如Stable Diffusion）使用U-Net进行噪声预测。DiT的创新在于：**将扩散模型的去噪过程建模为Transformer的序列预测任务**。

DiT的工作流程可以概括为：
1. 获取随机噪声视频，通过VAE压缩到潜在空间
2. 将潜在空间数据切分为时空Patches，转为Token序列
3. Token序列输入DiT（Transformer），结合文本提示进行去噪
4. 通过VAE解码器将潜在空间还原为视频

DiT的优势在于**可扩展性**——Transformer架构可以轻松扩展到数十亿参数，性能随模型规模和数据规模同步提升。

Sora就是建立在DiT之上的视频生成模型，它结合了**扩散模型**（生成能力）、**Transformer**（可扩展性）和**时空Patch**（灵活性）三者的优势。

---

## 四、音频处理基础

### 4.1 从声波到频谱图

音频数据是**一维时间序列**——声波在时间轴上的振幅变化。要让神经网络理解音频，首先需要将其转换为更合适的表示形式。

最常用的转换是**频谱图**（Spectrogram）：通过**短时傅里叶变换**（STFT），将一维的时间信号转换为二维的**时频表示**——横轴是时间，纵轴是频率，颜色表示能量强度。

**梅尔频谱图**（Mel Spectrogram）是频谱图的一种变体，它使用**梅尔滤波器组**模拟人耳对频率的非线性感知。人耳对低频信号更敏感、对高频信号相对迟钝，梅尔尺度正是对这种感知特性的数学建模。

Whisper采用**80维梅尔频谱图**作为音频特征表示。具体参数为：
- 80个梅尔滤波器组
- 帧长25ms，帧移10ms
- 频率覆盖0-8000Hz

```python
import torch
import torchaudio

def extract_mel_spectrogram(waveform, sample_rate=16000, n_mels=80):
    """
    从原始音频提取梅尔频谱图（Whisper风格）
    
    Args:
        waveform: [B, T] 原始音频波形
        sample_rate: 采样率（Whisper使用16kHz）
        n_mels: 梅尔滤波器数量（Whisper使用80）
    """
    # 1. 短时傅里叶变换
    spec = torchaudio.transforms.Spectrogram(
        n_fft=400,  # 25ms @ 16kHz
        hop_length=160,  # 10ms @ 16kHz
    )(waveform)  # [B, freq_bins, time_steps]
    
    # 2. 转换为梅尔尺度
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
    )(waveform)  # [B, n_mels, time_steps]
    
    # 3. 取对数（Log-Mel）
    log_mel_spec = torch.log(mel_spec + 1e-10)
    
    return log_mel_spec  # [B, 80, T]

# 示例
waveform = torch.randn(2, 16000 * 10)  # 2个样本，各10秒
mel = extract_mel_spectrogram(waveform)
print(f"梅尔频谱图形状: {mel.shape}")  # [2, 80, 约1000个时间步]
```

### 4.2 Whisper：音频编码的标杆

**Whisper**是OpenAI开源的通用语音识别模型，其架构设计清晰地展示了现代音频编码的标准方案。

Whisper采用**编码器-解码器**的Transformer架构：

**编码器**：
1. 输入：原始音频 $\rightarrow$ 80维梅尔频谱图
2. 通过**两层卷积**进行时频域特征压缩
3. 通过**Transformer编码器**的多头自注意力捕捉长时依赖

**解码器**：
1. 接收编码器的输出和文本前缀
2. 自回归地生成转录文本

Whisper的一个独特设计是：**在输入层保留了原始音频的静音段信息**，这使其能更好地处理口语中的停顿、呼吸声等非语言元素。

Whisper的训练数据规模达到**68万小时**的多语言音频，覆盖了多种语言、口音和录音环境，使其在零样本场景下也表现出色。

---

## 五、VALL-E：语音合成的新范式

### 5.1 传统TTS vs VALL-E

传统的文本到语音合成（TTS）系统通常采用**级联架构**：先通过声学模型将文本转换为梅尔频谱图，再通过声码器（Vocoder）将频谱图转换为音频波形。这种方法需要精心设计的中间表示，且对训练数据的质量和数量要求很高。

**VALL-E**提出了一种全新的思路：**将语音合成建模为条件语言模型任务**。

### 5.2 核心思想：语音即编码

VALL-E的核心思想可以概括为：

1. **音频编码**：使用神经音频编解码器（如EnCodec）将原始音频压缩为**离散的编码序列**（Codec Codes）
2. **语言建模**：训练一个Transformer语言模型，**以文本为条件，预测音频编码序列**
3. **零样本语音克隆**：仅需**3秒钟的目标说话人录音**作为声学提示（Acoustic Prompt），即可合成该说话人的任意文本语音

VALL-E的训练数据达到了**6万小时**的英语语音，是当时现有TTS系统的数百倍。这种规模的数据让VALL-E展现出了**上下文学习**能力——它能从提示中提取说话人的音色、情绪和声学环境。

### 5.3 VALL-E 2：迈向人类水平

VALL-E 2在VALL-E的基础上引入了两项关键改进：

1. **重复感知采样**（Repetition Aware Sampling）：改进原始的nucleus采样过程，考虑解码历史中的token重复情况，不仅稳定了解码过程，还避免了无限循环问题

2. **分组编码建模**（Grouped Code Modeling）：将编解码编码组织成组，有效缩短序列长度，提升推理速度并解决长序列建模的挑战

VALL-E 2首次在零样本TTS任务上达到了**人类水平**（Human Parity）。

---

## 六、总结：视听技术的交汇

本文对视频理解、视频生成和音频处理三个领域做了基础性的梳理。它们看似独立，实则正在走向深度融合：

| 领域 | 核心任务 | 关键架构 | 代表工作 |
|------|---------|---------|----------|
| 视频理解 | 理解视频内容 | 3D卷积、时空注意力 | VideoMAE, UniFormer |
| 视频生成 | 从文本/噪声生成视频 | DiT、时空Patch | Sora |
| 音频编码 | 将音频转为特征 | 梅尔频谱图、Transformer | Whisper |
| 语音合成 | 从文本生成语音 | 编解码语言模型 | VALL-E, VALL-E 2 |

这些技术正在汇聚成更强大的多模态系统——能够**看懂视频、听懂声音、生成视听内容**的通用模型。正如VideoMAE展示了视频自监督学习的潜力，Sora展示了视频生成的革命性突破，VALL-E展示了语音合成的新范式，它们共同描绘了多模态AI的未来图景：一个能够像人类一样同时处理视觉、听觉和语言的智能系统。