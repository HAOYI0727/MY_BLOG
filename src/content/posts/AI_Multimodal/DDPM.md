---
title: Denoising Diffusion Probabilistic Models —— DDPM扩散模型
published: 2026-08-10
description: 系统讲解扩散模型（DDPM）的数学原理与条件生成技术：从马尔可夫链的前向扩散与反向去噪出发，推导ELBO如何简化为噪声预测MSE损失；深入剖析潜空间扩散如何通过VAE压缩实现约50倍计算量降低；追溯去噪网络从U-Net到DiT的架构演进及其在Sora中的应用；解析CFG的线性外推公式及引导强度的权衡.
cover: "/assets/images/posts/ddpm.png"
coverInContent: false
tags: [扩散模型, DDPM, Diffusion, 多模态, 潜空间扩散, DiT]
category: AI_Multimodal
draft: false
---

# Denoising Diffusion Probabilistic Models —— DDPM扩散模型

## 引言：从噪声中“生长”出图像

想象一下，你有一张清晰的照片。现在，你开始往上面撒盐——一点点、一点点，直到照片完全被噪声淹没，变成一团毫无意义的雪花点。这个过程叫做**前向扩散**。

如果现在有人给你这团雪花点，要求你“逆向操作”，把原来的照片还原出来——这就是**反向去噪**。扩散模型（Diffusion Model）的核心思想，就是**学会这个逆向过程**：从一个纯噪声开始，一步步“去噪”，最终生成一张高质量的图像。

这个想法最初由Sohl-Dickstein等人在2015年提出，但直到2020年**DDPM**（Denoising Diffusion Probabilistic Models）的发布，扩散模型才真正展现出超越GAN的生成能力。如今，从Stable Diffusion到DALL-E，从Midjourney到Sora，几乎所有顶尖的图像和视频生成模型背后，都跳动着扩散模型的“心脏”。

本文将带你从零开始，深入扩散模型的数学原理、架构演进和工程实践。

---

## 一、DDPM的数学原理：从马尔可夫链到噪声预测

DDPM由两个互补的过程构成：

1. **前向扩散过程（固定）** ：逐步向数据添加高斯噪声，将图像转化为纯噪声
2. **反向去噪过程（学习）** ：训练神经网络学习逆向去噪，从噪声中生成数据

这两个过程构成了一个**马尔可夫链**——每一步的状态只依赖于前一步。

### 1.1 前向扩散过程：从图像到噪声

前向过程通过 $T$ 步逐渐向数据 $x_0$ 添加高斯噪声，最终得到纯噪声 $x_T$。每一步定义为：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$

其中 $\beta_t$ 是**噪声调度参数**（noise schedule），通常是一个从 $10^{-4}$ 到 $0.02$ 的递增序列。

通过**重参数化技巧**（reparameterization trick），我们可以直接从 $x_0$ 计算任意时间步 $t$ 的 $x_t$：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。

这个公式的重要性在于：**训练时我们不需要逐步迭代加噪，可以直接从原始图像跳到任意时间步的噪声版本**。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def forward_diffusion_sample(x0, t, alpha_bar, noise=None):
    """
    前向扩散：从x0直接计算xt
    
    Args:
        x0: 原始图像 [B, C, H, W]
        t: 时间步 [B]
        alpha_bar: 累积噪声调度 [T]
        noise: 可选，若不提供则随机采样
    
    Returns:
        xt: 加噪后的图像
        noise: 实际添加的噪声（用于训练）
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    # 获取每个样本对应时间步的 alpha_bar
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
    
    # 重参数化采样
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return xt, noise
```

### 1.2 反向去噪过程：从噪声到图像

反向过程的目标是学习从噪声 $x_T$ 逐步恢复数据 $x_0$：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right)$$

DDPM的关键简化是：**固定方差 $\Sigma_\theta$ 为常数，只学习均值 $\mu_\theta$** 。

通过进一步的数学推导，均值可以被重参数化为**噪声预测**的形式：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

这意味着：**模型不需要直接预测图像，只需要预测被添加的噪声 $\epsilon$** 。

### 1.3 变分下界（ELBO）与训练目标的简化

DDPM的训练目标本质上是**最大化数据的对数似然** $\log p_\theta(x_0)$。然而直接优化极其困难，因此退而求其次，优化其**变分下界**（ELBO, Evidence Lower Bound）。

ELBO被拆解为 $T$ 个时间步的累加，每一项都是两个高斯分布之间的KL散度：

$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_q \left[ \log \frac{q(x_T \mid x_0)}{p_\theta(x_T)} + \sum_{t=2}^T D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)) - \log p_\theta(x_0 \mid x_1) \right]$$

经过复杂的变分推断和重参数化，这个看似复杂的损失函数最终被简化为一个极其简洁的形式：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**这就是DDPM训练的全部**——让神经网络预测被添加的噪声，最小化预测噪声与真实噪声的均方误差（MSE）。

```python
def ddpm_training_step(model, x0, t, alpha_bar, optimizer):
    """
    DDPM单步训练
    
    Args:
        model: 噪声预测网络（U-Net或DiT）
        x0: 原始图像 [B, C, H, W]
        t: 随机采样的时间步 [B]
        alpha_bar: 累积噪声调度
        optimizer: 优化器
    """
    # 1. 采样噪声
    noise = torch.randn_like(x0)
    
    # 2. 前向扩散：生成xt
    xt, _ = forward_diffusion_sample(x0, t, alpha_bar, noise)
    
    # 3. 模型预测噪声
    predicted_noise = model(xt, t)
    
    # 4. 计算MSE损失
    loss = F.mse_loss(predicted_noise, noise)
    
    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 1.4 DDPM采样算法

训练完成后，采样（生成）过程从纯噪声 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，逐步去噪：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

其中 $z \sim \mathcal{N}(0, \mathbf{I})$ 是随机噪声，$\sigma_t$ 控制随机性。

```python
@torch.no_grad()
def ddpm_sample(model, shape, alpha, alpha_bar, betas, T, device):
    """
    DDPM采样：从噪声生成图像
    
    Args:
        model: 训练好的噪声预测网络
        shape: 生成图像的形状 [B, C, H, W]
        T: 总步数
    """
    # 从纯噪声开始
    x = torch.randn(shape, device=device)
    
    # 逐步去噪
    for t in reversed(range(1, T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # 预测噪声
        predicted_noise = model(x, t_tensor)
        
        # 计算去噪后的x_{t-1}
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = betas[t]
        
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        
        # 添加随机噪声（最后一步除外）
        if t > 1:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise
    
    return x
```

---

## 二、潜空间扩散（Latent Diffusion）：从像素到隐空间

DDPM虽然在理论上优雅，但在实践中有一个致命的问题：**在像素空间操作太慢了**。一张 $512 \times 512$ 的图像有约78万个像素，每一步去噪都要处理这么多数据，而DDPM通常需要1000步——计算量可想而知。

**潜空间扩散模型（LDM, Latent Diffusion Model）** 完美解决了这个问题。

### 2.1 核心思想：在压缩空间中扩散

LDM的核心思想非常简单：**不直接在像素空间扩散，而是先通过一个自动编码器（VAE）将图像压缩到低维的“潜空间”（Latent Space），在这个压缩空间中进行扩散，最后再解码回像素空间**。

工作流程如下：

1. **编码阶段**：图像 $\rightarrow$ VAE编码器 $\rightarrow$ 潜空间张量（如 $64 \times 64 \times 4$）
2. **扩散阶段**：在潜空间中添加噪声并逐步去噪
3. **解码阶段**：潜空间张量 $\rightarrow$ VAE解码器 $\rightarrow$ 重建图像

### 2.2 效率提升有多大？

潜空间的维度远小于像素空间。以一张 $512 \times 512$ 的图像为例：

| 空间 | 维度 | 相对大小 |
|------|------|----------|
| 像素空间 | $512 \times 512 \times 3 \approx 78.6万$ | 1× |
| 潜空间 | $64 \times 64 \times 4 \approx 1.6万$ | **约1/49** |

这意味着：**计算量减少约40-70%，训练速度提升8-10倍**。

这就是为什么 **Stable Diffusion** 等主流文生图模型都基于LDM架构——在保持生成质量的同时，大幅降低了计算门槛。

```python
class LatentDiffusionModel(nn.Module):
    """
    简化的潜空间扩散模型
    """
    def __init__(self, vae_encoder, vae_decoder, diffusion_model):
        super().__init__()
        self.encoder = vae_encoder      # 图像 → 潜空间
        self.decoder = vae_decoder      # 潜空间 → 图像
        self.diffusion = diffusion_model # 在潜空间操作的扩散模型
    
    def encode(self, x):
        """将图像编码到潜空间"""
        with torch.no_grad():
            z = self.encoder(x)  # [B, 4, 64, 64]
        return z
    
    def decode(self, z):
        """将潜空间解码为图像"""
        with torch.no_grad():
            x = self.decoder(z)  # [B, 3, 512, 512]
        return x
    
    def forward(self, x, t):
        """在潜空间中进行扩散训练"""
        z = self.encode(x)
        # 在潜空间添加噪声并预测
        return self.diffusion(z, t)
```

### 2.3 像素空间扩散 vs 潜空间扩散

两种方案各有优劣：

| 维度 | 像素空间扩散 | 潜空间扩散（LDM） |
|------|------------|-------------------|
| 操作空间 | 原始像素 | 压缩隐空间 |
| 计算复杂度 | $O(N^2)$（N为像素数） | $O(n^2)$（n为隐空间维度，n << N）|
| 信息完整性 | 保留全部信息 | 存在VAE压缩损失 |
| 典型代表 | PixelDiffusion | Stable Diffusion |

潜空间扩散虽然在细节保留上略逊于像素空间方案（例如文本渲染时可能出现字体边缘模糊），但其**效率和可扩展性**的优势使其成为当前的主流选择。

---

## 三、去噪网络的演进：从U-Net到DiT

DDPM中，噪声预测网络 $\epsilon_\theta$ 的核心架构经历了从**U-Net**到**Diffusion Transformer（DiT）** 的重大演进。

### 3.1 U-Net时代：卷积的归纳偏置

DDPM原论文采用**U-Net**作为噪声预测网络。U-Net是一种对称的编码器-解码器架构，通过**跳跃连接**（skip connections）将编码器的特征直接传递给解码器，保留了空间细节信息。

U-Net在图像生成任务中表现出色，得益于CNN的**归纳偏置**——局部感受野和权值共享让它在有限数据下也能高效学习。

### 3.2 DiT：Transformer接管扩散模型

2023年，**DiT**（Diffusion Transformer）的出现改变了这一格局。

DiT的核心创新是：**用可扩展的Transformer架构替换U-Net主干，将视觉数据视为一系列patch（图像块）** 。

具体来说，DiT的工作方式与ViT类似：
1. 将输入（潜空间特征）切分为patch
2. 通过线性投影将patch转换为token序列
3. 在token序列上应用Transformer的自注意力机制
4. 通过**自适应层归一化**（AdaLN）注入时间步和条件信息

DiT的优势在于：
- **全局感受野**：自注意力机制让每个token都能直接关注所有其他token
- **可扩展性**：参数规模可以从百万级扩展到数十亿级，性能随规模线性提升
- **统一架构**：与LLM共享Transformer基础，便于多模态融合

### 3.3 从DiT到Sora：视频生成的革命

2024年2月，OpenAI发布**Sora**，提出了基于DiT的视频生成架构，彻底改写了视频模型的技术路线。

Sora在DiT基础上引入了**时空块**（Spacetime Patches），将视频数据解耦为空间-时间联合特征。这种设计使得模型能够：
- 统一处理全画面的时空信息
- 稳定输出长时序、高一致性的视频
- 算力投入与模型性能呈线性正向提升

如今，DiT已成为视频生成领域的**事实标准架构**。

```python
class DiTBlock(nn.Module):
    """
    Diffusion Transformer基本模块
    包含自注意力 + 自适应层归一化（注入时间步条件）
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # 自适应缩放和偏移（用于注入时间步信息）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)  # 6 = 2 (scale) + 2 (shift) + 2 (gate)
        )
    
    def forward(self, x, t_embed):
        # t_embed: 时间步的嵌入
        # 计算自适应参数
        mod = self.adaLN_modulation(t_embed)  # [B, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod.chunk(6, dim=-1)
        
        # 自适应层归一化 + 自注意力
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm)[0]
        
        # 自适应层归一化 + FFN
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm)
        
        return x
```

---

## 四、Classifier-Free Guidance：文本条件控制

扩散模型可以无条件生成（从随机噪声生成随机图像），但实际应用中我们更想要**条件生成**——给定一段文字描述，生成对应的图像。

**Classifier-Free Guidance（CFG，无分类器引导）** 是目前最主流的条件控制方法。

### 4.1 为什么需要CFG？

早期的条件扩散模型使用**分类器引导**（Classifier Guidance）：训练一个独立的分类器来指导生成过程朝向特定类别。但这种方法有两个问题：
1. 需要额外训练一个分类器
2. 分类器可能被对抗样本攻击

CFG的解决方案是：**在训练时，以一定概率（如10%）随机丢弃条件信息，让模型同时学会有条件和无条件生成**。

### 4.2 CFG的数学原理

在推理时，CFG通过**线性外推**组合有条件和无条件预测：

$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))$$

其中：
- $\epsilon_\theta(x_t, t, c)$ 是**有条件**的噪声预测（给定文本条件 $c$）
- $\epsilon_\theta(x_t, t, \varnothing)$ 是**无条件**的噪声预测（条件被丢弃）
- $w$ 是**引导强度**（guidance scale），通常取值在 $[1, 15]$ 之间

当 $w=1$ 时，就是标准的有条件生成；当 $w>1$ 时，模型会更加“用力”地遵循文本条件。

### 4.3 CFG的效果与权衡

CFG已成为现代文生图系统的**事实标准**，几乎所有大规模扩散模型都在使用。

然而，CFG也存在一个**权衡**（trade-off）：
- **较大的 $w$** ：生成图像与文本更匹配，但可能牺牲图像质量和多样性（颜色过饱和、 artifacts）
- **较小的 $w$** ：图像更自然多样，但可能偏离文本描述

实际应用中，$w=7.5$ 是一个常用的平衡点。

```python
@torch.no_grad()
def sample_with_cfg(model, latents, prompt_embeddings, null_embeddings, 
                    guidance_scale, timesteps):
    """
    使用Classifier-Free Guidance进行采样
    
    Args:
        model: 扩散模型
        latents: 初始噪声 [B, C, H, W]
        prompt_embeddings: 文本条件的嵌入
        null_embeddings: 无条件（空文本）的嵌入
        guidance_scale: 引导强度 w
        timesteps: 时间步序列
    """
    x = latents
    
    for t in timesteps:
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # 复制latents，分别用有条件和无条件处理
        x_combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([t_tensor, t_tensor], dim=0)
        c_combined = torch.cat([prompt_embeddings, null_embeddings], dim=0)
        
        # 模型预测噪声
        noise_pred = model(x_combined, t_combined, c_combined)
        
        # 分离有条件和无条件预测
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
        
        # CFG外推
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        
        # 去噪一步
        x = denoise_step(x, noise_pred, t)
    
    return x
```

---

## 五、DDIM：加速采样的确定性跳跃

DDPM的一个显著缺点是**采样速度慢**——需要完整走完 $T$ 步（通常1000步）才能生成一张图像。

**DDIM**（Denoising Diffusion Implicit Models）通过**非马尔可夫**的确定性采样策略，将采样步数从1000步减少到**20-50步**，同时保持生成质量。

### 5.1 DDIM的核心思想

DDPM的反向过程是一个**随机过程**——每一步都包含随机噪声 $z$。DDIM的关键洞察是：**我们可以设计一个非马尔可夫的前向过程，使得反向过程变为确定性的**。

在DDIM中，采样过程变为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)$$

当 $\sigma_t = 0$ 时，采样过程完全**确定性**——给定相同的初始噪声，总是生成相同的图像。

### 5.2 DDIM的优势

1. **加速采样**：推理时间减少 **10-20倍**
2. **确定性**：相同种子和参数可完美复现结果
3. **一致性插值**：在潜空间中进行平滑插值

```python
@torch.no_grad()
def ddim_sample(model, shape, alpha_bar, T, n_steps, device):
    """
    DDIM采样：用更少的步数生成图像
    
    Args:
        model: 噪声预测网络
        shape: 生成图像的形状
        alpha_bar: 累积噪声调度
        T: 原始总步数（如1000）
        n_steps: 实际采样步数（如50）
    """
    # 在[0, T]范围内均匀采样n_steps个时间点
    step_indices = torch.linspace(0, T-1, n_steps, dtype=torch.long, device=device)
    
    # 从纯噪声开始
    x = torch.randn(shape, device=device)
    
    for i in reversed(range(1, n_steps)):
        t = step_indices[i]
        t_prev = step_indices[i-1]
        
        # 获取对应的alpha_bar
        alpha_bar_t = alpha_bar[t]
        alpha_bar_t_prev = alpha_bar[t_prev]
        
        # 预测噪声
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_theta = model(x, t_tensor)
        
        # DDIM去噪公式（确定性，sigma=0）
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
        
        # 直接预测x_{t-1}
        x = torch.sqrt(alpha_bar_t_prev) * x0_pred + \
            torch.sqrt(1 - alpha_bar_t_prev) * eps_theta
    
    return x
```

---

## 六、总结：扩散模型的技术全景

从DDPM到Stable Diffusion，从U-Net到DiT，从DDPM采样到DDIM加速，扩散模型在短短几年内完成了惊人的技术演进：

| 维度 | 早期（DDPM） | 当前（LDM + DiT + CFG + DDIM） |
|------|------------|-------------------------------|
| 操作空间 | 像素空间 | 潜空间（压缩约50倍） |
| 去噪网络 | U-Net（CNN） | DiT（Transformer） |
| 条件控制 | 无/分类器引导 | Classifier-Free Guidance |
| 采样步数 | 1000步 | 20-50步（DDIM） |
| 典型应用 | 学术研究 | Stable Diffusion, Sora, DALL-E |

扩散模型的成功可以归结为几个关键的设计选择：

1. **数学的简洁性**：复杂的变分下界最终简化为噪声预测的MSE
2. **潜空间的效率**：LDM将计算量降低了一个数量级
3. **Transformer的可扩展性**：DiT让扩散模型享受到了与LLM相同的规模效应
4. **CFG的控制力**：让文本到图像生成变得精准可控
5. **DDIM的实用性**：让扩散模型从实验室走向了产品

正如DDPM论文所展示的：**训练一个扩散模型，本质上就是一个回归问题——用神经网络预测被添加的噪声**。这个看似简单的想法，却开启了一个生成式AI的新时代。从文本生成图像，到文本生成视频，扩散模型正在以前所未有的速度改变着我们创造内容的方式。