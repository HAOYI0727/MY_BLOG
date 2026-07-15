---
title: Vision-Language-Action —— VLA具身智能
published: 2026-08-14
description: 系统讲解具身智能中视觉-语言-动作模型（VLA）的核心技术：从RT-2的“动作即文本”范式出发，剖析动作Token化与Co-Fine-Tuning联合微调策略的数学原理；深入解读动作分块如何通过一次性预测未来K步动作解决时序不一致性问题；推导扩散策略中条件去噪扩散过程的噪声预测损失及其与动作分块的天作之合；并通过三维对比揭示各自优势与融合趋势。
cover: "/assets/images/posts/vla.png"
coverInContent: false
tags: [具身智能, VLA, Diffusion, 多模态, RT-2, Action Chunking]
category: AI_Multimodal
draft: false
---

# Vision-Language-Action —— VLA具身智能

## 引言：当AI从“会说话”到“会动手”

2023年7月，Google DeepMind的一篇论文让整个具身智能圈为之震动。论文的标题直截了当：《RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control》。翻译成大白话就是：**让在互联网上“读过万卷书”的视觉语言模型，直接去控制机器人**。

这个想法听起来有些疯狂——让一个原本只会“看图说话”的AI，去指挥机械臂抓取物体、旋转关节、执行连续运动。但RT-2做到了，而且做得相当漂亮。它开创了 **VLA（Vision-Language-Action）** 这一全新模型类别，将视觉、语言和动作三个原本割裂的领域，统一到了一个模型中。

然而，将高维的连续动作空间塞进一个原本为离散文本设计的语言模型，绝非易事。这就引出了两个关键的技术问题：

1. **如何让语言模型“输出”动作？** —— 答案是 **动作Token化（Action Tokenization）**
2. **如何让输出的动作既平滑又连贯？** —— 答案是 **动作分块（Action Chunking）**

本文将带你深入VLA的技术内核，从RT-2的“动作即文本”思想出发，一路穿越动作分块Transformer（ACT）和扩散策略（Diffusion Policy）的数学原理，用代码和公式为你拆解具身智能最核心的技术栈。

---

## 一、RT-2与VLA：当语言模型学会“输出”动作

### 1.1 什么是VLA？

在RT-2之前，机器人控制通常采用“分层”架构：一个高级规划器将任务分解为子目标，一个低级控制器将子目标转换为关节力矩。这种架构的问题在于：**每一层都是独立优化的，信息在传递中不断丢失**。

RT-2提出了一个更激进的方案：**端到端的VLA模型**。它将视觉感知、语言理解和动作生成全部压缩到一个Transformer中。

VLA的核心思想可以概括为一句话：**让模型像生成文本一样生成动作**。对于模型来说，回答“图像中是什么？”（输出一个文本Token序列）和回答“如何捡起苹果？”（输出一个动作Token序列），在本质上是一样的。

### 1.2 动作Token化：将连续动作“翻译”成语言

机器人的动作空间是**连续**的——机械臂的每个关节可以旋转任意角度，夹爪可以张开任意程度。而语言模型的输出空间是**离散**的——词汇表里每个词是一个独立的Token。

RT-2的核心创新就是解决了这个“连续 vs 离散”的矛盾。具体来说：

**第一步：离散化（Discretization）**

RT-2采用8维动作空间：
- 6个自由度：末端执行器的位置变化（Δposx, Δposy, Δposz）和旋转变化（Δrotx, Δroty, Δrotz）
- 1个维度：夹爪张合程度（gripper_extension）
- 1个维度：终止标志（terminate）

每个连续维度被**离散化为256个区间（bins）** 。这个离散化方案继承自RT-1：每个维度的连续值被映射到0-255之间的一个整数。

```python
import numpy as np

class ActionTokenizer:
    """
    将连续动作离散化为Token
    继承自RT-1的256-bin离散化方案
    """
    def __init__(self, num_bins=256):
        self.num_bins = num_bins
        # 每个动作维度的取值范围（示例）
        self.action_ranges = {
            'delta_pos': (-0.1, 0.1),      # 位置变化：-0.1m ~ 0.1m
            'delta_rot': (-0.2, 0.2),      # 旋转变化：-0.2rad ~ 0.2rad
            'gripper': (0.0, 1.0),         # 夹爪：0（闭合）~ 1（张开）
            'terminate': (0, 1)            # 终止：0（继续）~ 1（完成）
        }
    
    def discretize(self, continuous_action):
        """
        将连续动作向量离散化为Token序列
        
        Args:
            continuous_action: dict，包含各维度的连续值
        Returns:
            list of int: 长度为8的Token序列，每个值在[0, 255]之间
        """
        tokens = []
        for dim, value in continuous_action.items():
            low, high = self.action_ranges[dim]
            # 将连续值映射到[0, num_bins-1]的整数
            normalized = (value - low) / (high - low)
            bin_idx = int(np.clip(normalized * (self.num_bins - 1), 0, self.num_bins - 1))
            tokens.append(bin_idx)
        return tokens
    
    def continuousize(self, tokens):
        """
        将Token序列解码回连续动作
        """
        action = {}
        dims = list(self.action_ranges.keys())
        for i, dim in enumerate(dims):
            low, high = self.action_ranges[dim]
            # 取区间的中心值作为连续值
            value = low + (tokens[i] / (self.num_bins - 1)) * (high - low)
            action[dim] = value
        return action

# 示例：将一个连续动作转化为Token序列
tokenizer = ActionTokenizer()
action = {'delta_pos': 0.03, 'delta_rot': 0.05, 'gripper': 0.7, 'terminate': 0}
tokens = tokenizer.discretize(action)
print(f"动作Token序列: {tokens}")
# 输出示例: [166, 166, 166, 126, 126, 126, 179, 0]
```

**第二步：嵌入语言模型**

这些离散化的动作Token被**嵌入到模型的语言字典中**，与自然语言Token共用同一表示空间。模型在训练时，既能预测“苹果”这样的文本Token，也能预测“128 91 241 5 101”这样的动作Token。

这种设计的精妙之处在于：**模型不需要学习任何新的“动作模块”** 。它原本的文本生成能力被直接“复用”到了动作生成上。

### 1.3 Co-Fine-Tuning：联合微调

仅仅把动作变成Token还不够——模型还需要学会在什么情况下生成什么样的动作Token。RT-2采用了**联合微调（Co-Fine-Tuning）** 策略。

训练时，每个Batch中同时混合了两种数据：
1. **机器人轨迹数据**：（图像，指令）→ 动作Token
2. **互联网视觉语言数据**：（图像，问题）→ 文本回答

这种设计的深层逻辑是：**让模型在“学动作”的同时不“遗忘”视觉语言知识**。只做机器人数据微调会导致模型“灾难性遗忘”——失去从互联网数据中获得的丰富语义理解能力。而联合微调让模型既能保持视觉语言的泛化能力，又能学会输出有效的动作。

RT-2构建了两个版本的模型：
- **RT-2-PaLI-X**：基于PaLI-X，最大版本达**550亿**参数
- **RT-2-PaLM-E**：基于PaLM-E，120亿参数

最大的55B模型可以以**1-3Hz**的频率运行，5B的小版本可达**5Hz**。

### 1.4 RT-2的涌现能力

当你在互联网规模的视觉语言数据上预训练一个模型，再让它学习输出动作Token时，一些意想不到的能力会“涌现”出来：

- **对新物体的泛化**：看到训练中从未见过的物体，也能正确抓取
- **对未见命令的理解**：能够执行训练数据中从未出现过的指令（如“把物体放到数字3上面”）
- **基础推理能力**：能够理解“拿起最小的物体”或“拿起离另一个物体最近的物体”
- **多步推理（Chain-of-Thought）** ：能够进行多阶段的语义推理，比如“找出可以用作临时锤子的物体”（答案是石头）或“给疲倦的人最适合喝什么”（答案是能量饮料）

---

## 二、动作分块（Action Chunking）：从“一步一动”到“批量规划”

### 2.1 为什么需要动作分块？

早期的模仿学习模型采用**单步预测**策略：每一帧观测输入模型，模型输出一个动作，机器人执行这个动作，然后重复这个过程。

这种方式存在一个致命的问题：**时序不一致性（Temporal Inconsistency）** 。模型在每一帧独立决策，前后动作之间缺乏连贯性，导致机器人的运动**抖动、不平稳**。

动作分块（Action Chunking）的核心思想是：**让模型一次性预测未来多步的动作序列，然后逐步执行**。

数学上，设当前时刻为$t$，模型一次性预测未来$K$步的动作：

$$\hat{\mathbf{a}}_{t:t+K} = \pi_\theta(\mathbf{o}_t, \mathbf{g})$$

其中$\mathbf{o}_t$是当前观测，$\mathbf{g}$是任务目标，$\hat{\mathbf{a}}_{t:t+K}$是未来$K$步的动作序列。

在执��时，通常采用**部分执行**策略：只执行前$k$步（$k < K$），然后重新观测、重新规划。这种“滑动窗口”策略既保证了动作的平滑性，又保持了对外界变化的响应能力。

### 2.2 ACT：动作分块Transformer

**ACT（Action Chunking with Transformers）** 是将动作分块与Transformer结合的代表性工作。

ACT的架构包含两个关键设计：

**（1）条件变分自编码器（CVAE）**

ACT引入CVAE来建模动作的多模态分布。对于同一个观测，可能存在多种合理的动作轨迹——比如抓取一个杯子，可以从左侧抓，也可以从右侧抓。CVAE通过一个**潜在变量$z$** 来捕捉这种多模态性：

$$z \sim \mathcal{N}(\mu_\phi(\mathbf{o}_t, \mathbf{a}_{t:t+K}), \sigma_\phi(\mathbf{o}_t, \mathbf{a}_{t:t+K}))$$

$$\hat{\mathbf{a}}_{t:t+K} = \text{Decoder}_\theta(\mathbf{o}_t, z)$$

**（2）Transformer解码器**

ACT使用Transformer解码器来生成动作序列。与RT-2将动作离散化为Token不同，ACT直接**回归连续动作值**。解码器以观测特征和潜在变量$z$为条件，自回归地生成$K$步动作。

```python
import torch
import torch.nn as nn

class ActionChunkingTransformer(nn.Module):
    """
    简化的ACT（Action Chunking with Transformers）模型
    """
    def __init__(self, obs_dim, action_dim, chunk_size=50, 
                 latent_dim=32, num_heads=8, num_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        
        # 观测编码器
        self.obs_encoder = nn.Linear(obs_dim, 256)
        
        # 潜在变量编码器（CVAE的encoder）
        self.latent_encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim * chunk_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mean + log_var
        )
        
        # Transformer解码器
        self.action_embed = nn.Linear(action_dim, 256)
        self.pos_embed = nn.Parameter(torch.randn(chunk_size, 256))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(256, num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.action_head = nn.Linear(256, action_dim)
    
    def forward(self, obs, actions=None, train=True):
        # obs: [B, obs_dim]
        obs_feat = self.obs_encoder(obs)  # [B, 256]
        
        if train and actions is not None:
            # 训练模式：使用CVAE
            # actions: [B, chunk_size, action_dim]
            latent_input = torch.cat([obs, actions.flatten(1)], dim=-1)
            latent_params = self.latent_encoder(latent_input)
            mean, log_var = latent_params.chunk(2, dim=-1)
            # 重参数化采样
            z = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        else:
            # 推理模式：从先验采样
            z = torch.randn(obs.size(0), self.latent_dim, device=obs.device)
        
        # 以观测特征和潜在变量为条件
        # 简化为：将条件拼接到每个时间步
        cond = torch.cat([obs_feat, z], dim=-1)  # [B, 256+latent_dim]
        cond_proj = nn.Linear(256 + self.latent_dim, 256).to(obs.device)(cond)
        cond_proj = cond_proj.unsqueeze(1).expand(-1, self.chunk_size, -1)
        
        # 生成动作序列
        # 用可学习的起始token + 位置编码
        action_tokens = torch.zeros(obs.size(0), self.chunk_size, 256, device=obs.device)
        action_tokens = action_tokens + self.pos_embed.unsqueeze(0)
        
        # Transformer解码器（以条件为memory）
        decoded = self.decoder(action_tokens, cond_proj)  # [B, chunk_size, 256]
        actions_pred = self.action_head(decoded)  # [B, chunk_size, action_dim]
        
        return actions_pred
```

ACT的核心优势在于：**通过一次性预测$K$步动作，模型被迫学习动作之间的时序依赖关系，从而生成更平滑、更连贯的运动**。

### 2.3 动作分块的核心参数：块大小（Chunk Size）

块大小$K$的选择是一个关键的**超参数**：

- **$K$太小**：模型退化为单步预测，动作抖动
- **$K$太大**：模型需要预测太远的未来，精度下降；且“开环”执行时间过长，无法应对外界变化

实际应用中，常见的配置是：**预测50步，执行15步**。这种“预测多一点、执行少一点”的策略既保证了平滑性，又保持了 reactivity。

### 2.4 动作分块的数学本质

从控制理论的角度看，动作分块可以理解为一种**模型预测控制（MPC）的变体**。MPC在每个时间步求解一个有限时域的最优控制问题，然后执行第一个控制量。动作分块与此类似，但区别在于：

- MPC需要在线求解优化问题（计算密集）
- 动作分块通过**离线学习的生成模型**直接输出最优轨迹（计算高效）

---

## 三、扩散策略（Diffusion Policy）：用扩散模型生成平滑动作

如果说RT-2代表了“用语言模型生成动作”的路线，ACT代表了“用Transformer生成动作序列”的路线，那么**扩散策略（Diffusion Policy）** 则代表了第三条路——**用扩散模型生成动作**。

### 3.1 为什么扩散模型适合机器人控制？

扩散模型在图像生成领域已经大放异彩（我们在第四篇中详细讨论过）。将扩散模型应用于机器人控制，有几个天然的优势：

1. **多模态分布建模**：机器人的动作分布天然是多模态的——同一个任务有多种完成方式。扩散模型擅长建模复杂的多模态分布
2. **时序一致性**：扩散模型的迭代去噪过程天然考虑了动作序列的整体结构，生成的轨迹平滑自然
3. **鲁棒性**：扩散模型对噪声和扰动具有较强的鲁棒性

### 3.2 Diffusion Policy的数学原理

**Diffusion Policy**将机器人的视觉运动策略建模为一个**条件去噪扩散过程**。

与DDPM类似，Diffusion Policy包含前向加噪和反向去噪两个过程。但区别在于：

- **DDPM的去噪目标**：图像像素
- **Diffusion Policy的去噪目标**：动作序列 $\mathbf{a}_{t:t+K}$

**前向过程**：向动作序列逐步添加高斯噪声

$$q(\mathbf{a}^{i} \mid \mathbf{a}^{i-1}) = \mathcal{N}(\mathbf{a}^{i}; \sqrt{1-\beta_i} \mathbf{a}^{i-1}, \beta_i \mathbf{I})$$

其中 $\mathbf{a}^{0} = \mathbf{a}_{t:t+K}$ 是原始动作序列，$\mathbf{a}^{N}$ 是纯噪声。

**反向过程**：学习从噪声中恢复动作序列

$$p_\theta(\mathbf{a}^{i-1} \mid \mathbf{a}^{i}, \mathbf{o}_t, \mathbf{g}) = \mathcal{N}(\mathbf{a}^{i-1}; \mu_\theta(\mathbf{a}^{i}, \mathbf{o}_t, \mathbf{g}, i), \Sigma_i)$$

与DDPM一样，训练目标简化为**噪声预测**：

$$\mathcal{L} = \mathbb{E}_{i, \mathbf{a}^0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\mathbf{a}^i, \mathbf{o}_t, \mathbf{g}, i) \|^2 \right]$$

其中 $\mathbf{a}^i = \sqrt{\bar{\alpha}_i} \mathbf{a}^0 + \sqrt{1-\bar{\alpha}_i} \epsilon$。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionPolicy(nn.Module):
    """
    简化的Diffusion Policy模型
    用扩散模型生成动作序列
    """
    def __init__(self, obs_dim, action_dim, chunk_size=16, 
                 n_diffusion_steps=100, hidden_dim=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_diffusion_steps = n_diffusion_steps
        self.action_dim = action_dim
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 噪声预测网络（类似DDPM中的U-Net/DiT）
        # 输入：带噪动作序列 + 时间步 + 观测条件
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim * chunk_size + hidden_dim + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, action_dim * chunk_size)
        )
        
        # 噪声调度（预计算）
        self.betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, obs, actions=None, train=True):
        """
        训练：预测噪声
        推理：从噪声生成动作序列
        """
        obs_feat = self.obs_encoder(obs)  # [B, hidden_dim]
        
        if train and actions is not None:
            # 训练模式
            B = actions.shape[0]
            # actions: [B, chunk_size, action_dim]
            actions_flat = actions.flatten(1)  # [B, chunk_size * action_dim]
            
            # 随机采样时间步
            t = torch.randint(0, self.n_diffusion_steps, (B,), device=obs.device)
            
            # 采样噪声
            noise = torch.randn_like(actions_flat)
            
            # 加噪
            alpha_bar = self.alpha_bars[t].view(-1, 1)
            noisy_actions = torch.sqrt(alpha_bar) * actions_flat + \
                            torch.sqrt(1 - alpha_bar) * noise
            
            # 预测噪声
            t_norm = t.float().view(-1, 1) / self.n_diffusion_steps
            model_input = torch.cat([noisy_actions, obs_feat, t_norm], dim=-1)
            noise_pred = self.denoiser(model_input)
            
            # 损失：预测噪声的MSE
            loss = F.mse_loss(noise_pred, noise)
            return loss
        
        else:
            # 推理模式：从纯噪声开始去噪
            B = obs.shape[0]
            # 从纯噪声开始
            x = torch.randn(B, self.chunk_size * self.action_dim, device=obs.device)
            
            # 逐步去噪
            for i in reversed(range(self.n_diffusion_steps)):
                t = torch.full((B,), i, device=obs.device, dtype=torch.long)
                t_norm = t.float().view(-1, 1) / self.n_diffusion_steps
                
                # 预测噪声
                model_input = torch.cat([x, obs_feat, t_norm], dim=-1)
                noise_pred = self.denoiser(model_input)
                
                # 去噪一步（DDPM采样）
                alpha = self.alphas[i]
                alpha_bar = self.alpha_bars[i]
                beta = self.betas[i]
                
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
                )
                
                if i > 0:
                    x = x + torch.sqrt(beta) * torch.randn_like(x)
            
            # 重塑为动作序列
            actions_pred = x.view(B, self.chunk_size, self.action_dim)
            return actions_pred
```

### 3.3 扩散策略 + 动作分块：天作之合

**扩散策略天然适配动作分块**。理由如下：

1. **扩散模型生成的是“序列”而非“单点”** ：DDPM的去噪过程输出的是一个完整的张量——在图像生成中是整张图，在动作生成中是完整的动作序列。这与动作分块“一次性预测多步”的理念完美契合

2. **双向上下文建模**：扩散模型的去噪过程是**迭代优化**的，每一步去噪都考虑了序列中所有位置之间的关系。这与自回归模型的“从左到右”生成不同，扩散模型能够利用**未来信息来优化当前预测**，从而生成更连贯的轨迹

3. **多模态性**：扩散模型通过随机采样不同的初始噪声，可以生成多样化的动作轨迹，适合处理机器人控制中的多模态问题

实验表明，基于扩散的VLA在机器人控制任务上表现优于自回归基准模型。

### 3.4 扩散策略的挑战与优化

扩散策略的主要挑战是**推理速度**。DDPM通常需要数十到上百步迭代才能生成一个动作序列，这对于实时机器人控制来说太慢了。

针对这一问题，研究者提出了多种优化方案：

- **DDIM加速**：将采样步数从100步减少到10-20步（我们在第四篇中详细讨论过）
- **实时分块（RTC）** ：一种推理时算法，支持扩散策略的异步实时执行
- **块自适应缓存（BAC）** ：通过缓存中间动作特征来加速推理

---

## 四、技术全景：三种路线的对比与融合

至此，我们已经介绍了VLA领域的三种核心技术路线。它们之间的关系可以用下表概括：

| 维度 | RT-2（VLA） | ACT（动作分块Transformer） | Diffusion Policy |
|------|------------|--------------------------|------------------|
| **核心思想** | 动作即文本 | 一次性预测多步动作 | 用扩散模型生成动作序列 |
| **动作表示** | 离散Token（256 bins） | 连续值回归 | 连续值（扩散去噪） |
| **生成方式** | 自回归（从左到右） | 自回归（Transformer解码） | 迭代去噪（双向上下文） |
| **多模态性** | 较弱（离散化损失） | CVAE潜在变量 | 强（随机采样噪声） |
| **平滑性** | 中等 | 好（多步预测） | 极好（整体去噪） |
| **推理速度** | 快（1-3Hz for 55B） | 中等 | 较慢（需多步迭代） |
| **代表工作** | RT-2, OpenVLA | ACT | Diffusion Policy, RDT-1B |

### 4.1 三条路线的演进逻辑

**RT-2的路线**代表了“最大化复用互联网知识”的思路。它不追求动作表示的精度，而是追求让VLM“原生”地输出动作。这种思路的优势在于**泛化性**——模型可以从互联网规模的视觉语言数据中汲取丰富的语义知识。

**ACT的路线**代表了“用Transformer建模时序依赖”的思路。它专注于让模型学会动作之间的时序关系，从而生成平滑、连贯的运动。

**Diffusion Policy的路线**代表了“用生成模型建模动作分布”的思路。它利用扩散模型强大的分布建模能力，生成高质量、多模态的动作序列。

### 4.2 融合趋势：新一代VLA

最新的VLA模型正在融合这三条路线的优势：

- **RDT-1B（Robotics Diffusion Transformer）** ：将扩散模型与Transformer架构结合，专为双臂精细操作设计
- **Dream-VLA**：以扩散语言模型为骨干，天生适合动作分块和并行生成
- **Diffusion Transformer Policy**：用大型多模态扩散Transformer直接对动作块进行去噪

这些工作的共同趋势是：**用扩散模型生成动作序列 + 用Transformer提供可扩展性 + 用动作分块保证时序一致性**。

---

## 五、代码实践：一个完整的动作分块VLA示例

让我们用一个简化的端到端示例，把本文的核心概念串起来。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class ActionChunkingVLA(nn.Module):
    """
    简化的动作分块VLA模型
    结合了：视觉编码器 + 语言编码器 + 动作分块解码器
    """
    def __init__(self, vocab_size, embed_dim=256, action_dim=8, 
                 chunk_size=16, num_heads=8, num_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # 视觉编码器（简化：用线性层代替ViT）
        self.vision_encoder = nn.Sequential(
            nn.Linear(3*224*224, 512),  # 假设输入224x224x3
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
        # 语言编码器（简化：词嵌入 + Transformer）
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True),
            num_layers=2
        )
        
        # 动作分块解码器
        self.action_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.action_head = nn.Linear(embed_dim, action_dim)
        
        # 可学习的动作查询（类似ACT的起始token）
        self.action_query = nn.Parameter(torch.randn(chunk_size, embed_dim))
        
        # 动作分词器（256 bins）
        self.tokenizer = ActionTokenizer(num_bins=256)
    
    def encode_vision(self, image):
        """编码视觉输入"""
        # image: [B, 3, 224, 224]
        B = image.shape[0]
        image_flat = image.view(B, -1)
        return self.vision_encoder(image_flat)  # [B, embed_dim]
    
    def encode_text(self, text_tokens):
        """编码文本指令"""
        # text_tokens: [B, seq_len]
        emb = self.text_embedding(text_tokens)
        return self.text_encoder(emb).mean(dim=1)  # [B, embed_dim]
    
    def forward(self, image, text_tokens, actions=None, train=True):
        """
        前向传播
        Args:
            image: [B, 3, 224, 224]
            text_tokens: [B, seq_len]
            actions: [B, chunk_size, action_dim] (训练时提供)
        """
        # 1. 编码多模态输入
        vision_feat = self.encode_vision(image)  # [B, embed_dim]
        text_feat = self.encode_text(text_tokens)  # [B, embed_dim]
        
        # 2. 融合视觉和语言特征
        cond = (vision_feat + text_feat) / 2  # [B, embed_dim]
        cond = cond.unsqueeze(1)  # [B, 1, embed_dim]
        
        # 3. 动作分块解码
        # 扩展条件到每个时间步
        cond_expanded = cond.expand(-1, self.chunk_size, -1)
        
        # 可学习的动作查询 + 条件
        query = self.action_query.unsqueeze(0).expand(image.shape[0], -1, -1)
        query = query + cond_expanded  # 注入条件
        
        # Transformer解码
        decoded = self.action_decoder(query, cond_expanded)
        actions_pred = self.action_head(decoded)  # [B, chunk_size, action_dim]
        
        if train and actions is not None:
            # 训练损失：MSE + 动作Token化辅助损失
            mse_loss = F.mse_loss(actions_pred, actions)
            
            # 可选：离散化辅助损失（让模型学会输出合理的bin）
            # 将预测的连续动作离散化，与真实动作的Token比较
            return mse_loss
        
        return actions_pred

# ============ 推理示例 ============
def run_robot_control():
    """模拟机器人闭环控制"""
    model = ActionChunkingVLA(vocab_size=10000)
    model.eval()
    
    # 模拟环境
    class SimEnv:
        def __init__(self):
            self.obs = torch.randn(1, 3, 224, 224)
            self.state = np.zeros(8)
        def step(self, action):
            # 执行动作，返回新观测
            self.state = action[0] if len(action.shape) > 1 else action
            self.obs = torch.randn(1, 3, 224, 224)
            return self.obs
    
    env = SimEnv()
    
    # 任务指令
    text = "pick up the red cube"
    text_tokens = torch.randint(0, 10000, (1, 20))
    
    # 控制循环
    action_buffer = deque()
    chunk_size = 16
    execute_steps = 5  # 每次执行5步，然后重新规划
    
    for step in range(100):
        # 如果缓冲区为空，重新规划
        if len(action_buffer) == 0:
            with torch.no_grad():
                # 生成动作块
                actions_chunk = model(env.obs, text_tokens, train=False)
                # actions_chunk: [1, chunk_size, action_dim]
                action_buffer.extend(actions_chunk.squeeze(0).tolist())
        
        # 从缓冲区取一个动作执行
        action = action_buffer.popleft()
        env.step(action)
        
        # 每执行execute_steps步，清空缓冲区强制重新规划
        if step % execute_steps == 0:
            action_buffer.clear()
    
    print("控制循环完成")

if __name__ == "__main__":
    run_robot_control()
```

---

## 六、总结与展望

从RT-2的“动作即文本”到ACT的“动作分块Transformer”，再到Diffusion Policy的“扩散生成动作”，VLA领域在短短两年内经历了飞速的演进：
1. **RT-2开创了VLA范式**：证明了将预训练VLM的知识直接迁移到机器人控制是可行的。动作Token化+联合微调成为后续工作的标准配置
2. **动作分块解决了时序一致性问题**：通过一次性预测多步动作，ACT和Diffusion Policy让机器人的运动从“抖动”变得“平滑”
3. **扩散策略带来了高质量的动作生成**：利用扩散模型的多模态建模能力，生成的动作序列既平滑又多样

当前，这三条路线正在走向融合。最新的VLA模型——如RDT-1B、Diffusion Transformer Policy、Dream-VLA——正在将**扩散模型**、**Transformer架构**和**动作分块**三者结合起来。

展望未来，VLA的发展可能沿着以下几个方向继续突破：
- **实时性**：通过模型蒸馏、量化、DDIM加速等技术，让大模型VLA能够在嵌入式设备上实时运行
- **长时程规划**：将动作分块与层次化规划结合，实现更长时间跨度的任务执行
- **跨具身泛化**：让同一个VLA模型能够控制不同形态的机器人（从机械臂到人形机器人）

正如RT-2论文所展示的：**当大模型学会“动手”时，具身智能的商业化时间表被往前挪了三年**。而动作分块和扩散策略，正是让这双手“稳得住、动得顺”的关键技术。