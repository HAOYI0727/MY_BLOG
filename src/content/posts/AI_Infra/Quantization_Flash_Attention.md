---
title: Quantization & Flash Attention —— LLM推理优化技术（二）
published: 2026-04-14
description: 系统讲解大模型推理优化的两大核心技术：量化（AWQ/GPTQ）与FlashAttention计算优化。从显存容量与内存带宽两大瓶颈出发，剖析AWQ如何通过激活感知逐通道缩放保护1%显著权重，以及GPTQ如何基于Hessian矩阵的闭式补偿更新实现逐层误差补偿；深入解读FlashAttention的三项核心技术及其从FA1到FA3的演进，揭示FlashAttention-3在H100上达75-85%峰值性能的Hopper专项优化。最后讨论KV Cache量化（FP8/INT8）与FlashAttention的协同效应。
cover: "/assets/images/posts/quantization_flash_attention.png"
coverInContent: false
tags: [量化, AWQ, GPTQ, FlashAttention, KV Cache, 推理优化]
category: AI_Infra
draft: false
---

# Quantization & Flash Attention —— LLM推理优化技术（二）

## 引言：大模型推理的两座大山

大模型推理时，有两座大山压在GPU上：**显存容量**和**内存带宽**。

70B参数模型仅权重就需要140GB（FP16），加上KV Cache、激活值，单卡H100的80GB显存根本装不下。即使装下了，每次生成一个token都要从显存加载140GB权重——内存带宽成了吞吐量的天花板。

应对这两座大山，有两类截然不同的优化思路：

- **量化（Quantization）** ：降低数值精度，把权重从16位降到4位，**显存占用减少75%** ，带宽需求同步下降
- **FlashAttention**：不改变精度，但**重新组织计算顺序**，把对慢速HBM的访问降到最低

前者是“减肥”，后者是“优化物流”。两者互补，共同构成了现代推理引擎的性能基座。

---

## 一、为什么需要量化？精度损失的根源

### 1.1 量化的收益

将模型权重从FP16（16位）降到INT4（4位），理论上：

- **显存占用**：减少 **75%** （从16位到4位）
- **内存带宽需求**：减少 **75%**
- **计算速度**：INT4矩阵乘法比FP16快 **2-4倍**（取决于硬件）

70B模型FP16需要140GB显存 → INT4仅需35GB。一张H100就能跑，还能省下显存放KV Cache。

### 1.2 精度损失的根源：激活异常值（Outliers）

但量化是有代价的——精度损失。为什么？

大语言模型的权重分布有一个显著特征：**少数通道的激活值远大于其他通道**。这些“异常值通道”（Outlier Channels）对模型输出贡献极大。如果粗暴地把所有权重量化到低精度，这些关键通道的量化误差会被放大，导致模型性能崩塌。

AWQ论文的核心发现是：**模型中仅有约0.1%-1%的权重是“显著权重”（Salient Weights），保护这些权重就能大幅减少量化误差**。关键问题是：如何找到这些显著权重？

答案是：**看激活值分布，而不是权重本身**。一个权重乘以一个大的激活值，对层输出的贡献远大于乘以小激活值的权重。所以，显著通道应该由**激活幅度**来定义。

---

## 二、AWQ：激活感知权重量化

### 2.1 核心思想：用激活分布指导量化

AWQ（Activation-aware Weight Quantization）的核心逻辑极其直白：

1. 在少量校准数据上运行模型，**收集每个通道的激活值分布**
2. 计算每个通道的**平均激活幅度** $\mathbb{E}[|X_c|]$
3. 平均激活幅度大的通道 → **显著通道** → 需要特殊保护
4. 对显著通道**放大权重**，再量化，从而降低量化误差

### 2.2 数学原理：等价变换保护显著通道

AWQ不做混合精度量化（部分通道FP16、部分INT4）——那会引入硬件不友好的非结构化存储。AWQ的做法是：

> 对权重矩阵做**逐通道缩放（Per-channel Scaling）** ，在数学上等价于原始矩阵乘法，但能显著降低量化误差。

设权重矩阵为 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$，激活为 $X \in \mathbb{R}^{1 \times d_{\text{in}}}$。层输出为：

$$Y = X \cdot W$$

AWQ对每个输入通道 $c$ 引入缩放因子 $s_c$：

$$Y = X \cdot W = (X \odot s^{-1}) \cdot (s \odot W)$$

其中 $s \odot W$ 表示对权重矩阵的每一行（输入通道）乘以 $s_c$，$X \odot s^{-1}$ 表示对激活的每一列除以 $s_c$。

**这个变换在数学上完全等价**（$X \cdot W = (X/s) \cdot (s \cdot W)$），但量化误差变了：

- 显著通道（激活幅度大）→ $s_c > 1$ → 权重被**放大** → 量化时相对误差变小
- 非显著通道（激活幅度小）→ $s_c < 1$ → 权重被**缩小** → 量化误差相对变大但影响小

### 2.3 缩放因子的搜索

缩放因子 $s_c$ 怎么确定？AWQ采用**网格搜索（Grid Search）** ：

$$s_c = \text{mean}(|X_c|)^{\alpha}, \quad \alpha \in [0, 1]$$

在网格上搜索最优的 $\alpha$，使得量化后的层输出与原始FP16层输出的均方误差（MSE）最小。

$$\alpha^* = \arg\min_{\alpha} \text{MSE}\left(X \cdot W, \ Q\left(X \odot s(\alpha)^{-1}\right) \cdot \left(s(\alpha) \odot W\right)\right)$$

其中 $Q(\cdot)$ 表示量化操作。

**关键优势**：AWQ不需要反向传播或重构训练，只依赖激活的统计信息，因此**泛化能力强**，不会过拟合校准集。AWQ获得了MLSys 2024最佳论文奖。

### 2.4 AWQ vs GPTQ：精度对比

AWQ和GPTQ是目前最主流的两种权重量化方法。综合多项研究：

- **4-bit权重量化**：AWQ通常比GPTQ精度退化更小
- **大模型（如405B）** ：AWQ在MT-Bench等基准上表现优于GPTQ
- **小模型（7B以下）** ：两种方法在4-bit下都可能触发明显的精度崩塌
- **FP8量化**：在所有任务中表现最稳健

但从实现角度看，GPTQ的Hessian加权方法在**数学上更严密**，而AWQ更**工程友好**——实现简单、速度快、泛化好。

---

## 三、GPTQ：基于Hessian的逐层误差补偿

### 3.1 核心思想：Optimal Brain Surgeon

GPTQ的核心思想来自1990年代的剪枝文献——**Optimal Brain Surgeon（OBS）** 。其洞察是：

> 当你改变一个权重时，应该同时调整其他权重来补偿，而不是孤立地处理每个权重。

GPTQ将这个思想形式化为**Hessian加权的闭式补偿更新**，并通过Cholesky分解和列分块使其在Transformer规模上可行。

### 3.2 算法流程

对于每一层，GPTQ执行以下步骤：

**Step 1：前向传播收集校准数据**

在少量校准输入上运行模型到当前层，收集激活 $X \in \mathbb{R}^{K \times d_{\text{in}}}$（$K$ 个样本，$d_{\text{in}}$ 为输入维度）。

**Step 2：计算Hessian矩阵**

$$H = \frac{2}{K} X^T X + \lambda I$$

其中 $\lambda$ 是一个小的正则化系数（ridge），保证矩阵可逆。

**Step 3：逐列量化 + 误差补偿**

对权重矩阵 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 逐列处理：

对第 $j$ 列：

1. **量化**：$q_j = \text{round\_to\_INT4\_grid}(w_j)$（每列有自己的scale）
2. **计算误差**：$\delta_j = w_j - q_j$
3. **补偿后续列**：$W_{:, k > j} \mathrel{-}= \delta_j \cdot \frac{H^{-1}_{j, k > j}}{H^{-1}_{j, j}}$

这里的补偿更新来自OBS公式：

$$\Delta W = -\frac{\delta_j}{[H^{-1}]_{jj}} \cdot H^{-1}_{:, j}$$

**物理含义**：量化第 $j$ 列造成的误差，通过Hessian矩阵的逆分配到**尚未量化的列**上，使得层输出的重构误差最小化。

### 3.3 工程实现的关键

GPTQ的工程贡献在于三个技巧：

1. **列分块（Block of 128）** ：每128列作为一个块处理，块内Hessian逆矩阵惰性计算
2. **Cholesky分解**：直接求逆在近奇异Gram矩阵上不稳定，用Cholesky分解保证数值稳定性
3. **逐列scale**：每个输出列有自己的量化scale（和zero-point）

### 3.4 GPTQ的精度表现

在LLaMA-class模型上，GPTQ-INT4通常能达到FP16基线**0.5-1 perplexity以内**的精度，小模型（7B）比大模型（70B）受INT4量化的影响更明显。

从2023年中到2024年，GPTQ是Hugging Face上最主流的INT4格式——数百万个checkpoint使用GPTQ量化。AutoGPTQ、ExLlama(V2)、vLLM的推理内核最早让INT4权重量化服务在速度上超过了FP16。

---

## 四、FlashAttention：IO-aware的计算优化

量化解决的是“模型太大装不下”的问题。FlashAttention解决的是“注意力计算太慢”的问题——但它不改变精度，而是**重新组织计算**。

### 4.1 标准Attention的IO瓶颈

标准自注意力计算为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

朴素实现需要：

1. 计算 $S = QK^T$，将 $N \times N$ 的 $S$ 矩阵写入HBM
2. 从HBM读回 $S$，计算softmax，再写回HBM
3. 从HBM读回softmax结果，乘以 $V$，写回HBM

**每个元素被反复读写2-4次，每次都经过慢速HBM**。

GPU的内存层次中：

| 存储类型 | 容量 | 带宽 |
|---------|------|------|
| SRAM（片上） | ~20MB（A100） | ~19 TB/s |
| HBM（显存） | 40-80GB | ~1.5-2.0 TB/s |

SRAM带宽是HBM的**10倍以上**，但容量小了3个数量级。标准Attention受内存带宽限制——它花在HBM读写上的时间远超实际计算时间。

### 4.2 FlashAttention的三项核心技术

FlashAttention通过三项技术解决IO瓶颈：

**技术一：分块（Tiling）**

不一次性处理整个矩阵，而是将 $Q, K, V$ 切成**能放进SRAM的小块（tiles）** ，逐块计算。

**技术二：在线Softmax（Online Softmax）**

标准softmax需要知道所有元素才能归一化。FlashAttention采用改进算法，**分块计算softmax**，维护两个中间值（当前最大值和指数和），随块更新。

**技术三：内核融合（Kernel Fusion）**

将所有注意力操作（矩阵乘、掩码、softmax、与V相乘）**融合到单个CUDA kernel中**。不再多次遍历HBM，而是一次性将块加载到SRAM，完成所有计算，只写回最终结果。

### 4.3 IO复杂度分析

标准Attention的HBM访问量为 $\Theta(Nd + N^2)$，而FlashAttention将其降低到 $\Theta(N^2 d^2 M^{-1})$，其中 $M$ 是SRAM大小，$d$ 是head dimension。

当 $M$ 足够大时，HBM访问量大幅减少。FlashAttention证明其IO复杂度在两级内存层次上**理论上最优**。

### 4.4 从FlashAttention-1到FlashAttention-3

**FlashAttention-1（2022）** ：首次提出IO-aware注意力，实现2-4倍加速。

**FlashAttention-2（2023）** ：优化并行化策略和warp分区，在Ampere（A100）上达到约**70%的峰值性能**。

**FlashAttention-3（2024，NeurIPS Spotlight）** ：专门针对**Hopper（H100）架构**优化。在H100上，FlashAttention-2的效率仅约**35%**——因为它没有利用Hopper的新硬件特性。

FlashAttention-3的三项Hopper专项优化：

1. **利用Tensor Core和TMA的异步性**：TMA（Tensor Memory Accelerator）是Hopper中专用于内存加载的硬件单元，与Tensor Core并行工作
2. **Warp专精化（Warp-specialization）** ：不同warp分别负责矩阵乘和softmax，通过**乒乓调度（Ping-pong scheduling）** 重叠计算
3. **FP8低精度支持**：利用Hopper的FP8硬件加速

**性能数据**：

- BF16：H100上达到 **740-840 TFLOPs/s**（**75-85%利用率**）
- FP8：达到接近 **1.2-1.3 PFLOPs/s**
- 相比FlashAttention-2加速 **1.5-2.0倍**

---

## 五、KV Cache量化：拓展上下文窗口的“杠杆”

### 5.1 为什么KV Cache需要量化？

KV Cache的显存占用随**批次大小 × 序列长度 × 层数 × 头数 × 头维度**线性增长。对于长上下文场景（如128K tokens），KV Cache往往比模型权重占用更多显存。

KV Cache量化与权重量化的区别在于：**量化的是缓存本身，不是模型权重**。KV Cache量化后，计算时动态反量化为FP16。

### 5.2 FP8 / INT8量化的收益

| 精度 | 相对FP32 | 相对FP16 |
|------|---------|---------|
| FP8 / INT8 | **4×** 显存节省 | **2×** 显存节省 |

INT8量化可将KV Cache显存占用减少 **75%**（相对FP32）或 **50%**（相对FP16）。

在vLLM中，FP8 KV Cache量化**允许约2倍的KV Cache分配空间**，可以：
- 处理单个请求的**更长上下文长度**
- 处理**更多并发请求**

### 5.3 对上下文窗口的拓展收益

以GLM-5为例，INT4量化将每个token的KV Cache状态体积压缩至**30KB**，相比FP8方案实现约**50%的显存减负**，使得在单节点上承载**128K长文本的高并发会话**成为可能。

**实践中的注意事项**：

- FP8 KV Cache**强依赖硬件**：Hopper/Ada架构（H100、RTX 4090）和AMD MI300系列支持，Ampere（如RTX 3090）不支持
- 部分attention后端**不支持FP8模式**，开启反而更慢
- vLLM中通过 `--kv-cache-dtype fp8` 启用

### 5.4 量化 + FlashAttention的协同

KV Cache量化与FlashAttention可以**协同工作**：

1. KV Cache以FP8/INT8存储 → **显存占用减半**
2. 加载时反量化为FP16 → 送入FlashAttention计算
3. FlashAttention在SRAM中完成注意力计算 → **HBM访问最小化**

两者叠加，长上下文推理的吞吐量可以提升**数倍**。

---

## 六、总结：一张图看懂

```
┌─────────────────────────────────────────────────────────────────────┐
│                         大模型推理的两大瓶颈                           │
│    显存装不下（70B=140GB） + 带宽喂不饱（HBM 1.5TB/s vs 需求 240TB/s）   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
  ┌─────────────────────┐              ┌─────────────────────────────┐
  │     量化（减肥）      │              │   FlashAttention（优化物流）  │
  ├─────────────────────┤              ├─────────────────────────────┤
  │  AWQ（激活感知）      │              │  FlashAttention-1（2022）    │
  │  · 保护1%显著权重     │              │  · 分块 + 在线Softmax         │
  │  · 逐通道缩放         │              │  · 2-4× 加速                 │
  │  · 网格搜索找scale    │              │                             │
  ├─────────────────────┤              │  FlashAttention-2（2023）    │
  │  GPTQ（Hessian补偿）  │              │  · 优化并行化                │
  │  · 逐层量化          │               │  · A100达70%峰值             │
  │  · Hessian加权补偿   │               │                             │
  │  · 闭式更新公式       │              │  FlashAttention-3（2024）    │
  ├─────────────────────┤              │  · Hopper专项优化            │
  │  效果：INT4显存减75%  │              │  · TMA异步 + Warp专精化       │
  │  AWQ精度通常优于GPTQ  │              │  · H100达75-85%利用率        │
  └─────────────────────┘              └─────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     KV Cache量化（杠杆效应）                           │
│         FP8/INT8 → 显存减半 → 2×上下文长度 或 2×并发数                   │
│         与FlashAttention协同 → 长上下文吞吐量数倍提升                    │
└─────────────────────────────────────────────────────────────────────┘
```

**核心要点**：

1. **量化解决“装不下”** ：INT4权重量化减少75%显存，AWQ通过激活感知保护关键通道，GPTQ通过Hessian加权补偿量化误差
2. **FlashAttention解决“算得慢”** ：通过分块+在线Softmax+内核融合，将HBM访问降到最低，FlashAttention-3在H100上达75-85%峰值性能
3. **KV Cache量化是长上下文的“杠杆”** ：FP8/INT8量化让同样显存容纳2倍token，是支撑百万级上下文窗口的关键技术
4. **组合使用效果最佳**：权重量化 + KV Cache量化 + FlashAttention = 大模型推理的“性能铁三角”