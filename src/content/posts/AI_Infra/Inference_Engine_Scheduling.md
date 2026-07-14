---
title: Continuous Batching & Chunked Prefill —— LLM推理引擎调度
published: 2026-04-10
description: 系统讲解大模型推理引擎调度的核心机制：从静态批处理的Head-of-Line Blocking问题出发，剖析连续批处理如何通过迭代级调度实现GPU利用率3-5倍的提升；深入解读Chunked Prefill如何将长Prefill分块交织，在消除队头阻塞的同时实现吞吐量翻倍；分析抢占机制在显存不足时的紧急刹车策略；对比vLLM与SGLang在调度器实现上的差异。
cover: "/assets/images/posts/inference_engine_scheduling.png"
coverInContent: false
tags: [推理引擎调度, 连续批处理, Chunked Prefill, vLLM, SGLang]
category: AI_Infra
draft: false
---

# Continuous Batching & Chunked Prefill —— LLM推理引擎调度

## 引言：调度问题伪装成了硬件问题

服务大语言模型，本质上是一个**调度问题伪装成了硬件问题**。GPU是一台追求吞吐的机器——它希望一次处理成千上万条指令。但推理服务收到的请求是一个一个到达的，时间不可预测，每个请求的输入长度和期望输出长度天差地别。

调度器的任务就是：**用一台擅长批量处理大任务的机器，去服务一股不规则流入的负载**。

本文从静态批处理的失败开始，逐步拆解现代推理引擎的两大调度利器——**连续批处理（Continuous Batching）** 和**分块预填充（Chunked Prefill）** ——以及vLLM与SGLang在调度器实现上的差异。

---

## 一、静态批处理：木桶的短板

### 1.1 最简单的做法，最低的效率

静态批处理的做法很直白：攒够一批请求，一起喂给GPU，等**所有请求都生成完**再处理下一批。

问题在于：LLM的生成长度是**高度不确定**的。一个用户只要5个token，另一个要500个token。在静态批处理中，5-token的请求生成完第5个token后就**干等**——等那个500-token的请求跑完最后495步。

```
静态批处理（典型输出长度方差16-512 tokens）：

Batch 1: [R1: 50 tokens ████░░░░░░] [R2: 500 tokens ██████████] [R3: 30 tokens ███░░░░░░░]
         ↑ R3在30步后完成，但被R2阻塞到500步

结果：GPU利用率损失40-70%
```

### 1.2 三大缺陷

静态批处理的缺陷可以归结为三点：

1. **短请求被长请求“绑架”** （Head-of-Line Blocking）——先到的短请求被后到的长请求卡住
2. **新请求必须等整批完成** ——即使GPU有空闲算力也无法接入新请求
3. **填充开销（Padding）** ——所有序列必须填充到最长长度，浪费计算和内存

---

## 二、连续批处理：迭代级的调度

### 2.1 核心理念：每步调度，而非每批调度

连续批处理（Continuous Batching）最早由**ORCA系统**提出，称为**迭代级调度（Iteration-level Scheduling）** 。核心思想极其简单：

> **在每个解码迭代（即每一步token生成）结束后，重新决定下一轮batch包含哪些请求**。

```python
# 连续批处理的调度循环（伪代码）
while True:
    # 1. 移除已完成的请求
    for req in active_batch:
        if req.is_finished():
            active_batch.remove(req)
    
    # 2. 从等待队列补充新请求（受显存限制）
    while can_fit_in_batch() and waiting_queue:
        req = waiting_queue.pop()
        active_batch.append(req)
    
    # 3. 执行一步模型推理（所有请求共享一次前向传播）
    step_forward(active_batch)
```

vLLM的调度器在每轮迭代中动态决定batch组成，考虑当前队列状态、prompt长度和显存限制。一个请求完成后，其位置**立即被新请求填充**。

### 2.2 为什么连续批处理如此有效？

LLM推理有两个阶段：

| 阶段 | 特性 | 瓶颈 |
|------|------|------|
| **Prefill（预填充）** | 一次性处理整个prompt | **计算密集**（Compute-bound） |
| **Decode（解码）** | 逐token生成 | **内存密集**（Memory-bound） |

连续批处理的价值在于：**让解码阶段的batch始终保持满载**。由于解码是内存密集型的，batch越大，每个token分摊的模型权重加载开销就越小，吞吐量越高。

**实测效果**：连续批处理相比静态批处理可带来**3-5倍的GPU利用率提升**。

### 2.3 连续批处理 ≠ 动态批处理

容易混淆的两个概念：

- **动态批处理（Dynamic Batching）** ：攒够一批或超时后启动，请求到达时间不同但会在批次边界等待
- **连续批处理（Continuous Batching）** ：**每生成一个token就重新决策**，没有“批次边界”的概念

连续批处理是动态批处理的**极致版本**——调度粒度从“请求级”降到了“token级”。

---

## 三、抢占与交换：显存不足时的“紧急刹车”

### 3.1 为什么需要抢占？

连续批处理每步都尝试往batch里塞更多请求，直到显存用完。但有一个问题：**如果显存在请求生成到一半时不够了怎么办？**

解决方案是**抢占（Preemption）** ——把一个正在运行的请求“踢出去”，腾出显存给其他人。

### 3.2 两种抢占模式

vLLM支持两种抢占模式：

**模式一：Swap（交换）**

将抢占请求的KV Cache从GPU显存**交换到CPU内存**，等显存有空时再换回来。

```
GPU显存紧张时：
  Request A (正在生成，已用50个token的KV) 
  → 将50个token的KV打包 → 拷贝到CPU内存
  → GPU显存释放 → 服务其他请求

Request A恢复时：
  → 从CPU内存读回KV → 继续生成
```

**模式二：Recompute（重计算）**

直接丢弃被抢占请求的KV Cache，等恢复时**重新计算**之前所有token的KV。

**权衡**：
- **Swap**：换入换出有I/O开销，但保留已计算成果
- **Recompute**：无I/O开销，但会浪费之前的计算

vLLM V1默认使用**Recompute**，因为在V1架构中重计算的开销更低。

### 3.3 调度器的优先级策略

vLLM调度器有一个重要设计：**每次调度周期优先处理被抢占的请求（preempted_reqs），只有当preempted_reqs为空时才会处理等待队列（waiting）**。

这保证了被抢占的请求不会被“饿死”（starvation）——它们拥有最高的恢复优先级。

---

## 四、Chunked Prefill：解决“队头阻塞”的关键

### 4.1 问题：长Prefill阻塞一切

连续批处理解决了解码阶段的效率问题，但**Prefill阶段仍然是阻塞的**。

当一个长prompt请求（如10万token）进入Prefill阶段时：

```
时间线：
[长请求Prefill 10万token ... 计算数秒] → [之后所有请求才能开始]
                                        ↑ 其他请求全部阻塞
```

这就是**队头阻塞（Head-of-Line Blocking）** ——一个长Prefill请求卡住了整个系统。

### 4.2 核心思想：把Prefill切成小块

**Chunked Prefill**（也称Splitfuse）的核心思想：

> 将长prompt的Prefill阶段**切成多个Chunk**，每个Chunk只包含部分token，与Decode请求**交织**执行。

```
传统方式（阻塞）：
[长请求Prefill 10万token] → [Decode Step] → [Decode Step] → ...

Chunked Prefill（交织）：
[Chunk1: 512 tokens Prefill] → [Decode Step] → [Chunk2: 512 tokens Prefill] → [Decode Step] → ...
                              ↑ 其他请求的Decode插在中间，不被阻塞
```

### 4.3 为什么Chunked Prefill有效？

Prefill是**计算密集**的，Decode是**内存密集**的。两者的资源需求互补：

- Prefill占满GPU的计算单元（Tensor Cores/CUDA Cores）
- Decode占满内存带宽（加载模型权重和KV Cache）

将两者交织，可以让**计算单元和内存带宽同时满负荷工作**，整体吞吐量提升。

**计算总量不变**：Chunked Prefill只是把一次大计算拆成多次小计算，总计算量不变。但通过交织，避免了其他请求的“干等”。

### 4.4 Chunk Size的控制

vLLM中，Chunk Size主要由`max_num_batched_tokens`控制——每次调度时根据剩余token预算自动切分。

```python
# vLLM调度逻辑（简化）
max_num_batched_tokens = 2048  # 每轮最多处理的token数

while True:
    token_budget = max_num_batched_tokens
    # 1. 优先放入Decode请求（每个只需1个token）
    for req in decode_requests:
        if token_budget >= 1:
            add_to_batch(req)
            token_budget -= 1
    
    # 2. 用剩余预算放入Prefill Chunk
    if token_budget > 0 and waiting_prefill_requests:
        chunk_tokens = min(token_budget, remaining_prompt_tokens)
        add_prefill_chunk(chunk_tokens)
```

### 4.5 vLLM V1：Chunked Prefill强制开启

在vLLM V0中，Chunked Prefill是**条件性开启**的。但在**vLLM V1中，Chunked Prefill被设计为核心机制，默认强制开启，无法关闭**。

> V1调度器不再区分Prefill和Decode阶段，Chunked Prefill被视为**一等公民、非可选特性**。

如果尝试在V1中禁用Chunked Prefill，会触发断言错误。要禁用只能用V0引擎（设置`VLLM_USE_V1=0`）。

---

## 五、vLLM与SGLang的调度器差异

### 5.1 共同基础

vLLM和SGLang共享相同的调度基础：

- **连续批处理（Continuous Batching）** ：迭代级调度
- **分页KV Cache（PagedAttention）** ：消除内存碎片
- **Chunked Prefill**：长Prefill分块交织
- **前缀缓存（Prefix Caching）** ：复用共享前缀

所有现代推理引擎（vLLM、SGLang、TGI、TensorRT-LLM）都在用这套组合拳。

### 5.2 调度器的核心差异

尽管基础相同，两者的调度器实现有明显差异：

| 维度 | vLLM | SGLang |
|------|------|---------|
| **调度粒度** | 迭代级（每步调度） | 迭代级 + 请求级混合 |
| **CPU开销** | 中等 | **近零开销（Zero-Overhead）**  |
| **优先级策略** | 优先处理Prefill（优化TTFT） | 多优先级队列 |
| **前缀感知** | 依赖PagedAttention的Block哈希 | **RadixAttention原生集成** |
| **抢占策略** | Swap/Recompute可选 | 同样支持 |

### 5.3 关键差异解读

**CPU开销**：SGLang v0.4实现了“近零开销批调度器”。未优化的推理引擎可能将**高达一半的时间花在CPU调度开销上**。SGLang通过精简调度逻辑、减少Python层开销来逼近零开销。

**前缀感知调度**：vLLM的调度器默认采用**FCFS（先来先服务）** 。而SGLang的RadixAttention天然感知前缀树结构，可以优先调度那些**能复用已有KV Cache**的请求——如果一个请求的前缀已经在GPU缓存中，优先处理它能获得更快的首token延迟。

**Prefill vs Decode优先级**：vLLM默认**优先处理Prefill**，以优化TTFT（Time To First Token）。但这会导致Decode请求的ITL（Inter-Token Latency）变慢。开启Chunked Prefill后，策略会反转——**优先处理Decode请求**，用剩余token预算调度Prefill Chunk。

### 5.4 实测表现差异

在特定场景下，差异显著：

- **16并发请求**：SGLang比vLLM快**4.5倍**（得益于RadixAttention的前缀复用）
- **纯文本生成（无前缀共享）** ：vLLM吞吐略高（约2500 vs 2000-2200 tokens/s）
- **多轮对话**：SGLang单轮延迟约40-50ms，vLLM约80ms

---

## 六、总结：一张图看懂调度演进

```
┌─────────────────────────────────────────────────────────────────────┐
│                       静态批处理（Request-level）                     │
│                 攒一批 → 等最慢的完成 → 再攒下一批                       │
│               问题：短请求被长请求阻塞，GPU利用率40-70%                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ ORCA (OSDI 2022)
┌─────────────────────────────────────────────────────────────────────┐
│                       连续批处理（Iteration-level）                   │       
│           每生成一个token就重新决策batch组成                            │
│           效果：GPU利用率提升3-5倍                                     │
│           支撑技术：PagedAttention（消除碎片）+ 抢占（Swap/Recompute     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Sarathi-Serve / vLLM V1
┌─────────────────────────────────────────────────────────────────────┐
│                     Chunked Prefill（分块预填充）                     │
│                  长Prefill切成小块，与Decode交织执行                   │
│                  效果：消除队头阻塞，吞吐量提升可达2×                    │
│                  vLLM V1：强制开启                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              现代推理引擎调度器（vLLM / SGLang / TGI）                  │
│    共同基础：连续批处理 + PagedAttention + Chunked Prefill + 前缀缓存    │
│    差异：vLLM追求通用吞吐，SGLang追求结构化复用（RadixAttention）          │
└─────────────────────────────────────────────────────────────────────┘
```

**核心权衡**：**吞吐量 vs 延迟 vs 通用性**。

连续批处理让GPU始终保持忙碌，Chunked Prefill让长请求不再阻塞短请求，抢占机制让显存不足时系统依然优雅降级。三者共同构成了现代推理引擎调度器的基石——你用的vLLM、SGLang、TGI，底层都在跑这套逻辑。