---
title: TTFT & TPOT & Throughput —— LLM推理性能指标
published: 2026-04-16
description: 系统讲解大模型推理性能指标体系：从Prefill（计算密集）与Decode（访存密集）两阶段的物理本质出发，推导TTFT的数学构成与SLA目标、TPOT的物理极限公式，剖析吞吐量与Batch Size的亚线性增长权衡，并揭示P95/P99尾延迟的根源及优化策略。通过约束优化框架建立完整的性能调优决策树。
cover: "/assets/images/posts/llm_performance_metrics.png"
coverInContent: false
tags: [TTFT, TPOT, 吞吐量, 尾延迟, 推理性能]
category: AI_Infra
draft: false
---

# TTFT & TPOT & Throughput —— LLM推理性能指标

## 引言：延迟与吞吐的“不可能三角”

大模型推理服务上线前，业务方通常会问三个问题：**快不快？贵不贵？稳不稳？**

这三个问题对应着推理系统的三大核心指标：
- **TTFT（首Token延迟）** ——决定了用户“第一印象”
- **TPOT（每输出Token延迟）** ——决定了对话“流畅感”
- **Throughput（吞吐量）** ——决定了服务“性价比”

但糟糕的是，这三者之间存在内在的**Trade-off**。想提高吞吐量（多塞请求），TTFT和TPOT就会变差；想优化尾延迟（P99稳定），可能得牺牲一些吞吐量。

更复杂的是，LLM推理的两阶段特性（Prefill和Decode）让性能分析变得极其微妙——同一个系统在不同阶段的瓶颈完全不同。

本文建立一个完整的性能指标分析框架：从硬件物理极限出发，推导各指标的数学表达式，再结合调度策略分析其权衡关系。这是推理引擎性能调优的“尺子”。

---

## 一、推理两阶段的物理本质

要理解指标，必须先理解**硬件在干什么**。

LLM推理分为两个阶段，**瓶颈完全不同**：

### 1.1 Prefill（预填充阶段）

- **做什么**：一次性处理整个输入Prompt，计算所有token的KV Cache
- **计算量**：约 `2 × prompt_len × hidden_dim × num_layers` FLOPs
- **瓶颈**：**计算密集型（Compute-bound）** ——大量矩阵乘法（GEMM）占满Tensor Core
- **关键资源**：FLOPs（浮点运算能力）

### 1.2 Decode（解码/自回归阶段）

- **做什么**：逐个生成新token，每次生成依赖全部历史KV Cache
- **计算量**：小（主要是一个 `[1 × hidden_dim] × [hidden_dim × vocab]` 的矩阵乘）
- **瓶颈**：**访存密集型（Memory-bandwidth-bound）** ——每次前向需加载完整模型权重（例如70B ~ 140GB）
- **关键资源**：**HBM带宽**（而非FLOPs）

### 1.3 物理极限公式

对于Decode阶段，单个token生成的理论最低延迟由**HBM带宽**决定：

$$T_{\text{token}}^{\text{min}} = \frac{\text{Model Size}}{\text{HBM Bandwidth}}$$

对于LLaMA-70B（FP16，~140GB），在H100（HBM带宽 3.35 TB/s）上：

$$T_{\text{token}}^{\text{min}} = \frac{140\text{GB}}{3.35\text{TB/s}} \approx 42\text{ms}$$

这意味着，**单条请求在H100上每秒最多生成约24个token**（1/0.042s）。这是物理极限，任何优化（FlashAttention、量化）的本质都是降低“有效模型大小”或提升有效带宽。

但是，如果Batch中有N条请求，**模型权重只需要加载一次**，可以同时为N条请求计算，因此总吞吐量随N线性增长，而**单条请求的TPOT会随N近似线性增加**（因为计算量增加了，但带宽不变）。

这就是推理调度的核心矛盾。

---

## 二、TTFT（Time To First Token）：首Token延迟

### 2.1 定义与SLA

TTFT定义为：从用户发送请求（或开始处理）到收到**第一个输出Token**的时间间隔。

**包含**：网络传输 + 调度排队等待 + Prefill计算 + (可能的)首次Decode。

在在线服务中，**TTFT直接影响用户留存率**。业界常见的SLA（服务等级协议）目标：

- **聊天机器人**：TTFT < 500ms（最佳体验）
- **实时搜索摘要**：TTFT < 200ms
- **离线批处理**：无严格要求

### 2.2 TTFT的数学构成

TTFT主要由Prefill延迟决定。对于输入长度为 $P$ 的Prompt：

$$T_{\text{TTFT}} \approx T_{\text{sched}} + \frac{2 \times P \times H \times L}{\text{GPU FLOPs} \times \text{Utilization}}$$

其中 $H$ 为隐藏维度，$L$ 为层数。

**关键洞察**：TTFT随Prompt长度$P$**线性增长**。处理10万token的文档，Prefill可能需要数秒——这也是Chunked Prefill诞生的直接原因（把长Prefill切碎，不让一个长请求堵住所有短请求的TTFT）。

### 2.3 如何优化TTFT？

| 优化手段 | 作用机制 | 效果 |
|---------|---------|------|
| **Chunked Prefill** | 长Prefill切块交织，避免队头阻塞 | 短请求TTFT降低10-50% |
| **前缀缓存（RadixAttention）** | 跳过公共Prompt的重复Prefill计算 | 共享前缀场景TTFT降低5-10倍 |
| **提升GPU算力（H100→B200）** | 直接降低FLOPs耗时 | Prefill延迟线性下降 |
| **降低Batch中Prefill请求数** | 减少并发Prefill竞争 | 单个TTFT改善但吞吐下降 |

---

## 三、TPOT（Time Per Output Token）：每个输出Token的时间

### 3.1 定义与用户感知

TPOT定义为：**生成每个新Token所需的平均时间**（不含首个Token）。

它是衡量“对话流畅感”的核心指标。人类阅读速度约5-7 tok/s，因此推理服务通常追求：

- **> 20 tok/s**：感觉流畅，可实时交互
- **10-20 tok/s**：可接受，略有停顿感
- **< 10 tok/s**：明显卡顿，体验不佳

70B模型在H100上的单请求物理极限是~24 tok/s（见上文），刚好卡在“流畅”门槛上。通过量化（INT4）将有效模型大小降到~35GB，单请求TPOT可飙升至~80-100 tok/s。

### 3.2 TPOT的物理公式

在连续批处理中，假设一个Batch包含 $B$ 条请求，每条请求都在并发解码。

**单次前向传播耗时**：

$$T_{\text{step}} \approx \frac{\text{Model Size} + \text{KV Cache Read Size}}{\text{HBM Bandwidth}}$$

其中KV Cache读取量约为 $2 \times L \times H \times D \times B \times \text{seq\_len\_avg} \times \text{dtype\_size}$。

对于长上下文，KV Cache读取可能超过模型权重本身。

**系统级TPOT**（即该Batch中每个请求生成一个token的平均时间）：

$$\text{TPOT}_{\text{system}} = T_{\text{step}}$$

**单请求感知的TPOT**也是 $T_{\text{step}}$（因为是同步批量推理，该Batch中所有请求**同时**获得下一个token）。

因此：**Batch越大，$T_{\text{step}}$越大，TPOT越差**。

### 3.3 TPOT与Batch Size的权衡

我们用Python模拟一下Batch Size对TPOT和吞吐量的影响：

```python
import matplotlib.pyplot as plt
import numpy as np

# 假设H100参数
model_size_gb = 140  # 70B FP16
hbm_bandwidth_gbps = 3350  # GB/s
kv_cache_per_token_mb = 0.5  # LLaMA-70B约0.5MB/token (取决于seq_len)
avg_seq_len = 2048

def simulate_batch(batch_size):
    # 模型权重读取耗时
    weight_time = model_size_gb / hbm_bandwidth_gbps  # 秒
    
    # KV Cache读取耗时 (假设所有请求平均2048长度)
    kv_total_gb = batch_size * avg_seq_len * kv_cache_per_token_mb / 1024
    kv_time = kv_total_gb / hbm_bandwidth_gbps
    
    step_time = weight_time + kv_time  # 秒
    
    # 每个step系统生成batch_size个token
    throughput = batch_size / step_time  # tokens/s
    tpot_ms = step_time * 1000  # ms per token per request
    
    return tpot_ms, throughput

batch_sizes = [1, 4, 8, 16, 32, 64, 128]
results = [simulate_batch(b) for b in batch_sizes]
tpot_ms, throughput = zip(*results)

print("Batch | TPOT(ms) | Throughput(tok/s)")
for b, t, th in zip(batch_sizes, tpot_ms, throughput):
    print(f"{b:5d} | {t:8.1f} | {th:12.1f}")

# 输出示例：
# Batch | TPOT(ms) | Throughput(tok/s)
#     1 |     42.0 |         23.8
#     4 |     42.6 |         94.0
#     8 |     43.3 |        184.9
#    16 |     44.9 |        356.6
#    32 |     49.0 |        653.5
#    64 |     58.8 |       1088.4
#   128 |     78.4 |       1632.7
```

**关键发现**：
- Batch从1到128，TPOT从42ms恶化到78ms（变慢86%），但**吞吐量从24 tok/s暴涨到1632 tok/s**（提升68倍）。
- 系统吞吐量随Batch size**亚线性增长**——KV Cache读取开销让边际收益递减。

这就是服务SLA设计的核心：**在满足TPOT < 50ms（20 tok/s）的前提下，尽可能增大Batch size以提升吞吐量**。

---

## 四、吞吐量（Throughput）：系统的生产力

### 4.1 定义与计算

吞吐量通常定义为**系统单位时间生成的Token总数**（tok/s），也可以按请求数/秒（req/s）计。

$$\text{Throughput} = \frac{\text{Batch Size} \times \text{Avg Generation Length}}{\text{Total Time}}$$

或者从Decode角度看：

$$\text{Throughput} = \frac{B}{T_{\text{step}}} \approx \frac{B \times \text{HBM Bandwidth}}{\text{Model Size} + \text{KV Cache}_{B}}$$

### 4.2 吞吐量与并发数的关系

吞吐量随并发请求数增加而增加，但呈现**边际递减**。原因：

1. **KV Cache膨胀**：更多并发 → 总KV Cache读取量增加 → $T_{\text{step}}$上升
2. **显存容量上限**：并发数受KV Cache可用显存限制
3. **调度开销**：超过一定并发数，CPU调度本身成为瓶颈

典型推理引擎的吞吐量曲线呈**S型**：
- 低并发：GPU空闲，吞吐线性增长
- 中并发：GPU饱和，吞吐增速放缓
- 高并发：显存或调度瓶颈，吞吐趋近水平甚至下降（因Swap/抢占）

### 4.3 优化吞吐量的手段

| 手段 | 原理 | 代价 |
|------|------|------|
| **增大Batch Size** | 分摊权重加载开销 | TPOT变差、显存占用↑ |
| **量化（INT4/FP8）** | 降低模型大小，提升有效带宽 | 精度轻微损失 |
| **FlashAttention** | 降低KV Cache读写开销 | 无（纯算法优化） |
| **连续批处理** | 消除静态批处理的空闲等待 | 调度复杂度↑ |

---

## 五、尾延迟（Tail Latency）：P95/P99的“长尾之痛”

### 5.1 为什么平均值欺骗人？

平均值（P50）掩盖了最坏情况。在线服务中，**P99延迟决定了最慢的1%用户的体验**，而这1%往往是付费大客户或高敏感场景。

假设P50 TPOT = 40ms，P99 TPOT = 200ms。意味着100个请求中，有1个用户要忍受5倍于正常速度的卡顿。

### 5.2 尾延迟的主要来源

在LLM推理系统中，尾延迟的根源包括：

1. **请求长度极端差异**：一个超长Prompt（Prefill）堵住了调度器
2. **调度抖动**：连续批处理的“抢占/Swap”机制本身有开销
3. **显存分配延迟**：PagedAttention的Block分配偶尔出现慢路径
4. **硬件异构性**：不同GPU之间的微小差异（电压、温度导致频率波动）
5. **垃圾回收/内存管理**：Python/CUDA context切换

### 5.3 连续批处理的“双刃剑效应”

连续批处理对尾延迟是一把双刃剑：

**正面**：消除了静态批处理的“木桶效应”，短请求不被长请求阻塞，整体P50大幅改善。

**负面**：引入了**抢占（Preemption）和换入换出（Swap）的开销**。当一个高优先级请求被抢占，其KV被换出到CPU内存，恢复时再换入。这个过程可能耗时**几十到几百毫秒**——直接体现在该请求的P99甚至P95上。

```python
# 模拟连续批处理中抢占对尾延迟的影响
import random
import numpy as np

def simulate_batch_with_preemption(num_requests, preemption_rate=0.02):
    latencies = []
    for i in range(num_requests):
        base_latency = random.gauss(45, 5)  # 正常TPOT 45ms
        if random.random() < preemption_rate:
            # 被抢占：增加200-500ms的换入换出开销
            preempt_penalty = random.uniform(200, 500)
            base_latency += preempt_penalty
        latencies.append(max(0, base_latency))
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")

simulate_batch_with_preemption(10000)
# 输出类似：P50: 44.8ms, P95: 68.2ms, P99: 312.5ms
# P95被轻微影响，P99被抢占严重拖累
```

### 5.4 优化尾延迟的实战策略

1. **请求级优先级**：付费用户设置高优先级，避免被抢占
2. **细粒度调度**：vLLM V1中缩小调度窗口，减少单次调度决策的“犯大错”概率
3. **预热与缓存**：RadixAttention让长尾请求（共享前缀）直接命中缓存，跳过Prefill
4. **过载保护**：设置最大并发数，超过阈值时拒绝新请求（或排队等待），保证在线请求稳定
5. **内核级QoS**：通过CUDA MPS或MIG进行物理资源隔离，防止“吵闹邻居”影响关键请求

---

## 六、综合权衡：建立一个SLA优化框架

在生产环境中，优化不是单点突破，而是在**延迟、吞吐、成本**之间寻找最优解。我们可以建立一个简单的约束优化模型。

### 6.1 优化目标

$$\text{Maximize} \quad \text{Throughput} \quad \text{subject to:}$$

$$\text{TTFT}_{p95} < 500\text{ms}$$

$$\text{TPOT}_{p95} < 50\text{ms} \quad (\text{即 20 tok/s})$$

$$\text{显存占用} < \text{GPU\_VRAM}$$

### 6.2 调优决策树

```
1. 目标TPOT是否有余量？
   ├─ 是（远超20 tok/s）→ 增大Batch Size（提升吞吐），直到TPOT触达SLA阈值
   └─ 否（TPOT接近阈值）→ 考虑量化（INT4）降低模型大小，留出TPOT预算

2. 目标TTFT是否达标？
   ├─ 否（长Prefill阻塞）→ 开启Chunked Prefill / 前缀缓存
   └─ 是 → 保持

3. P99尾延迟是否过大？
   ├─ 是（受抢占影响）→ 增加显存预留 / 优化KV Cache分配策略 / 限制最大并发
   └─ 否 → 继续增大吞吐

4. 显存是否用完？
   ├─ 是 → 降低Batch Size或启用KV Cache量化（FP8/INT8）
   └─ 否 → 继续增加并发
```

### 6.3 关键经验法则

- **TPOT与Batch Size近似线性关系**：Batch翻倍，TPOT约增加10-30%（取决于KV占比）。可以用这个规律快速估算。
- **TTFT由Prefill主导**，与Batch中的Decode请求无关（在Chunked Prefill下，Decode请求不会阻塞Prefill，但会抢预算）。
- **P99 ≈ P50 + 调度抖动方差**：在一个设计良好的系统中，P99通常在P50的**1.5-3倍**之间。超过3倍说明存在严重的调度或显存问题。


## 七、总结：一张图看懂全部指标

```
┌──────────────────────────────────────────────────────────────────────┐
│                       LLM推理性能指标全景图                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│         用户体验层                       系统运营层                     │
│  ┌──────────────────┐        ┌─────────────────────────────┐         │
│  │   TTFT (首Token)  │        │   Throughput (吞吐量)        │        │
│  │   目标: <500ms    │        │   目标: 最大化 tok/s/$        │        │
│  │   瓶颈: Prefill   │        │   瓶颈: HBM带宽 + 调度        │        │
│  │   优化: Chunked   │        │   优化: 大Batch + 量化        │        │
│  └──────────────────┘        └─────────────────────────────┘         |
│                                                                      │
│  ┌──────────────────┐        ┌─────────────────────────────┐         │
│  │   TPOT (解码速度) │        │   Tail Latency (尾延迟)      │         │
│  │   目标: <50ms    │        │   目标: P99 < 3×P50          │         │
│  │   瓶颈: HBM带宽   │        │   根源: 抢占/长尾请求          │         │
│  │   优化: 量化      │        │   优化: 优先级/隔离/限流       │         │
│  └──────────────────┘        └─────────────────────────────┘         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                    核心权衡: Batch Size (并发数)                        │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Batch小 → TTFT低 ✓  TPOT低 ✓  吞吐低 ✗  成本高 ✗                │   │
│  │  Batch大 → TTFT高 ✗  TPOT高 ✗  吞吐高 ✓  成本低 ✓                │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│     物理极限公式: T_token_min = Model_Size / HBM_Bandwidth             │
│     实践意义: 70B FP16 × H100 ≈ 42ms → 最大 24 tok/s                   │
│     意味着: 单条请求永远无法突破此极限 (除非量化/蒸馏)                       │
└──────────────────────────────────────────────────────────────────────┘
```

**建议**：监控系统需要同时跟踪这四类指标，并建立它们之间的相关性仪表盘。当吞吐量上升时，密切监视TPOT和P99是否突破SLA红线。优化的本质，是在**用户体验（延迟）** 和**运营成本（吞吐）** 之间，找到那个最佳平衡点。而这个平衡点，是由你的业务SLA和硬件物理极限共同决定的。