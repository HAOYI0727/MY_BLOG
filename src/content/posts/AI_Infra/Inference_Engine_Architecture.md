---
title: PagedAttention & RadixAttention —— LLM推理引擎架构
published: 2026-04-08
description: 系统讲解大模型推理引擎的两大核心架构：vLLM的PagedAttention与SGLang的RadixAttention。从KV Cache的显存占用公式出发，剖析PagedAttention如何借鉴OS虚拟内存分页思想将KV Cache分块存储以消除显存碎片；深入解读RadixAttention以基数树实现token级最长前缀匹配，支持跨请求共享System Prompt/Few-shot，缓存命中率比哈希方案提升5倍。通过设计哲学对比与选型建议，揭示“通用吞吐 vs 结构化复用”的核心权衡。
cover: "/assets/images/posts/inference_engine_architecture.png"
coverInContent: false
tags: [推理引擎, vLLM, PagedAttention, SGLang, RadixAttention, KV Cache, 前缀缓存]
category: AI_Infra
draft: false
---

# PagedAttention & RadixAttention —— LLM推理引擎架构

## 引言：推理的“内存墙”

大模型推理时，模型权重占一份显存，KV Cache占另一份——而且后者**随着序列长度和并发数动态增长**。当你在100个并发请求上跑Llama 3 70B（隐藏层8192），KV Cache的显存占用会迅速变成一个天文数字。

传统方案为每个请求分配**连续**的显存空间来存放KV Cache，这带来了三个问题：

1. **显存碎片化**：序列长度动态变化，分配和释放不同大小的连续内存块会产生大量“内存空洞”
2. **冗余计算**：多轮对话或共享相同System Prompt的请求，KV被反复计算
3. **内部碎片**：如果按最大长度预分配，实际使用率可能低于40%

vLLM和SGLang从两个不同维度正面攻击了这些问题。本文深入拆解它们的核心机制。

---

## 一、KV Cache：LLM推理的“记忆体”

### 1.1 为什么需要KV Cache？

Transformer的自回归生成中，生成第 $t$ 个token时，需要计算第 $t$ 个token与**前面所有 $t-1$ 个token**的注意力分数。如果不做缓存，每次生成都要重新计算所有历史token的Key和Value——复杂度从 $O(n)$ 变成 $O(n^2)$。

KV Cache的本质是**用显存换时间**：把历史token的Key和Value存下来，后续生成直接复用。

### 1.2 显存占用公式

对于一个Transformer模型，**单个token**的KV Cache显存占用为：

$$\text{KV\_per\_token} = 2 \times L \times H \times D \times \text{dtype\_size}$$

其中：
- $L$ = 层数（num_layers）
- $H$ = 注意力头数（num_heads）
- $D$ = 每个头的维度（head_dim）
- $\text{dtype\_size}$ = 数据类型字节数（FP16为2字节）

以LLaMA-7B为例（$L=32, H=32, D=128$，FP16）：
- 单token KV Cache = $2 \times 32 \times 32 \times 128 \times 2 = 524,288$ 字节 $\approx$ **512 KB**

这意味着：一个 **4096 token** 的请求，仅KV Cache就占用 **2 GB** 显存。100个并发请求就是200GB——远超单卡容量。

---

## 二、vLLM的PagedAttention：操作系统级的内存管理

### 2.1 核心思想：分页

PagedAttention的灵感直接来自操作系统的**虚拟内存分页**机制。

传统方案要求每个请求的KV Cache在物理显存中**连续存放**。vLLM的做法是：

> 将KV Cache划分为固定大小的**块（Block）** ，每个块存储固定数量token（如16或32个）的KV数据。这些块可以存储在**非连续的物理内存**中，通过**块表（Block Table）** 维护逻辑位置到物理位置的映射。

```
传统连续分配：
[ReqA: #####][空洞..][ReqB: ###][空洞....][ReqC: ######]  ← 利用率 < 50%

PagedAttention分页管理：
[块1: A][块2: A][块3: B][块4: C][块5: C]  ← 利用率 > 90%
```

**物理非连续**意味着：分配时不需要找一大块连续空间，只要有空闲块就能用。释放时直接把块放回内存池，没有碎片。

### 2.2 显存占用的精确计算

PagedAttention不改变“单token KV成本”，改变的是**分配方式**。单个Block的显存占用为：

$$\text{KV\_per\_block} = \text{block\_size} \times 2 \times L \times H \times D \times \text{dtype\_size}$$

LLaMA-7B，`block_size=16`时：
- 单Block = $16 \times 512\text{KB} \approx$ **8 MB**

vLLM中每个Block包含`BLOCK_SIZE`个token的KV数据，按head独立存储。

### 2.3 Block Size的权衡

Block Size是vLLM的核心配置参数，默认16。选择多大？

| Block Size | 优点 | 缺点 |
|-----------|------|------|
| **16（默认）** | 粒度细，内存利用率高（~50%） | 元数据开销稍大 |
| **32-64** | 减少元数据，长序列吞吐更高 | 内部碎片增加（~70%内存占用） |
| **128** | 极致吞吐 | 内存碎片明显增加 |

**选择原则**：
- 短序列多 → 小Block（16），减少浪费
- 长序列为主（500+ tokens）→ 大Block（32-64），减少元数据开销
- FlashAttention要求Block Size为16的倍数

### 2.4 前缀缓存：vLLM的“有限共享”

vLLM也支持**自动前缀缓存（Automatic Prefix Caching, APC）** ，但有几个限制：

- **默认关闭**（v0.9版本需手动`--enable-prefix-caching`开启）
- **Block粒度匹配**：缓存命中按完整的16-token Block进行哈希匹配
- **对齐问题**：如果共享前缀不是Block Size的整数倍，剩余token无法被缓存

例如：System Prompt有100个token，Block Size=16。前96个token（6个完整Block）可缓存，剩余4个token无法复用。

**问题本质**：前缀的语义边界（一句话、一个指令）跟Block的物理边界没有对应关系。

---

## 三、SGLang的RadixAttention：前缀树级别的共享

### 3.1 核心洞察：请求间共享前缀

实际生产环境中，大量请求共享相同的前缀：

- **System Prompt**：每个请求都带相同的角色设定
- **Few-shot示例**：所有请求共用相同的示例
- **多轮对话历史**：当前轮次包含之前所有对话
- **Agent工具描述**：工具定义被反复使用

vLLM对每个请求**从头计算**这些共享前缀的KV Cache，浪费大量时间和显存。

### 3.2 基数树（Radix Tree）：数据结构

RadixAttention的核心数据结构是**基数树（Radix Tree）** ——前缀树（Trie）的压缩版本。标准Trie每个节点存一个token，基数树允许一个节点存储**多个token组成的序列**，减少树的深度。

SGLang维护一棵**全局的Radix Tree**：

- **每个节点**存储一段连续token序列的KV Cache物理指针
- **从根到叶子的路径**代表一个完整请求的token序列
- **共享前缀**复用同一个节点，无需重复存储和计算

```
Radix Tree示例（三个请求共享System Prompt）：

Root
└── "System: You are helpful\nUser: What's "
    ├── "AI?" → [KV cache for request 1]
    ├── "ML?" → [KV cache for request 2]
    └── "DL?" → [KV cache for request 3]

共享前缀 "System: You are helpful\nUser: What's " 
→ 计算一次，复用3次 → 5倍加速！
```

### 3.3 最长前缀匹配（Longest Prefix Match）

新请求到达时，SGLang执行**最长前缀匹配**：

1. 从Root出发，用请求的token序列沿树向下匹配
2. 找到匹配的最长路径
3. **匹配到的部分**：直接复用KV Cache（零计算）
4. **未匹配的尾部**：计算新的KV，作为新分支挂在树上

匹配精度是 **token级别**，不受Block Size约束。

### 3.4 LRU淘汰策略

显存有限，Radix Tree需要淘汰策略。SGLang采用 **LRU（Least Recently Used）** ：

- 每个节点维护 `ref_count`（引用计数）和最后访问时间
- 淘汰时从 **叶子节点开始剪枝**（leaf-first）
- 优先保留**根节点附近**的公共前缀（如System Prompt）
- 只有 `ref_count = 0` 的节点才能被淘汰

```
淘汰前（显存满）：
Root
├── "System A" (5分钟前访问)
│   ├── "Task 1" (1分钟前) ← 保留（最近）
│   └── "Task 2" (30分钟前) ← 淘汰（旧 + 叶子）
└── "System B" (60分钟前) ← 淘汰（非常旧）

淘汰后：
Root
└── "System A"
    └── "Task 1"
```

### 3.5 HiCache：三级缓存扩展

RadixAttention的扩展版本**HiCache**将缓存从GPU内存扩展到三级层次：

| 层级 | 存储位置 | 容量 | 延迟 |
|------|---------|------|------|
| L1 | GPU显存 | 小 | 最低 |
| L2 | 主机内存 | 中 | 中等 |
| L3 | 分布式存储 | 大 | 较高 |

HiRadixTree记录每个节点的KV Cache存储在哪个层级，实现**显存不够时自动降级到内存或远端存储**。

---

## 四、设计哲学对比

### 4.1 核心差异

| 维度 | vLLM PagedAttention | SGLang RadixAttention |
|------|-------------------|----------------------|
| **管理单元** | 固定大小Block（16/32 tokens） | 变长Token序列（节点） |
| **复用粒度** | Block级别（完整Block哈希匹配） | Token级别（最长前缀匹配） |
| **跨请求共享** | 有限（需Block对齐） | 原生支持（全局Radix Tree） |
| **默认开启** | 前缀缓存默认关闭 | RadixAttention默认开启 |
| **实现语言** | C++ CUDA扩展 | Python路由（有GIL瓶颈） |
| **设计哲学** | 操作系统内核 → 通用资源调度 | 智能运行时 → 结构化复用 |

### 4.2 性能特征

**vLLM的优势场景**：
- 独立文本生成（请求间无共享前缀）
- 高并发场景（C++实现，无GIL瓶颈）
- 追求通用吞吐量

实测数据（150并发，L4 GPU）：
- vLLM吞吐：**363.76 req/s**，延迟0.414s
- SGLang吞吐：**150.00 req/s**，延迟1.013s

**SGLang的优势场景**：
- 前缀重叠率 > 60%
- 多轮对话、Agent、RAG（大量共享System Prompt/Few-shot）
- 结构化生成（约束解码集成在调度层）

RadixAttention的缓存命中率可比哈希方案**提升5倍**。对于没有前缀重叠的请求，RadixAttention几乎没有额外开销，性能与vLLM相当。

### 4.3 选型建议

```
你的场景有大量共享前缀吗？
├─ 是（多轮对话/Agent/RAG/大批量Few-shot）
│   └─ 优先考虑 SGLang（RadixAttention token级共享）
│       └─ 注意：高并发下Python GIL可能成为瓶颈
│
└─ 否（独立文本生成/请求间无关联）
    └─ 优先考虑 vLLM（PagedAttention通用吞吐）
        └─ 如需要前缀缓存，手动开启 --enable-prefix-caching
```

---

## 五、总结：一张图看懂

```
┌────────────────────────────────────────────────────────────────┐
│                     问题：KV Cache显存瓶颈                        │
│  单token 512KB（LLaMA-7B）→ 4096 token = 2GB → 100并发 = 200GB   │
└────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────────┐              ┌─────────────────────┐
│  vLLM PagedAttention │              │SGLang RadixAttention│
├──────────────────────┤              ├─────────────────────┤
│ 灵感：OS虚拟内存分页    │              │ 灵感：文件系统缓存     │
│ 管理单元：固定Block    │              │ 管理单元：变长节点     │
│ 复用粒度：Block级      │              │ 复用粒度：Token级     │
│ 跨请求共享：有限       │              │ 跨请求共享：原生       │
│ 实现：C++ CUDA       │               │ 实现：Python + C++   │
├─────────────────────┤               ├─────────────────────┤
│ 适用：通用高吞吐场景    │              │ 适用：前缀共享密集型    │
│ 优势：无GIL瓶颈       │               │ 优势：Token级复用     │
└─────────────────────┘               └─────────────────────┘
```

**核心权衡**：**通用性 vs 结构化复用**。

vLLM像操作系统内核——精于通用内存管理，对所有请求一视同仁。SGLang像智能运行时——专攻结构化场景，在共享前缀多的任务中性能飞跃。

没有绝对的“更好”，只有“更适合”。理解两者的设计哲学和适用边界，才能在具体场景中做出正确的架构选择。