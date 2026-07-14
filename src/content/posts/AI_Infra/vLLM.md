---
title: vLLM：大模型推理系统的分页内存革命
published: 2026-03-02
description: 系统梳理vLLM大模型推理系统的核心技术原理，解析PagedAttention的分页内存管理、连续批处理与分块预填充等创新机制，揭示其如何通过系统架构革新实现2-24倍的吞吐量提升。
cover: "/assets/images/posts/vLLM.png"
coverInContent: false
tags: [vLLM, 推理增强, PagedAttention, KV Cache]
category: AI_Infra
draft: false
---

# vLLM：大模型推理系统的分页内存革命

> [!note]
> 
> [vllm.ai](https://vllm.ai/)
>
> [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
>
> [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)

## 一、背景：大模型推理的“内存之痛”

### 1.1 自回归生成与KV Cache

大语言模型（LLM）采用自回归（autoregressive）方式生成文本：模型基于输入的prompt逐步生成下一个token，每一步都依赖之前所有已生成的token。在Transformer的self-attention机制中，每个token需要计算Query（Q）、Key（K）和Value（V）三个向量。每生成一个新token，模型需要将当前token的Query与之前所有token的Key做点积计算注意力分数，再对Value做加权求和。

为了避免每次生成时重复计算历史token的Key和Value，推理系统会将它们缓存下来，这就是**KV Cache**（Key-Value缓存）。KV Cache极大地节省了重复计算，但也带来了一个严重的问题：**内存消耗巨大，且动态增长**。

### 1.2 传统内存管理方式的三大困境

在高并发场景下，传统的KV Cache管理方式面临三大困境：

**困境一：显存碎片化严重。** 每个请求的KV Cache大小取决于其序列长度，而不同请求的长度差异巨大。传统系统为每个请求分配一段连续的内存空间，随着请求的到达和完成，内存中会留下大量不规则的碎片。这些碎片无法被有效利用，导致显存利用率往往只有60%左右。

**困境二：冗余存储浪费巨大。** 在并行采样（parallel sampling）和集束搜索（beam search）等解码算法中，多个生成序列共享相同的前缀。传统系统会为每个序列独立存储完整的KV Cache，造成了大量的冗余重复。

**困境三：静态批处理效率低下。** 传统的静态批处理（static batching）要求批次中所有请求同时开始、同时结束。如果一个请求早早完成，其占用的GPU计算资源只能空闲等待批次中其他请求全部完成。

这些问题直接限制了可同时处理的请求数量（batch size），严重拉低了系统的整体吞吐量。

### 1.3 为什么传统方案解决不了？

在vLLM出现之前，业界主要依赖两种思路：一是扩大GPU显存（成本高昂），二是优化算子实现（治标不治本）。问题的根源在于**内存管理的系统架构层面**，而非单个算子的效率。正如vLLM的核心哲学所揭示的：“不动模型结构，只动系统架构。”它不追求修改Attention机制或压缩模型权重，而是从操作系统设计中寻找灵感。

## 二、PagedAttention：从操作系统中借来的“救命稻草”

### 2.1 核心思想：虚拟内存分页

vLLM的核心创新**PagedAttention**，其灵感直接来源于操作系统中经典的**虚拟内存分页机制（Virtual Memory & Paging）**。

在操作系统中，虚拟内存技术将进程的地址空间划分为固定大小的“页”（page），而物理内存也划分为同样大小的“页框”（page frame）。通过页表（page table）将逻辑页映射到物理页框，使得进程的地址空间可以**非连续地**分布在物理内存中，从而消除了外部碎片，实现了按需分配。

PagedAttention将这一思想完美地移植到了LLM推理的KV Cache管理中：

- 将每个请求的KV Cache划分为固定大小的**KV块（KV Block）**
- 每个块包含固定数量token的Key和Value张量
- 通过**块表（Block Table）** 将逻辑块映射到物理GPU显存中非连续的块
- 仅在需要时按需分配物理块，而非一次性预留整个序列的空间

### 2.2 PagedAttention的详细工作原理

**（1）块级存储与物理内存池**

vLLM在GPU显存中预先分配一个物理块池（physical block pool），所有块的大小固定（例如每个块存储16个token的KV数据）。BlockSpaceManager（块空间管理器）负责管理这些物理块的分配、回收和复用。

**（2）逻辑块到物理块的映射**

每个请求拥有一个逻辑块列表，通过块表（page table）映射到实际的物理块地址。这种设计使得一个请求的KV Cache可以分散存储在显存的任意位置，完全消除了连续内存分配带来的碎片问题。

**（3）PagedAttention Kernel的执行流程**

在执行注意力计算时，vLLM的PagedAttention kernel接收query指针（q）和KV缓存指针（k_cache、v_cache），这些指针指向分页存储的KV数据。kernel通过块表查找每个逻辑块对应的物理地址，从全局内存中读取数据。这种非连续访问模式对GPU kernel的实现提出了挑战——vLLM为此专门设计了多头查询注意力内核（`csrc/attention/attention_kernels.cu`），通过特定的内存布局和访问方法来保证高性能。

**（4）块级共享与Copy-on-Write**

PagedAttention的另一大创新是**块级共享**（block-level sharing）。当多个请求（或同一请求的多个生成序列）共享相同的前缀时，它们可以指向**同一组物理KV块**，无需复制数据。

当某个序列需要修改共享块中的内容时（例如在beam search中不同序列开始分叉），vLLM采用**Copy-on-Write（写时复制）** 策略：
- 系统维护每个物理块的**引用计数器**（reference count）
- 当需要修改一个共享块时，只有在该时刻才创建该块的副本
- 原块继续被其他序列共享，新序列使用副本进行修改

这种机制可以**减少30%-50%的显存占用**，尤其适用于并行采样和集束搜索等场景。

**（5）自动前缀缓存（Automatic Prefix Caching）**

vLLM将所有KV块存储在一个**哈希表**中。每个块根据其包含的token以及该块之前前缀中的token计算哈希值。当新请求到达时，系统可以快速查找是否已有相同前缀的KV块被缓存，如果有则直接复用。这避免了重复计算相同前缀的attention，大幅节省了计算资源和时间。

### 2.3 PagedAttention的效果数据

论文实验表明，PagedAttention使得vLLM实现了：
- **KV Cache内存近乎零浪费**（near-zero waste）
- 与FasterTransformer和Orca等最先进系统相比，**吞吐量提升2-4倍**
- 与HuggingFace Transformers相比，**吞吐量最高可达24倍**
- 与HuggingFace TGI相比，**吞吐量最高可达3.5倍**
- 在长序列、大模型和复杂解码算法上，提升效果更加显著

## 三、连续批处理（Continuous Batching）：让GPU永不闲置

### 3.1 静态批处理的困境

传统的静态批处理将多个请求打包成一个固定批次，一次性提交给GPU处理。问题是：批次中不同请求的生成长度不同，短请求完成后必须等待长请求，导致GPU计算单元空闲。

### 3.2 连续批处理的革命

vLLM引入了**连续批处理（Continuous Batching）** ——也称为**动态批处理**或**持续批处理**。其核心思想极其简单却威力巨大：**调度器在每个解码迭代（decode iteration）都做出调度决策，而非每个批次只做一次**。

具体工作方式如下：
1. 当一个请求完成生成时，其占用的“槽位”**立即**被队列中的下一个等待请求填充
2. 新加入的请求可以是处于prefill阶段（首次处理prompt），也可以是处于decode阶段（继续生成）
3. 调度器在每个迭代步都重新评估所有请求的状态，动态决定本轮处理哪些请求、处理多少token

这种设计使GPU计算单元始终保持满载状态，**GPU利用率相比静态批处理提升3-5倍**。

### 3.3 调度器的三层队列架构

vLLM的调度器维护三个核心队列：
- **Waiting队列**：等待调度的新请求
- **Running队列**：当前正在执行的请求
- **Swapped队列**：因显存不足被暂时交换到CPU内存的请求

调度器每轮根据当前GPU显存资源、token预算和请求状态，决定哪些请求可以从Waiting进入Running，哪些Running请求需要继续执行，哪些需要被Swap出去。

## 四、分块预填充（Chunked Prefill）：让长序列不再“霸占”GPU

### 4.1 问题的提出

在vLLM的早期版本中，一个长prompt的prefill阶段会一次性处理完所有输入token。如果prompt非常长（例如数万个token），这会长时间独占GPU，导致其他decode请求被“饿死”（starvation），显著影响系统的响应延迟（TTFT，Time To First Token）。

### 4.2 Chunked Prefill的解决方案

**分块预填充（Chunked Prefill）** 将长prompt切分成多个较小的chunk，每个chunk单独进行prefill计算。关键特性包括：

- 每个chunk的大小受`max_num_batched_tokens`参数限制
- 各个chunk之间**存在依赖关系**，必须顺序处理——后一个chunk的prefill依赖前一个chunk的KV Cache结果
- 每个chunk的attention mask会根据其在整体prompt中的位置动态设置
- 最重要的是：**prefill chunk可以和decode请求混合批处理**

这意味着GPU不再被一个长prefill请求独占——它可以在处理长prompt的某个chunk的同时，也处理其他请求的decode步骤。这既保证了长序列请求的推进，又避免了decode请求的饥饿。

## 五、vLLM的系统架构全景

### 5.1 核心组件

vLLM采用分层架构设计，核心是`LLMEngine`类：

**（1）输入处理层（Input Processing）** ：负责使用指定的tokenizer对输入文本进行tokenization。

**（2）调度层（Scheduler Layer）** ：决定每个迭代步处理哪些请求。这是vLLM的大脑，集成了连续批处理、分块预填充等调度策略。

**（3）执行层（Model Execution）** ：管理模型的实际执行，支持跨多GPU的分布式执行。

**（4）输出处理层（Output Processing）** ：将模型生成的token ID解码为人类可读的文本。

`AsyncLLMEngine`则在此基础上增加了异步请求处理能力，支持在线服务场景。

### 5.2 vLLM V1架构升级

vLLM社区在2025年推出了**V1引擎**，对核心架构进行了全面重组。V1采用多进程架构分离关注点，核心改进包括：
- 更简单的模块化代码库
- 近乎零的CPU开销
- 默认启用各项优化，零配置即可获得高性能
- CPU密集型任务（如tokenization、多模态输入处理）与核心执行循环实现更大程度的重叠

### 5.3 分布式支持

vLLM支持完整的分布式部署方案：
- 张量并行（Tensor Parallelism）：将单个模型层切分到多GPU
- 流水线并行（Pipeline Parallelism）：将模型的不同层分配到不同GPU
- 支持通过Kubernetes实现跨节点资源调度，单集群可扩展至100+节点

## 六、vLLM的适用场景与优缺点

### 6.1 适用场景

- **高并发在线推理服务**：需要同时处理大量用户请求的场景，如聊天机器人、AI助手
- **长序列处理**：处理长文档摘要、代码生成等需要大上下文的任务
- **离线批量推理**：大规模数据处理、模型评估等批量推理任务
- **多租户部署**：需要高效共享GPU资源的场景

### 6.2 核心优势

1. **极高的显存利用率**：通过PagedAttention实现85%以上的显存利用率
2. **卓越的吞吐量**：相比传统系统提升2-24倍
3. **灵活的批处理**：连续批处理让GPU永不闲置
4. **强大的兼容性**：支持超过50种主流开源模型
5. **完善的生态**：提供OpenAI兼容API、丰富的文档和活跃的社区
6. **持续演进**：V1架构的推出标志着框架的持续成熟

### 6.3 局限性

1. **首次Token延迟（TTFT）可能较高**：在高并发下，新请求可能需要等待调度
2. **对硬件有一定要求**：需要较新的NVIDIA GPU和CUDA环境
3. **长序列prefill仍有优化空间**：虽然chunked prefill缓解了问题，但极端长序列仍可能影响其他请求
4. **学习曲线**：虽然易用性很好，但深入理解和调优仍需要一定的系统知识

## 七、总结：vLLM的历史意义

vLLM的出现标志着大模型推理系统设计的一次**范式转移**。它证明了：**系统架构的创新**——而非仅仅是模型或算子的优化——可以在不改变模型结构的前提下，带来数量级的性能提升。

PagedAttention将操作系统的内存管理思想引入深度学习推理，这一跨界借鉴展示了系统设计中的通用智慧。连续批处理则彻底改变了人们对“批处理”的认知——从“静态批次”到“动态流式”，让GPU从一个“等所有人都到齐才开饭”的低效食堂，变成了“随到随吃、吃完即走”的高效流水线。

正如vLLM论文中所说：“我们的评估表明，vLLM在相同延迟水平下，将主流LLM的吞吐量提升了2-4倍，且这种提升在长序列、大模型和更复杂的解码算法上更为显著。”这一成就不仅来自算法创新，更来自对系统本质的深刻理解——**在大模型推理中，内存管理就是性能，而性能就是成本**。