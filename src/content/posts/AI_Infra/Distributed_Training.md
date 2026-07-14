---
title: NCCL & DP/MP/PP/FSDP/ZeRO —— 分布式训练
published: 2026-04-06
description: 系统讲解分布式训练的核心技术栈：从硬件互连出发，剖析NCCL集合通信库的原语及Ring/Tree算法的自动选择逻辑；深入对比数据并行（DP）、张量并行（TP）、流水线并行（PP）三种策略的通信频率与适用场景；详解ZeRO三个阶段的显存节省与通信开销权衡，以及FSDP作为PyTorch原生ZeRO-3实现的训练流程。
cover: "/assets/images/posts/distributed_training.png"
coverInContent: false
tags: [分布式训练, NCCL, 数据并行, 张量并行, 流水线并行, ZeRO]
category: AI_Infra
draft: false
---

# NCCL & DP/MP/PP/FSDP/ZeRO —— 分布式训练

## 引言：分布式训练的“通信税”

单卡放不下大模型——这是今天AI Infra面对的基本事实。一个70B参数的模型，仅参数本身就需要约140GB（FP16），加上梯度、优化器状态（Adam需要额外2倍参数量的动量），单卡显存需求直逼500GB+。而H100的显存只有80GB。

解决方案只有一个：**把模型和数据分布到多张GPU上**。

但分布式训练有一个隐形成本——**通信**。GPU之间传递数据的时间，可能占到总训练时间的30%以上。当你在3072张GPU上训练万亿参数模型时，数据并行的通信带宽需求高达**13TB/s**。

所以，理解分布式训练，本质上要理解两件事：
1. **通信怎么走**——硬件互连（NVLink/PCIe）和通信原语（NCCL All-Reduce等）
2. **模型怎么切**——数据并行、模型并行、流水线并行、ZeRO/FSDP

本文从硬件到软件，逐层拆解。

---

## 一、硬件互连：NVLink vs PCIe 的带宽鸿沟

### 1.1 数字对比

GPU之间通信，走什么路？两条路：

| 互连方式 | 带宽 | 延迟 | 路径 |
|---------|------|------|------|
| PCIe 5.0 x16 | 128 GB/s（双向） | 较高 | 经CPU/PCIe Switch |
| NVLink 4.0（H100） | 900 GB/s（8卡聚合） | 亚微秒 | GPU直连 |

NVLink 4.0的总带宽是PCIe 5.0的**7倍以上**。在DGX H100的8卡全互联拓扑中，总带宽可达**7.2 TB/s**。到了NVLink 5.0（B300），这个数字飙升至**1.8 TB/s**。

### 1.2 为什么差距这么大？

PCIe是**通用总线**，设计目标是连接CPU与各种外设（GPU、网卡、SSD等）。数据从GPU A到GPU B的路径是：

```
GPU A → PCIe Switch → CPU内存（DMA）→ PCIe Switch → GPU B
```

每一步都有开销，CPU还要参与地址转换和中断处理。

NVLink是**GPU专用直连总线**，设计目标只有一个：**让GPU之间直接通信，不需要CPU当中间人**。NVLink支持**统一内存访问**——GPU可以直接读写对方显存，无需CPU介入。

### 1.3 实践意义

- **NVLink适合节点内**（同一台机器内的8张卡）：张量并行（TP）强烈依赖高频通信，应放在NVLink域内
- **PCIe适合节点间**（跨机器）：数据并行（DP）每个step只同步一次梯度，对带宽不那么敏感

---

## 二、GPUDirect RDMA：绕过CPU的“高速公路”

即使走PCIe，传统路径也有问题：**数据必须经过CPU内存**。

```
传统路径：GPU VRAM → CPU DRAM → 网卡 → 网络 → 网卡 → CPU DRAM → GPU VRAM
```

CPU成了瓶颈——不仅要搬运数据，还要处理中断、地址映射。

**GPUDirect RDMA**彻底改变了这一点：

```
GPUDirect路径：GPU VRAM → 网卡（RDMA）→ 网络 → 网卡（RDMA）→ GPU VRAM
```

网卡（如NVIDIA ConnectX SmartNIC或BlueField DPU）可以直接读写GPU显存，**完全绕过CPU和系统内存**。

效果：带宽提升**2-8倍**，延迟降低**10倍以上**。CPU被解放出来做真正的计算，而不是当搬运工。

任何使用以太网、InfiniBand或RoCE的网络框架都可以启用GPUDirect RDMA。

---

## 三、NCCL：集合通信的“加速引擎”

有了硬件（NVLink、GPUDirect RDMA），还需要软件来使用它们。**NCCL（NVIDIA Collective Communications Library）** 就是这个软件层。

NCCL是一个**拓扑感知**的多GPU集合通信库——它知道GPU之间是NVLink还是PCIe连接，知道哪些GPU在同一个NUMA节点，然后选择最优的通信路径。

### 3.1 核心通信原语

**All-Reduce**：分布式训练中最核心的原语。将N个GPU上的数据进行归约（求和、取最大等），然后把结果**广播到所有GPU**。

数学上，对于K个rank，每个rank提供数组 $V_k$，All-Reduce后每个rank都得到：

$$S[i] = V_0[i] + V_1[i] + \cdots + V_{K-1}[i]$$

**All-Gather**：从所有rank收集数据，每个rank得到完整的 $K \times N$ 结果。

**Reduce-Scatter**：归约后将结果**分散**到各个rank，每个rank只拿到自己那份。

**关键关系**：`Reduce-Scatter` + `All-Gather` = `All-Reduce`。

| 操作 | 方向 | 典型场景 |
|------|------|---------|
| Reduce-Scatter | 归约 + 分散 | ZeRO梯度分片 |
| All-Gather | 收集 + 广播 | ZeRO参数收集 |
| All-Reduce | 归约 + 全广播 | 数据并行梯度同步 |

### 3.2 Ring All-Reduce：带宽最优

NCCL根据数据量自动选择算法：

- **小数据（<32KB）** ：Tree算法，$O(\log_2 N)$ 步，延迟最优
- **中等数据（32KB-2MB）** ：Double Binary Tree，平衡延迟和带宽
- **大数据（>2MB）** ：Ring算法，带宽最高

Ring All-Reduce的原理：

1. **Reduce-Scatter阶段**：N个GPU构成逻辑环，沿环传递N-1轮，每轮每个GPU发送 $1/N$ 的数据并归约接收到的数据。完成后每个GPU持有 $1/N$ 的完整归约结果。

2. **All-Gather阶段**：再沿环传递N-1轮，每轮发送自己的归约结果。完成后所有GPU拥有完整结果。

**通信量**：每个GPU单向发送 $2 \times (N-1)/N \times \text{data\_size}$。8 GPU时约为 $1.75 \times \text{data\_size}$。带宽效率高，但延迟随N线性增长。

Tree All-Reduce只需 $O(\log_2 N)$ 步——8 GPU时3步完成，远快于Ring的14步。但带宽效率不如Ring。

### 3.3 NCCL代码示例

```python
import torch
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(backend='nccl')

# 假设每个rank有张量tensor
tensor = torch.randn(1024, 1024).cuda()

# All-Reduce：所有GPU求和，结果广播到所有GPU
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-Gather：收集所有GPU的数据
gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
dist.all_gather(gather_list, tensor)

# Reduce-Scatter：归约后分散
dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
```

---

## 四、三种并行策略：切数据 vs 切模型 vs 切层

### 4.1 数据并行（Data Parallelism, DP）

**原理**：每张GPU存一份完整模型，把数据切成多份，各卡处理自己的数据分片，梯度通过All-Reduce同步。

**通信量**：每个训练step一次All-Reduce，数据量为**模型参数量的2倍**（梯度大小=参数量）。

$$
\text{通信量} \approx 2 \times \text{参数量} \times (N-1)/N
$$

**优点**：实现简单，通信量不随模型深度增加。

**缺点**：每张卡都要存完整模型，无法突破单卡显存限制。

**适用场景**：模型能放进单卡，想提升吞吐量。

### 4.2 模型并行 / 张量并行（Model Parallelism / Tensor Parallelism, MP/TP）

**原理**：把**每一层的权重矩阵切分**到多张GPU上。以Transformer的Attention层为例，将 $[\text{hidden\_dim}, \text{num\_heads} \times \text{head\_dim}]$ 的权重矩阵沿列方向切开。

**通信量**：**非常高**——每层前向和反向都需要All-Reduce同步。

**优点**：突破单卡显存限制，适合超大模型。

**缺点**：通信频繁（每层都同步），**通信占比可达30%+**。

**适用场景**：**节点内**（NVLink高速互联），单卡放不下单层。

### 4.3 流水线并行（Pipeline Parallelism, PP）

**原理**：把模型的**层切分**到不同GPU——GPU 0负责前N层，GPU 1负责接下来的N层，依此类推。前向传播时数据从左向右流经各卡。

**通信量**：**相对较小**——只在相邻设备间传递**激活值**（前向）和**梯度**（反向），不是完整的梯度同步。

**优点**：通信量小，可跨节点扩展。

**缺点**：存在**流水线气泡**（bubble）——部分GPU空闲等待。

**适用场景**：**节点间**，模型极深（千亿参数以上）。

### 4.4 通信量对比总结

| 并行策略 | 通信频率 | 通信量 | 推荐位置 |
|---------|---------|--------|---------|
| 数据并行（DP） | 每step一次 | 2×参数量 | 节点间 |
| 张量并行（TP） | 每层多次 | 极高 | 节点内（NVLink） |
| 流水线并行（PP） | 每微批次 | 小（仅激活/梯度） | 节点间 |

**最佳实践**：**节点内做TP，节点间做DP**——TP走NVLink高速通道，DP走PCIe/网络。

---

## 五、ZeRO与FSDP：消除显存冗余

数据并行虽然简单，但有一个致命问题：**每张GPU都存了一份完整模型**。对于70B模型，这是浪费——8张卡存了8份参数，但每张卡只需要自己那部分数据。

**ZeRO（Zero Redundancy Optimizer）** 的核心思想：**把模型状态（参数、梯度、优化器状态）分片存储到所有GPU上，需要时通过通信获取**。

### 5.1 模型状态的构成

以Adam优化器为例，训练时每张GPU需要存储：

- **参数（Parameters）** ：$P$
- **梯度（Gradients）** ：$P$
- **优化器状态（Optimizer States）** ：$2P$（Adam的动量和方差）

总计：**$4P$**

### 5.2 ZeRO三个阶段

| 阶段 | 分片内容 | 单卡显存 | 通信开销 |
|------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | $P + P + 2P/N$ | 最低 |
| ZeRO-2 | 优化器状态 + 梯度 | $P + P/N + 2P/N$ | 中等 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | $P/N + P/N + 2P/N = 4P/N$ | 最高 |

**ZeRO-1**：将优化器状态（Adam动量）分片到N个GPU。单卡显存从 $4P$ 降至 $P + P + 2P/N \approx (2 + 2/N)P$。8卡时减少约62.5%。

**ZeRO-2**：进一步分片梯度。单卡显存降至 $P + P/N + 2P/N = (1 + 3/N)P$。8卡时减少约75%。

**ZeRO-3**：分片一切——参数、梯度、优化器状态全部打散。单卡显存降至 $4P/N$。8卡时减少约87.5%。

**代价**：每增加一个stage，通信开销就增加一层。ZeRO-3需要在前向传播前All-Gather参数，在反向传播后Reduce-Scatter梯度。

### 5.3 FSDP：PyTorch的ZeRO-3实现

**FSDP（Fully Sharded Data Parallel）** 是PyTorch对ZeRO-3的原生实现。它将所有模型状态（参数、梯度、优化器状态）分片到所有数据并行worker上，达到理论上的最小单卡显存。

FSDP的训练流程：

1. **前向传播前**：All-Gather当前层的参数分片 → 所有GPU获得完整参数
2. **前向计算**：用完整参数计算
3. **前向传播后**：释放非本地的参数分片（`reshard_after_forward=True`）
4. **反向传播**：类似地All-Gather参数，计算梯度
5. **梯度同步**：Reduce-Scatter梯度，每张卡只保留自己那部分

FSDP支持ZeRO-2和ZeRO-3两种模式。

---

## 六、混合并行：实战中的组合

实际训练大模型时，往往**混合使用多种并行策略**：

- **8卡节点内**：张量并行（TP）切分每一层，利用NVLink高速通信
- **节点间**：数据并行（DP）或FSDP，利用ZeRO减少显存冗余
- **极深模型**：叠加流水线并行（PP），进一步扩展

```
DP（节点间） × PP（节点间） × TP（节点内）
```

在3072张GPU上训练万亿模型时，数据并行通信带宽需求高达**13TB/s**，流水线并行需要**892GB/s**。

**选择原则**：
- 模型能放进单卡 → 纯DP（最简单）
- 单卡放不下但单层能放下 → ZeRO/FSDP
- 单层都放不下 → TP（节点内）+ PP（节点间）+ DP/FSDP

---

## 七、总结：一张图看懂分布式训练

```
┌───────────────────────────────────────────────────────────────┐
│                        硬件层                                  │
├───────────────────────────────────────────────────────────────┤
│  NVLink（900GB/s）← 节点内GPU直连    PCIe（128GB/s）← 通用互连    │
│  GPUDirect RDMA ← 绕过CPU，网卡直读GPU显存                       │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                        通信层（NCCL）                           │
├───────────────────────────────────────────────────────────────┤
│  All-Reduce（梯度同步）← Reduce-Scatter + All-Gather            │
│  算法自动选择：小数据→Tree（低延迟） 大数据→Ring（高带宽）            │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        并行策略层                                │
├────────────────────────────────────────────────────────────────┤
│  数据并行（DP）：每卡完整模型，切数据，通信少                         │
│  张量并行（TP）：切每一层，通信频繁，放节点内（NVLink）                │
│  流水线并行（PP）：切层，通信少，放节点间                            │
│  ZeRO/FSDP：分片存储模型状态，消除冗余，通信换显存                    │
└────────────────────────────────────────────────────────────────┘
```

**核心权衡**：**通信 vs 显存 vs 计算效率**。

- DP：通信少、显存浪费大
- TP：通信多、显存节省、适合节点内
- PP：通信少、有气泡浪费
- ZeRO/FSDP：用更多通信换更少显存