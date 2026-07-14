---
title: GPU & Grid-Block-Thread —— CUDA编程心智模型
published: 2026-04-02
description: 系统讲解CUDA编程的核心心智模型：从GPU物理架构出发，剖析软件层次Grid-Block-Thread到硬件SM-Warp-CUDA Core的完整映射关系，深入对比寄存器、共享内存与全局内存的延迟差异与作用域，并详解Kernel启动配置、同步/异步数据传输以及多Stream实现计算与传输重叠的工程实践。
cover: "/assets/images/posts/cuda_programming_model.png"
coverInContent: false
tags: [CUDA, GPU架构, SIMT, Warp, 共享内存]
category: AI_Infra
draft: false
---

# GPU & Grid-Block-Thread —— CUDA编程心智模型

## 引言：为什么要理解心智模型？

写CUDA程序，最难的不是语法，而是“想对”——你得在脑子里建立起一个精确的模型：代码写下去，硬件上到底发生了什么？

很多初学者把CUDA kernel当成一个“并行for循环”——`<<<grid, block>>>` 不过是告诉GPU要开多少线程。这种理解能跑通代码，但写不出高性能代码。真正的CUDA高手，脑子里同时运行着两套模型：**软件层面的Grid/Block/Thread层次结构**，和**硬件层面的SM/Warp/CUDA Core物理架构**。两套模型之间的映射关系，就是CUDA性能优化的全部秘密。

本文从硬件出发，自下而上地建立这个心智模型。

---

## 一、GPU的物理架构：SM是“心脏”

### 1.1 SM（流式多处理器）

NVIDIA GPU 架构围绕**可扩展的多线程流式多处理器（SM）阵列**构建。你可以把SM理解为GPU里的“计算核心”——但和CPU核心不同，一个GPU包含数十甚至上百个SM（H100有132个SM），每个SM又包含大量的计算单元。

一个典型的SM内部包含：

- **Warp Scheduler**：负责调度warps的执行
- **Dispatch Unit**：将指令分发到计算单元
- **CUDA Core**：执行标量整数和浮点运算
- **Tensor Core**：执行矩阵乘加运算（AI加速专用）
- **Register File**：寄存器堆，存储线程私有数据
- **Shared Memory / L1 Cache**：片上共享存储

以H100为例，每个SM被分成4个相同的子分区（subpartition），每个子分区包含一个Tensor Core、16K个32位寄存器，以及一个Warp Scheduler。

### 1.2 SIMT：单指令多线程

SM采用**SIMT（Single-Instruction, Multiple-Thread）** 架构。这是什么意思？

简单说：**一条指令，控制多个线程并行执行**。但和传统的SIMD（单指令多数据）不同，SIMT的每个线程都有自己的指令地址计数器和寄存器状态。这意味着：
- 线程们**一起出发**（执行同一条指令）
- 但可以**各自走不同的路**（条件分支时独立执行）

多处理器以**32个并行线程为一组**来创建、管理、调度和执行线程，这个组叫做**warp**（线程束）。“warp”这个词源于编织——32根线并行编织。

**关键洞察**：warp是GPU**调度和执行**的基本单位，不是thread。你在代码里写的是thread，但硬件真正调度的是warp。

### 1.3 Warp的执行模型

一个warp包含32个线程。当SM执行一个warp时：

1. Warp Scheduler选择一条指令
2. 这条指令被广播到warp中所有32个CUDA Core
3. 32个线程**在同一时钟周期**执行同一条指令

如果所有32个线程走相同的执行路径（没有分支分歧），效率是100%。但如果线程因条件分支而发散（比如一半走if分支，一半走else分支），warp会**依次执行每个分支路径**，并禁用不在该路径上的线程。这就是**分支发散**——它会让warp的有效吞吐量下降。

**重要区别**：分支发散**只发生在同一个warp内部**。不同的warp可以独立执行不同的指令，互不影响。

### 1.4 硬件多线程与延迟隐藏

一个SM可以同时管理多个warp（H100每个SM最多64个并发warp）。当某个warp因为等待内存访问而暂停时，Warp Scheduler会立即切换到另一个准备好的warp执行。

这就是GPU的**延迟隐藏**机制：**用计算掩盖访存延迟**。GPU不是通过降低延迟来提速，而是通过制造足够的并行度，让SM永远有事可做。这也是为什么GPU需要成千上万的线程——线程越多，warp越多，调度器越容易找到“准备好”的warp来执行。

---

## 二、显存层次：从快到慢，从贵到便宜

GPU的存储体系是一个**金字塔结构**：越靠近计算单元，速度越快、容量越小、成本越高。

### 2.1 寄存器（Register）

**最快、最小、线程私有**。

寄存器是片上存储，访问延迟约**1个时钟周期**。每个线程都有自己的寄存器，其他线程无法访问。H100的每个SM子分区有16K个32位寄存器。

**关键约束**：寄存器是SM的**稀缺资源**。如果一个kernel每个线程使用太多寄存器，SM能同时容纳的线程块数量就会减少（occupancy下降）。

```cpp
// 寄存器变量：默认情况下，函数内的局部变量都存储在寄存器中
__global__ void kernel() {
    float x = 1.0f;        // x存储在寄存器中
    int i = threadIdx.x;   // i存储在寄存器中
    // 寄存器是线程私有的，每个线程都有自己的副本
}
```

### 2.2 共享内存（Shared Memory）

**片上、低延迟、线程块内共享**。

共享内存位于芯片上，访问延迟约**20-30个时钟周期**。同一个线程块内的所有线程可以访问同一块共享内存，这使得线程间协作成为可能。

**容量**：每个SM的共享内存通常为**64KB-228KB**（架构相关）。在Ampere架构中，共享内存和L1缓存共享一块片上存储，可以配置为不同的比例（如16KB共享+48KB L1，或反之）。

共享内存的访问速度比全局内存快约**100倍**——这是CUDA优化中最核心的杠杆。

```cpp
// 共享内存：用 __shared__ 关键字声明
__global__ void kernel_with_shared(int *data) {
    __shared__ int shared_data[256];  // 每个block有一份共享内存
    int tid = threadIdx.x;
    
    // 从全局内存加载到共享内存
    shared_data[tid] = data[tid];
    __syncthreads();  // 等待所有线程完成加载
    
    // 所有线程可以访问共享内存中的数据
    int sum = shared_data[0] + shared_data[255];
}
```

### 2.3 全局内存（Global Memory）

**最大、最慢、所有线程可见**。

全局内存就是我们常说的“显存”，位于GPU芯片外部。访问延迟约**数百个时钟周期**。A100的HBM2e显存带宽可达1.2TB/s，但延迟依然很高。

**核心优化原则**：尽量减少全局内存访问，把频繁使用的数据加载到共享内存或寄存器中。

```cpp
// 全局内存：用 cudaMalloc 分配
__global__ void kernel(int *d_data) {
    // d_data指向全局内存
    int val = d_data[threadIdx.x];  // 数百周期的延迟！
}
```

### 2.4 层次对比总结

| 存储类型 | 位置 | 延迟 | 容量(每SM) | 作用域 |
|---------|------|------|-----------|--------|
| 寄存器 | 片上 | ~1周期 | ~64KB-256KB | 单线程 |
| 共享内存 | 片上 | ~20-30周期 | 64KB-228KB | 线程块 |
| 全局内存 | 片外 | ~数百周期 | GB级 | 所有线程 |

**一个直观的比喻**：
- 寄存器 = 你的**大脑**（最快，容量最小）
- 共享内存 = 你面前的**白板**（快，团队共享）
- 全局内存 = 图书馆的**书架**（慢，但什么都有）

---

## 三、三级并行结构：Grid → Block → Thread 的硬件映射

### 3.1 软件层次结构

CUDA编程模型中，kernel以**三层结构**组织：

```
Grid（线程网格）
  └── Block（线程块）
        └── Thread（线程）
```

- **Thread**：最基本的执行单元
- **Block**：一组线程，可以同步（`__syncthreads()`）和通过共享内存通信
- **Grid**：一组Block，构成一次完整的kernel启动

Grid和Block都可以是1D、2D或3D的。

### 3.2 硬件映射：从抽象到物理

**这是整个心智模型的核心**：

| 软件概念 | 硬件映射 | 说明 |
|---------|---------|------|
| **Grid** | 整个GPU | 一个kernel启动对应一个Grid，分布到所有SM |
| **Block** | SM | 一个Block被分配到一个SM上执行 |
| **Warp** | SM调度器 | Block在SM内部被划分为warp（32线程/组） |
| **Thread** | CUDA Core | 线程在CUDA Core上执行 |

### 3.3 详细映射过程

**第一步：GigaThread Engine分发Block**

当CPU启动一个kernel时，GPU上的**GigaThread Engine**（全局调度中枢）接收这个Grid，将其中的Block放入全局Block队列。

各个SM采用**pull模型**：当SM有空闲资源时，主动向GigaThread Engine请求一个新的Block。GigaThread Engine从队列中弹出一个Block分配给该SM。

**关键认知**：Block在SM之间是**动态负载均衡**的——哪个SM先空闲，哪个SM就拿到下一个Block。这意味着：
- Block之间**没有固定的执行顺序**
- 程序**不能假设**Block A在Block B之前完成

**第二步：SM内部将Block划分为Warp**

当一个Block被分配到SM后，SM将其线程按照**连续threadIdx**划分为warp：
- Warp 0：线程 0-31
- Warp 1：线程 32-63
- Warp 2：线程 64-95
- ...

每个warp包含32个连续线程。

**第三步：Warp Scheduler调度执行**

Warp Scheduler以warp为单位进行调度。每个时钟周期，Warp Scheduler选择一个ready的warp，将其下一条指令发射到CUDA Core执行。

**多个Block可以同时在一个SM上执行**——只要SM的资源（寄存器、共享内存）足够容纳它们。这叫做**并发Block**。

### 3.4 关键约束

每个GPU设备对Block大小和Grid大小有限制：
- **每个Block最多1024个线程**
- **Grid的每个维度最大2³¹-1**
- **每个线程使用的寄存器数量**影响SM能容纳的Block数量
- **每个Block使用的共享内存大小**同样影响occupancy

```cpp
// 查询设备限制
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max grid size: %d x %d x %d\n", 
       prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
```

---

## 四、Kernel函数配置与CPU-GPU数据搬运

### 4.1 Kernel启动配置

CUDA kernel的启动语法是：

```cpp
kernel_name<<<grid_dim, block_dim, shared_mem_size, stream>>>(args);
```

- **grid_dim**：Grid的维度（dim3类型），即Block的数量
- **block_dim**：Block的维度（dim3类型），即每个Block的线程数
- **shared_mem_size**（可选）：动态分配的共享内存字节数
- **stream**（可选）：CUDA流句柄

```cpp
// 1D配置：处理N个元素
int N = 1024 * 1024;
int threads_per_block = 256;
int blocks = (N + threads_per_block - 1) / threads_per_block;  // 向上取整
kernel<<<blocks, threads_per_block>>>(d_data, N);

// 2D配置：处理图像
dim3 block(16, 16);   // 16x16 = 256线程
dim3 grid((width + 15) / 16, (height + 15) / 16);
kernel_2d<<<grid, block>>>(d_image, width, height);
```

### 4.2 内存分配与释放

```cpp
float *d_data;
size_t bytes = N * sizeof(float);

// 在GPU上分配内存
cudaError_t err = cudaMalloc(&d_data, bytes);
if (err != cudaSuccess) {
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
}

// 使用完毕后释放
cudaFree(d_data);
```

### 4.3 同步数据传输：cudaMemcpy

`cudaMemcpy`是**同步（阻塞）** 的——函数返回时，数据传输已经完成。

```cpp
float *h_data = (float*)malloc(bytes);  // 主机内存
float *d_data;
cudaMalloc(&d_data, bytes);

// 主机 → 设备（同步）
cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

// 启动kernel
kernel<<<grid, block>>>(d_data, N);

// 设备 → 主机（同步）
cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost);
```

### 4.4 异步传输与Stream：让计算和传输重叠

**问题**：同步模式下，程序按 `H2D传输 → Kernel执行 → D2H传输` 的顺序执行，数据传输时GPU计算单元空闲，Kernel执行时拷贝引擎空闲。

**解决方案**：使用**CUDA Stream**和**异步传输**。

CUDA Stream是一系列操作（内存拷贝、Kernel启动）的**执行队列**。不同Stream中的操作可以**并发执行**。

**使用Stream的前提条件**：
1. 使用**页锁定内存（Pinned Memory）**：`cudaMallocHost()` 或 `cudaHostAlloc()`
2. 使用 **`cudaMemcpyAsync()`** 替代 `cudaMemcpy()`

```cpp
// 1. 分配页锁定内存（主机端）
float *h_data;
cudaMallocHost(&h_data, bytes);  // 页锁定，可被DMA直接访问

// 2. 分配设备内存
float *d_data;
cudaMalloc(&d_data, bytes);

// 3. 创建Stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// 4. 异步传输 + 异步Kernel启动（都在同一个stream中）
cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_data, N);
cudaMemcpyAsync(h_result, d_data, bytes, cudaMemcpyDeviceToHost, stream);

// 5. 等待stream完成
cudaStreamSynchronize(stream);

// 6. 清理
cudaStreamDestroy(stream);
cudaFreeHost(h_data);
cudaFree(d_data);
```

### 4.5 多Stream重叠：高级模式

真正的威力来自**多个Stream**：将数据分成多个chunk，每个chunk使用不同的Stream，实现传输和计算的重叠。

```
时间轴（单Stream）：
H2Dchunk1 | Kernelchunk1 | D2Hchunk1 | H2Dchunk2 | Kernelchunk2 | D2Hchunk2

时间轴（双Stream重叠）：
Stream1: H2Dchunk1 | Kernelchunk1 | D2Hchunk1
Stream2:            H2Dchunk2 | Kernelchunk2 | D2Hchunk2
                      ↑ 重叠区域：Stream1计算时，Stream2在传输
```

这样可以让**拷贝引擎（Copy Engine）和计算引擎（SM）同时工作**，充分利用硬件资源。

```cpp
const int STREAM_COUNT = 4;
cudaStream_t streams[STREAM_COUNT];
for (int i = 0; i < STREAM_COUNT; i++) {
    cudaStreamCreate(&streams[i]);
}

int chunk_size = N / STREAM_COUNT;
for (int i = 0; i < STREAM_COUNT; i++) {
    int offset = i * chunk_size;
    cudaMemcpyAsync(&d_data[offset], &h_data[offset], 
                    chunk_size * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(&d_data[offset], chunk_size);
    cudaMemcpyAsync(&h_result[offset], &d_data[offset],
                    chunk_size * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
}

// 等待所有stream完成
for (int i = 0; i < STREAM_COUNT; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

---

## 五、心智模型总结：一张图看懂全部

```
┌─────────────────────────────────────────────────────────────────────┐
│                        软件抽象 (CUDA编程模型)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Grid (线程网格)                                                     │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │
│  │Block│Block│Block│Block│Block│Block│Block│Block│  ← 多个Block     │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                 │
│     │     │     │     │     │     │     │     │                    │
│     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼                    │
│  Thread Thread Thread Thread Thread Thread Thread Thread           │
│  (0-31) (0-31) (0-31) (0-31) (0-31) (0-31) (0-31) (0-31)           │
│     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑                    │
│     └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                    │
│        每个Block包含多个Thread (最多1024)                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ 映射
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      硬件物理架构 (NVIDIA GPU)                        │
├─────────────────────────────────────────────────────────────────────┤
│  GPU Chip                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │   SM 0  │  │   SM 1  │  │   SM 2  │  │   SM N  │  ← 多个SM        │
│  │┌───────┐│  │┌───────┐│  │┌───────┐│  │┌───────┐│                 │
│  ││Warp 0 ││  ││Warp 0 ││  ││Warp 0 ││  ││Warp 0 ││  ← Warp调度      │
│  ││(T0-31)││  ││(T0-31)││  ││(T0-31)││  ││(T0-31)││                 │
│  │├───────┤│  │├───────┤│  │├───────┤│  │├───────┤│                 │
│  ││Warp 1 ││  ││Warp 1 ││  ││Warp 1 ││  ││Warp 1 ││                 │
│  ││(T32-63││  ││(T32-63││  ││(T32-63││  ││(T32-63││                 │
│  │├───────┤│  │├───────┤│  │├───────┤│  │├───────┤│                 │
│  ││  ...  ││  ││  ...  ││  ││  ...  ││  ││  ...  ││                 │
│  │└───────┘│  │└───────┘│  │└───────┘│  │└───────┘│                 │
│  │ 寄存器   │  │ 寄存器   │  │ 寄存器   │  │ 寄存器   │                 │
│  │ 共享内存 │  │ 共享内存 │   │ 共享内存 │  │ 共享内存 │                  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  全局内存 (HBM, 片外)                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

**核心映射规则**：
1. **Grid → 整个GPU**：一次kernel启动，分布到所有SM
2. **Block → SM**：一个Block固定在一个SM上执行
3. **Warp → SM调度器**：Block在SM内被切分为32线程的Warp
4. **Thread → CUDA Core**：线程在CUDA Core上执行

**内存映射规则**：
1. **寄存器 → 线程私有**：每个线程独享
2. **共享内存 → Block共享**：同一Block内所有线程可见
3. **全局内存 → Grid共享**：所有Block的所有线程可见

---

## 六、实践原则

基于以上心智模型，写CUDA代码时的几个核心原则：

1. **选择合适的Block大小**：通常是32的倍数（warp大小），常见值为128、256、512。过小浪费SM资源，过大可能超过1024限制。
2. **尽量减少全局内存访问**：全局内存延迟数百周期，而寄存器仅1周期。把数据先加载到共享内存再使用，能带来数量级的性能提升。
3. **避免warp内的分支发散**：同一warp的32个线程尽量走相同的执行路径。
4. **使用Stream重叠传输和计算**：用`cudaMemcpyAsync` + 多个Stream，让拷贝引擎和SM并行工作。
5. **关注Occupancy**：SM上活跃warp的数量越多，延迟隐藏效果越好。但寄存器用量和共享内存用量会限制occupancy。