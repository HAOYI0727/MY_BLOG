---
title: Shared Memory & Coalesced Access —— CUDA性能优化技术
published: 2026-04-04
description: 系统讲解CUDA性能优化的两大核心技术：共享内存与合并访存。从矩阵乘法Tiling的分块复用原理出发，剖析共享内存如何将全局内存访问从20亿次降至6.5万次；深入解读Bank Conflict的成因与Padding解法；详解合并访存的条件及其对带宽的10-20倍提升效果。
cover: "/assets/images/posts/cuda_optimization.png"
coverInContent: false
tags: [CUDA, 共享内存, 合并访存, 矩阵乘法Tiling, 性能优化]
category: AI_Infra
draft: false
---

# Shared Memory & Coalesced Access —— CUDA性能优化技术

## 引言：性能瓶颈在哪？

写CUDA程序，第一课是“让代码跑起来”；第二课是“让代码跑得快”。而让代码跑得快，90%的精力都要花在**内存**上。

为什么？因为GPU的计算能力增长远超内存带宽的增长。一个H100的FP32算力高达60 TFLOPS，而HBM3显存带宽约3 TB/s。做个简单的算术：如果每个浮点运算需要从全局内存读一次数据，60 TFLOPS × 4 bytes = 240 TB/s——远超3 TB/s的带宽上限。这意味着，**绝大多数CUDA kernel都是内存带宽受限的**，而非计算受限。

所以，CUDA性能优化的核心命题只有一个：**如何更高效地喂数据给计算单元**。

答案藏在两个关键技术里：**共享内存（Shared Memory）** 和**合并访存（Coalesced Access）** 。前者用片上高速缓存减少对片外全局内存的访问，后者让全局内存访问本身达到最大带宽。本文深入这两个技术的原理与实践。

---

## 一、为什么共享内存是关键？

### 1.1 数字会说话

先看一组具体数据：

| 存储类型 | 位置 | 延迟 | 带宽（约） |
|---------|------|------|-----------|
| 寄存器 | 片上 | ~1周期 | - |
| 共享内存 | 片上 | ~5-30周期 | ~1.7 TB/s |
| 全局内存 | 片外 | ~400-500周期 | ~150-900 GB/s |

共享内存的延迟比全局内存低**约100倍**，带宽高**约10倍**。片上的共享内存与计算单元在同一芯片上，而全局内存在片外的DRAM中，数据需要通过内存总线传输——物理距离决定了延迟差异。

**关键认知**：共享内存不是“更快的全局内存”，而是一种**完全不同的编程范式**。全局内存是隐式、自动的；共享内存需要**显式管理**——你得手动把数据从全局内存搬进共享内存，用完再搬出去。

### 1.2 用共享内存解决“重复读取”问题

什么时候该用共享内存？**当数据被多次读取时**。

考虑矩阵乘法 $C = A \times B$。对于输出矩阵 $C$ 的每个元素 $C_{i,j}$：

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

朴素实现中，每个 $A_{i,k}$ 和 $B_{k,j}$ 会被反复从全局内存读取——$A$ 的每个元素被读取 $N$ 次（对应 $C$ 的每一列），$B$ 的每个元素被读取 $M$ 次（对应 $C$ 的每一行）。对于 $1024 \times 1024$ 的矩阵，这意味着约 **20亿次** 全局内存访问。

共享内存的解决思路：**把数据切成Tile（分块），每次把一个Tile加载到共享内存，Block内的所有线程共享这个Tile，用完再加载下一个**。

---

## 二、矩阵乘法Tiling：共享内存的经典应用

### 2.1 朴素实现的瓶颈

先看一个不使用共享内存的朴素矩阵乘法kernel：

```cpp
// 朴素矩阵乘法：每个线程计算 C 的一个元素
__global__ void matmul_naive(float *A, float *B, float *C, 
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 每次循环都从全局内存读取
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

问题在哪？每次内层循环的 $k$ 迭代，都要从全局内存读取 `A[row * K + k]` 和 `B[k * N + col]`。一个线程计算一个输出元素需要 $2K$ 次全局内存读取。对于 $1024\times1024$ 的矩阵，每个线程需要读2048次全局内存，总共约 **20亿次全局内存事务**。

### 2.2 Tiling的核心思想：分块复用

Tiling的核心是**分而治之**：

1. 将矩阵 $A$ 和 $B$ 分成若干个 **Tile（分块）**
2. 每个Block负责计算输出矩阵 $C$ 的一个Tile
3. Block内的线程**协作**将 $A$ 和 $B$ 的对应Tile加载到共享内存
4. 所有线程从共享内存读取数据计算，**避免重复访问全局内存**

```
矩阵乘法分块示意（Tile Size = 2）：

A (4x4)          B (4x4)          C (4x4)
┌─────┬─────┐     ┌─────┬─────┐     ┌─────┬─────┐
│ a00 │ a01 │  ×  │ b00 │ b01 │  =  │ c00 │ c01 │
│ a10 │ a11 │     │ b10 │ b11 │     │ c10 │ c11 │
├─────┼─────┤     ├─────┼─────┤     ├─────┼─────┤
│ a20 │ a21 │     │ b20 │ b21 │     │ c20 │ c21 │
│ a30 │ a31 │     │ b30 │ b31 │     │ c30 │ c31 │
└─────┴─────┘     └─────┴─────┘     └─────┴─────┘

Block(0,0) 计算 C 的左上 Tile：
  从 A 加载 Tile(0,0) 到共享内存
  从 B 加载 Tile(0,0) 到共享内存
  计算 C 的左上 2x2
  然后从 A 加载 Tile(0,1)，从 B 加载 Tile(1,0)...
```

### 2.3 Tiled矩阵乘法的完整实现

下面是一个完整的Tiled矩阵乘法kernel：

```cpp
#define TILE_SIZE 32  // 常用值：16或32

__global__ void matmul_tiled(float *A, float *B, float *C,
                             int M, int N, int K) {
    // 声明共享内存：每个Block有两个Tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 线程在Block内的索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程在全局矩阵中的位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历 K 维度上的所有 Tile
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 1. 协作加载 A 的 Tile 到共享内存
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[a_row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;  // 边界填充
        }
        
        // 2. 协作加载 B 的 Tile 到共享内存
        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();  // 确保所有线程完成加载
        
        // 3. 从共享内存计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();  // 确保所有线程完成计算再加载下一Tile
    }
    
    // 写回结果到全局内存
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**性能提升从哪来？**

- 朴素实现：每个线程计算一个元素需要 $2K$ 次全局内存读取
- Tiled实现：每个线程每Tile加载2个元素到共享内存（$2 \times \text{num\_tiles}$ 次全局读取），但计算时从共享内存读取 $2 \times \text{TILE\_SIZE} \times \text{num\_tiles}$ 次

对于 $1024\times1024$ 矩阵，`TILE_SIZE=32`：
- 全局内存读取：从 **20亿次** 降到约 **6.5万次**（每个线程加载 $2 \times 32 \times 32 = 2048$ 个元素到共享内存，$1024\times1024$ 个线程 = 约20亿次，等等——这里需要重新算清楚）

**正确的计算**：每个Block有 $32\times32=1024$ 个线程。每个Tile阶段，Block协作加载 $32\times32=1024$ 个 $A$ 元素和1024个 $B$ 元素到共享内存，共2048次全局读取。共有 $1024/32=32$ 个Tile阶段。所以全局内存读取总量为 $2048 \times 32 = 65536$ 次——比朴素实现的约20亿次减少了**3个数量级**。

这就是Tiling的力量：**用少量的全局内存加载，支撑大量的共享内存计算**。

---

## 三、Bank Conflict：共享内存的“暗礁”

共享内存很快，但**不合理的使用方式会让它变慢**。

### 3.1 什么是Bank Conflict？

共享内存在硬件上被划分为 **32个独立的Bank**（存储体）。每个Bank在每个时钟周期可以响应一次访问。

- **32个Bank** ↔ **Warp的32个线程**（这不是巧合）

理想情况：一个Warp的32个线程同时访问**不同Bank**的数据，32个请求并行完成。

问题情况：多个线程访问**同一个Bank的不同地址**，访问被**串行化**——这就是Bank Conflict。

```
Bank映射示意（32个Bank，每个Bank 4字节宽）：

地址 0-3    → Bank 0
地址 4-7    → Bank 1
地址 8-11   → Bank 2
...
地址 124-127 → Bank 31
地址 128-131 → Bank 0  ← 循环
地址 132-135 → Bank 1
...
```

对于 `float` 数组（4字节），Bank索引的计算公式为：

$$\text{bank\_id} = \text{address} / 4 \mod 32$$

### 3.2 冲突的代价

**最坏情况**：一个Warp的32个线程全部访问同一个Bank的32个不同地址 → 32次访问全部串行化，有效带宽降到 **1/32**。

**常见冲突场景**：矩阵转置中按列访问 `shared[32][32]`。

```cpp
// 假设 tile[32][32] 在共享内存中
__shared__ float tile[32][32];

// 场景1：按行访问 — 无冲突
float val = tile[threadIdx.y][threadIdx.x];  
// thread 0 → tile[0][0] → Bank 0
// thread 1 → tile[0][1] → Bank 1
// ... 完美并行

// 场景2：按列访问 — 32路冲突！
float val = tile[threadIdx.x][threadIdx.y];  
// thread 0 → tile[0][0] → Bank 0
// thread 1 → tile[1][0] → Bank 1  ← 等等，这里其实没问题
// 实际上按列访问 tile[row][col] 时，threadIdx.x 控制行
// thread 0 → tile[0][0] → Bank 0
// thread 1 → tile[1][0] → Bank 1
// ... 如果行数 < 32，看起来也没冲突？

// 真正的问题场景：按 stride=32 访问
float val = tile[threadIdx.x][0];  
// thread 0 → tile[0][0] → Bank 0
// thread 1 → tile[1][0] → Bank 1  (因为每行32个float，行偏移32×4=128字节)
// 等等，tile[1][0] 的地址是 tile[0][0] + 32*4 = +128字节
// 128/4=32，所以 tile[1][0] 也在 Bank 0！
// 所有32个线程全部访问 Bank 0 → 32路冲突！
```

核心规律：当访问步长（stride）是 **32的倍数** 时，所有线程撞到同一个Bank。

### 3.3 解决方案：Padding（填充）

解决Bank Conflict的经典方法：**在数组每行末尾插入一个“哑元”**，打破32的对齐。

```cpp
// 有冲突的版本
__shared__ float tile[32][32];  // 每行32个float，步长32

// 无冲突的版本
__shared__ float tile[32][33];  // 每行33个float，步长33 ≠ 32的倍数
```

为什么 `tile[32][33]` 有效？

- `tile[0][0]` → Bank 0
- `tile[1][0]` → 地址偏移 $33 \times 4 = 132$ 字节 → $132/4 = 33$ → $33 \mod 32 = 1$ → Bank 1
- `tile[2][0]` → 地址偏移 $66 \times 4 = 264$ 字节 → $264/4 = 66$ → $66 \mod 32 = 2$ → Bank 2
- ...

步长从32变成33，32个线程分别落在32个不同的Bank上，**冲突完全消除**。

**注意**：Padding增加了一列存储开销（从1024个float变成1056个），换来的是**共享内存访问带宽从1/32提升到100%**——这笔交易非常划算。

---

## 四、合并访存：全局内存的“正确打开方式”

共享内存解决了“数据复用”的问题，但**第一次从全局内存加载数据到共享内存**时，我们仍然要访问全局内存。如何让这次访问也达到最高效率？

答案是**合并访存（Coalesced Access）** 。

### 4.1 什么是合并访存？

当一个Warp的32个线程访问全局内存时，GPU硬件会**合并**这些请求：

- 如果32个线程访问**连续的内存地址**，硬件将它们合并成**少数几个内存事务**
- 如果访问是**分散的**，每个线程的请求独立处理，产生大量事务

**一个合并访问可以达到分散访问10-20倍的带宽**。

### 4.2 合并访存的条件

对于计算能力6.0及以上的设备，合并访存的基本条件是：

> **第 $k$ 个线程访问第 $k$ 个32字节对齐的字**

简化理解：**Warp内线程按连续地址访问**。

```cpp
// ✅ 合并访问：线程访问连续地址
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];  
// thread 0 → data[0], thread 1 → data[1], ... 
// 连续，合并为一个事务

// ❌ 非合并访问：步长访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx * 32];  
// thread 0 → data[0], thread 1 → data[32], thread 2 → data[64]
// 分散，需要32个独立事务
```

### 4.3 非合并访问的代价

实验数据（Tesla T4）：

| 访问步长 | 相对性能 |
|---------|---------|
| stride=1（合并） | 1×（基准） |
| stride=8 | 慢约8倍 |
| stride=16 | 慢约14.6倍 |
| stride=32 | 慢约25.5倍 |

**非合并访问可能浪费90%以上的可用带宽**。

### 4.4 二维数组的合并访问

处理二维数组时，需要注意**内存布局**：

```cpp
// 行优先存储（Row-major）：C/C++默认
// data[row * width + col]

// ✅ 合并：按行访问（同一行内连续列）
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[row * width + col];  
// 同一Warp内的线程 col 连续 → 合并

// ❌ 非合并：按列访问
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[row * width + col];  
// 等等，这和上面看起来一样？
// 关键在于：Warp内 threadIdx.x 变化时访问的是同一行的连续列 → 合并
// 但如果用 threadIdx.x 控制行：
float val = data[(blockIdx.y * blockDim.y + threadIdx.x) * width + col];
// thread 0 → row 0, thread 1 → row 1 → 地址间隔 width 个元素 → 非合并！
```

**核心原则**：让 `threadIdx.x` 对应**内存中连续变化的维度**。

### 4.5 用共享内存“修复”非合并访问

有些算法天然需要非合并访问模式（如矩阵转置）。这时候可以用共享内存作为**中转站**：

1. 从全局内存**合并地**读取数据到共享内存
2. 在共享内存中**重新排列**（共享内存没有合并访问的限制，只有Bank Conflict的限制）
3. 从共享内存**合并地**写回全局内存

这就是为什么矩阵转置优化中常用 `__shared__ float tile[32][33]`——既保证了全局内存的合并访问，又避免了共享内存的Bank Conflict。

---

## 五、实践总结

### 5.1 优化决策树

写CUDA kernel时，按这个顺序思考：

```
1. 数据会被多次读取吗？
   ├─ 是 → 用共享内存Tiling
   └─ 否 → 直接从全局内存读取（但要保证合并访问）

2. 全局内存访问是合并的吗？
   ├─ 是 → 带宽利用率高 ✓
   └─ 否 → 调整数据布局或使用共享内存转置

3. 共享内存访问有Bank Conflict吗？
   ├─ 无 → 带宽利用率高 ✓
   └─ 有 → 加Padding或调整访问模式
```

### 5.2 核心数值记忆

| 指标 | 数值 | 含义 |
|------|------|------|
| 全局内存延迟 | ~400-500周期 | 一次全局读取够执行几百条指令 |
| 共享内存延迟 | ~5-30周期 | 约100倍于全局内存 |
| Warp大小 | 32线程 | 所有优化都围绕这个数字 |
| Shared Memory Banks | 32个 | 与Warp大小一致 |
| Bank大小 | 4字节（默认） | 可配置为8字节 |
| 合并访问提升 | 10-20× | 相对于分散访问 |

### 5.3 常见陷阱

1. **盲目使用共享内存**：如果数据只读取一次，用共享内存反而增加了一次额外的拷贝开销
2. **忽略Bank Conflict**：以为用了共享内存就万事大吉，结果被Bank Conflict拖累
3. **Tile Size选择不当**：太大浪费共享内存、降低Occupancy；太小无法充分利用数据复用
4. **忘记 `__syncthreads()`**：在一个线程还没加载完数据时，另一个线程就开始读取

### 5.4 调试工具

- **`nvprof` / `ncu`（Nsight Compute）**：查看 `shared_efficiency`、`warp_serialize` 等指标
- **编译选项 `--ptxas-options=-v`**：查看寄存器、共享内存使用量
- **CUDA Sample `simpleTemplates`**：矩阵乘法的参考实现