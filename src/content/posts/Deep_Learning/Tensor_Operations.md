---
title: Tensor Operations —— 向量、矩阵与张量运算
published: 2025-08-02
description: 系统讲解深度学习中的线性代数基础：从张量的维度与Shape变换出发，深入区分Hadamard积（*）与矩阵乘法（@）的本质差异，解析L1/L2/Frobenius范数的几何意义与机器学习用途，并直观理解特征分解的“换基→缩放→换回”几何解释，以及SVD在降维与数据压缩中的核心价值。
cover: "/assets/images/posts/tensor_operations.png"
coverInContent: false
tags: [Tensor, 张量, 矩阵乘法, Hadamard积, 范数, 特征分解, SVD, 深度学习]
category: Deep_Learning
draft: false
---

# Tensor Operations —— 向量、矩阵与张量运算

## 一、引言：从“数据语言”到“模型骨架”

欢迎来到深度学习的世界。在接触**自动微分、MLP或优化器**之前，我们必须先厘清数据在计算机中的物理载体——**张量（Tensor）**。

本系列将本博客置于开篇之首，是后续一切复杂模型的“**刚需**”。无论是后续**全连接层**的特征混合（`Wx+b`），**卷积核**在图像上的滑动，还是**循环状态**在时序上的传递，其底层本质都归结于**高效的张量变换**。若在**手推反向传播**时被“维度不匹配”折磨，或理解**权重初始化**时对矩阵形状生疏，根源往往在于本篇基础尚未夯实。

本文不满足于“**高维数组**”的模糊直觉，而是直击五个根本痛点：其一，厘清`reshape`与`transpose`背后 **“存储与视图”的分离思想**，这是高效内存管理的前提；其二，严格区分**Hadamard积（`*`）与矩阵乘法（`@`）** —— 前者是**注意力与门控机制**的“调制器”，后者是**全连接层**的“特征混合器”；其三，解析**L1/L2/Frobenius范数**的几何意义，这将是理解后续**权重衰减**（正则化）的数学基石；其四，直观拆解特征分解 **“换基-缩放-换回”** 的几何过程；其五，揭示**SVD**作为任意矩阵降维与压缩的终极价值。

需要特别留意的是，本篇对线性变换的空间感知，将直接服务于卷积神经网络中**感受野**的推导。请带着“**数据如何流动**”的疑问进入正文——唯有基座稳固，后续的模型大厦方能拔地而起。

---

## 二、张量的维度与Shape变换

### 2.1 什么是张量？

**张量（Tensor）是深度学习中数据的基本容器**。你可以把它理解为**多维数组**的统称：

- **0阶张量**：标量（一个数），如 `3.14`
- **1阶张量**：向量（一维数组），如 `[1, 2, 3]`
- **2阶张量**：矩阵（二维数组），如一张灰度图
- **3阶张量**：如一张RGB彩色图（高×宽×通道）
- **4阶张量**：如一个批次的RGB图片（批次×高×宽×通道）

张量的**形状（Shape）** 是一个元组，描述了每个维度的大小。例如 `shape=(3, 4)` 表示一个3行4列的矩阵。

```python
import torch
import numpy as np

# 创建各种张量
scalar = torch.tensor(3.14)          # 0阶：标量
vector = torch.tensor([1, 2, 3])     # 1阶：向量
matrix = torch.tensor([[1, 2], [3, 4]])  # 2阶：矩阵
tensor_3d = torch.randn(2, 3, 4)     # 3阶：形状(2,3,4)

print(f"标量形状: {scalar.shape}")    # torch.Size([])
print(f"向量形状: {vector.shape}")    # torch.Size([3])
print(f"矩阵形状: {matrix.shape}")    # torch.Size([2, 2])
print(f"3D张量形状: {tensor_3d.shape}")  # torch.Size([2, 3, 4])
```

### 2.2 Shape变换的核心操作

形状变换是指**在不改变张量数据内容的情况下，改变其维度和每个维度的大小**。理解这些操作的关键，在于搞清楚**存储（Storage）** 与**视图（View）** 的区别：

- **存储**：数据在内存中实际存放的连续区域
- **视图**：我们“看待”这段内存的方式（通过shape和stride元数据来定义）

```python
# 创建一个连续存储的张量
x = torch.arange(12)
print(f"原始数据: {x}")  # tensor([0, 1, 2, ..., 11])

# reshape：改变视图，数据在内存中的顺序不变
x_reshaped = x.reshape(3, 4)
print(f"reshape后:\n{x_reshaped}")
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
```

#### （1）`reshape` —— 重塑形状

`reshape`是最常用的形状变换函数，它**不改变元素数量和元素值**，只改变张量的逻辑形状。

```python
# 将24个元素重塑为不同形状
x = torch.arange(24)

# 变成 2×3×4
a = x.reshape(2, 3, 4)
print(a.shape)  # torch.Size([2, 3, 4])

# 使用 -1 让PyTorch自动计算该维度大小
b = x.reshape(2, 3, -1)
print(b.shape)  # torch.Size([2, 3, 4])，-1自动算为4

# 展平为一维
c = x.reshape(-1)
print(c.shape)  # torch.Size([24])
```

**关键点**：`reshape`要求新形状的元素总数与原张量相同。`-1`是一个便利写法，表示“自动计算这个维度的大小”。

#### （2）`transpose` 与 `permute` —— 交换维度

`transpose`用于**交换两个指定的维度**，`permute`可以**重新排列任意数量的维度**。

```python
# 创建一个3维张量
x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)

# transpose：交换维度0和维度1
y = x.transpose(0, 1)
print(y.shape)  # torch.Size([3, 2, 4])

# permute：重新排列所有维度
z = x.permute(2, 0, 1)
print(z.shape)  # torch.Size([4, 2, 3])
```

**重要区别**：`reshape`只是改变“看待数据的方式”，而`transpose`/`permute`会**改变维度之间的逻辑关系**。经过`transpose`后，张量在内存中变得**不连续（non-contiguous）** ，此时如果再调用`reshape`，PyTorch会先拷贝数据使其连续。

```python
x = torch.arange(12).reshape(3, 4)
print(x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# transpose后，内存布局改变了
x_t = x.transpose(0, 1)
print(x_t)
# tensor([[ 0,  4,  8],
#         [ 1,  5,  9],
#         [ 2,  6, 10],
#         [ 3,  7, 11]])

print(x_t.is_contiguous())  # False
```

#### （3）`squeeze` 与 `unsqueeze` —— 压缩与扩展维度

- `squeeze`：删除所有大小为1的维度
- `unsqueeze`：在指定位置插入一个大小为1的维度

```python
x = torch.randn(1, 3, 1, 4)

# squeeze：删除所有尺寸为1的维度
y = x.squeeze()
print(y.shape)  # torch.Size([3, 4])

# unsqueeze：在指定位置插入维度
z = torch.tensor([1, 2, 3])
w = z.unsqueeze(0)  # 在第0维插入
print(w.shape)  # torch.Size([1, 3])
```

### 2.3 实战：图像数据的Shape变换

在计算机视觉中，图像数据经常需要在不同形状间转换：

```python
# 假设有一批RGB图像：batch_size=32, 通道数=3, 高=224, 宽=224
images = torch.randn(32, 3, 224, 224)

# 展平为特征向量（用于全连接层）
flattened = images.reshape(32, -1)
print(flattened.shape)  # torch.Size([32, 150528])

# 交换通道维度和宽高维度（某些框架需要）
# 从 (B, C, H, W) 变为 (B, H, W, C)
images_chw = images.permute(0, 2, 3, 1)
print(images_chw.shape)  # torch.Size([32, 224, 224, 3])
```


## 三、矩阵乘法（`@`）与Hadamard积（`*`）的区别

### 3.1 Hadamard积（逐元素乘法）

**Hadamard积（Hadamard Product）** ，也叫**逐元素乘法（Element-wise Multiplication）** 或**Schur积**，是两个**同型矩阵**对应位置元素相乘：

$$(A \circ B)_{ij} = a_{ij} \cdot b_{ij}$$

在Python中，`*`运算符默认执行的就是Hadamard积。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Hadamard积（逐元素乘法）
C = A * B
print("Hadamard积 (A * B):")
print(C)
# [[ 5 12]
#  [21 32]]
# 验证：1×5=5, 2×6=12, 3×7=21, 4×8=32
```

Hadamard积具有**交换律**和**结合律**，这是它与普通矩阵乘法的关键区别之一。

### 3.2 矩阵乘法（普通乘积）

**矩阵乘法**（也叫点积、`matmul`）遵循“左行乘右列”的规则：

$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$$

要求：**A的列数 = B的行数**。结果的形状为 `(A的行数, B的列数)`。

```python
# 矩阵乘法（使用 @ 运算符）
D = A @ B
print("矩阵乘法 (A @ B):")
print(D)
# [[19 22]
#  [43 50]]
# 验证：19 = 1×5 + 2×7, 22 = 1×6 + 2×8
#       43 = 3×5 + 4×7, 50 = 3×6 + 4×8

# 形状规则
A2 = np.ones((2, 3))
B2 = np.ones((3, 4))
C2 = A2 @ B2
print(C2.shape)  # (2, 4)
```

在Python中，矩阵乘法可以通过三种方式实现：
- `@` 运算符（Python 3.5+引入，最推荐）
- `np.matmul()` 函数
- `np.dot()` 函数（对二维数组等同于矩阵乘法，但对高维数组行为不同）

### 3.3 关键区别总结

| 特性 | Hadamard积（`*`） | 矩阵乘法（`@`） |
|------|-------------------|-----------------|
| 运算规则 | 对应元素相乘 | 行×列求和 |
| 形状要求 | 必须完全相同 | A的列数 = B的行数 |
| 结果形状 | 与原矩阵相同 | (A的行数, B的列数) |
| 交换律 | ✅ 成立 | ❌ 不成立 |
| AI中的典型用途 | 注意力权重、门控机制 | 全连接层、卷积层 |

```python
# 直观对比
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A * B (Hadamard):")
print(A * B)
# [[ 5 12]
#  [21 32]]

print("A @ B (矩阵乘法):")
print(A @ B)
# [[19 22]
#  [43 50]]
```

在神经网络中，**全连接层** `y = Wx + b` 使用的是矩阵乘法（`@`），而**注意力机制**中的权重计算、**门控循环单元**中的门控信号则常使用Hadamard积（`*`）。


## 四、范数（L1/L2/Frobenius）的几何意义

**范数（Norm）** 的本质是**度量向量或矩阵“大小”的工具**。不同的范数从不同角度衡量“大小”，在机器学习中各有用途。

### 4.1 L1范数：曼哈顿距离

L1范数定义为向量**各元素绝对值之和**：

$$\|x\|_1 = \sum_{i=1}^{n} |x_i|$$

**几何意义**：L1范数对应**曼哈顿距离（Manhattan Distance）** ——就像在曼哈顿街区行走，只能沿着街道直角转弯，距离是各段路程的绝对值之和。

```python
import torch

v = torch.tensor([3.0, -4.0])

# L1范数 = |3| + |-4| = 7
l1_norm = torch.linalg.norm(v, ord=1)
print(f"L1范数: {l1_norm.item()}")  # 7.0
```

**为什么L1范数能产生稀疏解？** 在二维平面上，L1范数的“单位球”是一个菱形（顶点在坐标轴上）。当用L1范数做正则化时，优化问题的解更容易落在坐标轴上——这意味着某些维度的权重被压缩为0，从而实现**特征选择**。这就是Lasso回归使用L1正则化的原因。

### 4.2 L2范数：欧几里得距离

L2范数定义为向量**各元素平方和的平方根**：

$$\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**几何意义**：L2范数就是**欧几里得距离（Euclidean Distance）** ——从原点出发到目标点的直线距离。

```python
# L2范数 = sqrt(3² + (-4)²) = 5
l2_norm = torch.linalg.norm(v, ord=2)
print(f"L2范数: {l2_norm.item()}")  # 5.0
```

L2范数的平方有一个常用形式：$\|x\|_2^2 = x^T x$。在机器学习中，L2正则化（岭回归/权重衰减）就是惩罚权重的L2范数平方，倾向于让所有权重都**较小但非零**，从而防止过拟合。

### 4.3 Frobenius范数：矩阵的“L2范数”

**Frobenius范数**是L2范数在**矩阵**上的推广——将矩阵所有元素的平方和开根号：

$$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} |a_{ij}|^2}$$

从另一个角度看，如果把矩阵的所有列“堆叠”成一个长向量，Frobenius范数就是这个长向量的L2范数。

```python
M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Frobenius范数 = sqrt(1² + 2² + 3² + 4²) = sqrt(30) ≈ 5.477
fro_norm = torch.linalg.norm(M, ord='fro')
print(f"Frobenius范数: {fro_norm.item():.3f}")  # 5.477
```

**Frobenius范数的几何意义**：它衡量的是矩阵作为一个整体在“欧几里得空间”中的长度。在矩阵低秩近似中（如SVD截断），我们通常用Frobenius范数来衡量**重建误差**——保留的奇异值越多，重建误差（Frobenius范数）越小。

### 4.4 三种范数的直观对比

| 范数 | 公式 | 几何意义 | 机器学习用途 |
|------|------|----------|-------------|
| L1 | $\sum |x_i|$ | 曼哈顿距离 | Lasso回归（特征选择） |
| L2 | $\sqrt{\sum x_i^2}$ | 欧几里得距离 | 岭回归、权重衰减 |
| Frobenius | $\sqrt{\sum a_{ij}^2}$ | 矩阵的“长度” | 矩阵近似、SVD重建误差 |


## 五、特征分解（Eigenvalue Decomposition）的直观理解

### 5.1 特征向量与特征值：线性变换的“主轴”

**特征分解**（Eigendecomposition）是将一个**方阵**分解为特征向量和特征值的操作。

**核心思想**：一个矩阵 $A$ 代表一个**线性变换**。绝大多数向量经过这个变换后，方向和长度都会改变。但存在一些**特殊的向量**，它们的方向在变换后**保持不变**，只是长度被拉伸或压缩了。

这些特殊的向量就是**特征向量（Eigenvector）** ，拉伸或压缩的倍数就是**特征值（Eigenvalue）** ：

$$A v = \lambda v$$

其中 $v$ 是特征向量，$\lambda$ 是特征值。

### 5.2 特征分解的公式

如果一个 $n \times n$ 的方阵 $A$ 有 $n$ 个线性无关的特征向量，它可以被分解为：

$$A = Q \Lambda Q^{-1}$$

其中：
- $Q$ 是特征向量组成的矩阵（每一列是一个特征向量）
- $\Lambda$ 是对角矩阵，对角线上的元素是特征值

### 5.3 几何直觉：特征分解 = “换基 → 缩放 → 换回”

想象一个线性变换 $A$ 作用于空间中的向量：

1. **换基**：用特征向量作为新的坐标系（乘以 $Q^{-1}$）
2. **缩放**：在新坐标系下，变换只是沿各坐标轴做拉伸/压缩（乘以 $\Lambda$）
3. **换回原基**：将结果转回原来的坐标系（乘以 $Q$）

```python
import numpy as np

# 创建一个对称矩阵（保证可特征分解）
A = np.array([[4, 1], [1, 3]])

# 特征分解
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)        # [4.618, 2.382]
print("特征向量:\n", eigenvectors)
# 每一列是一个特征向量

# 验证：A @ v = λ * v
v = eigenvectors[:, 0]
lam = eigenvalues[0]
print("A @ v:", A @ v)
print("λ * v:", lam * v)
# 两者应该相等
```

**特征值的大小表示“重要性”** ：特征值越大，对应的特征向量方向在变换中越重要。这就是为什么PCA（主成分分析）会选择**最大特征值对应的特征向量**作为主成分——它们代表了数据中**方差最大的方向**。

### 5.4 特征分解的局限

**特征分解要求矩阵必须是方阵**。但在实际数据中，我们经常遇到**非方阵**的数据矩阵（如 $m$ 个样本 × $n$ 个特征，$m \neq n$）。这时候就需要**奇异值分解（SVD）** 出场了。


## 六、奇异值分解（SVD）：降维与数据压缩的利器

### 6.1 从特征分解到SVD

**特征值分解的局限**：只有方阵才能做特征分解。

**SVD的突破**：**任意矩阵**（无论是否方阵）都可以进行奇异值分解。

### 6.2 SVD的数学形式

任意 $m \times n$ 的矩阵 $A$ 都可以分解为：

$$A = U \Sigma V^T$$

其中：
- $U$：$m \times m$ 的**正交矩阵**，列向量称为**左奇异向量**
- $\Sigma$：$m \times n$ 的**对角矩阵**，对角线上的值称为**奇异值**（按从大到小排列）
- $V$：$n \times n$ 的**正交矩阵**，列向量称为**右奇异向量**

```python
from scipy.linalg import svd

# 任意非方阵矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])  # shape: (4, 3)

U, S, Vt = svd(A)

print("U的形状:", U.shape)      # (4, 4)
print("S（奇异值）:", S)        # [25.46, 1.72, 0.00]
print("Vt的形状:", Vt.shape)    # (3, 3)
```

### 6.3 几何直觉：SVD = “旋转 → 缩放 → 再旋转”

从几何角度看，任何矩阵 $A$ 的线性变换效果都可以分解为三个步骤：

1. **旋转/反射**（$V^T$）：在原始空间中旋转向量
2. **沿坐标轴缩放**（$\Sigma$）：在各方向上做拉伸或压缩
3. **旋转/反射**（$U$）：在目标空间中旋转

奇异值 $\sigma_i$ 就是**第 $i$ 个方向上的缩放倍数**。奇异值越大，说明这个方向在变换中越重要。

### 6.4 SVD的核心价值：低秩近似

SVD最强大的应用在于**低秩近似（Low-Rank Approximation）** ——用少数几个最大的奇异值及其对应的奇异向量来**近似**原始矩阵。

$$A \approx A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

其中 $k$ 远小于矩阵的秩。**保留的奇异值越多，近似越精确；保留的越少，压缩率越高**。

### 6.5 实战：用SVD压缩图像

这是理解SVD数据压缩能力最直观的例子。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image

# 加载灰度图像并转为矩阵
# 这里用随机数据模拟，实际使用时替换为真实图像
# img = Image.open('image.jpg').convert('L')
# A = np.array(img, dtype=float)

# 为演示，创建一个有结构的矩阵（模拟图像）
np.random.seed(42)
A = np.random.randn(200, 200)
# 加一些结构使其更像图像
A = A @ A.T  # 使其具有低秩结构

def svd_compress(A, k):
    """用SVD将矩阵A压缩到k个奇异值"""
    U, S, Vt = svd(A, full_matrices=False)
    # 只保留前k个奇异值
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    # 重建矩阵
    A_k = U_k @ S_k @ Vt_k
    return A_k, S

# 分别用不同数量的奇异值重建
ks = [5, 20, 50, 100, 200]
fig, axes = plt.subplots(1, len(ks), figsize=(15, 3))

for i, k in enumerate(ks):
    A_k, S = svd_compress(A, k)
    axes[i].imshow(A_k, cmap='gray')
    axes[i].set_title(f'k={k}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 计算压缩比
def compression_ratio(m, n, k):
    """SVD压缩的存储压缩比"""
    # 原始存储: m*n
    # 压缩后存储: U(m*k) + Sigma(k) + Vt(k*n)
    compressed = m * k + k + k * n
    original = m * n
    return compressed / original

print(f"原始大小: {200*200} = 40000")
print(f"k=20时压缩后大小: {compression_ratio(200, 200, 20):.2%}")
# 输出类似: 19.90% —— 只需约20%的存储空间！
```

**SVD图像压缩的直观理解**：

- 奇异值从大到小排列，**前几个奇异值往往占了总能量的绝大部分**
- 保留前 $k$ 个奇异值重建图像，相当于**用少数几个“特征模式”来近似整张图**
- $k$ 越小，压缩率越高，但图像越模糊；$k$ 越大，图像越清晰，但存储越大

实际案例中，**保留前100个奇异值（$k=100$）重建的图像与原图几乎无法区分，但存储空间只需要约12%**。

### 6.6 SVD vs 特征分解：谁更通用？

| 特性 | 特征分解（EVD） | 奇异值分解（SVD） |
|------|----------------|------------------|
| 适用矩阵 | 仅限方阵 | **任意矩阵** |
| 分解结果 | $A = Q\Lambda Q^{-1}$ | $A = U\Sigma V^T$ |
| 正交性 | 特征向量不一定正交 | 奇异向量**一定正交** |
| 数值稳定性 | 对病态矩阵敏感 | 数值稳定性更好 |
| PCA实现 | 对协方差矩阵做EVD | **直接对数据矩阵做SVD** |

在实际工程中，**PCA通常用SVD实现而非特征分解**，因为SVD直接作用于数据矩阵，数值更稳定，且适用于大规模数据。


## 七、总结

回顾本文的核心脉络：

1. **张量**是AI数据的通用容器，理解`shape`、`reshape`、`transpose`、`squeeze`等操作的本质——**存储与视图的分离**——是高效处理数据的基础。

2. **矩阵乘法（`@`）** 与**Hadamard积（`*`）** 服务于完全不同的目的：前者是“特征混合”（全连接层），后者是“特征调制”（注意力机制）。

3. **范数**是衡量“大小”的尺子：L1是曼哈顿距离（产生稀疏），L2是欧几里得距离（权重衰减），Frobenius是矩阵的“L2长度”（衡量重建误差）。

4. **特征分解**将方阵分解为“方向×重要性”，是理解线性变换“主轴”的窗口。

5. **SVD**是特征分解的**终极推广**——适用于任意矩阵，是降维（PCA）、数据压缩（图像压缩）、推荐系统的数学基石。

这些概念不是孤立的数学公式，而是贯穿AI模型设计、训练、部署全流程的**思维工具**。理解它们，你就能真正“读懂”深度学习模型的每一个运算。