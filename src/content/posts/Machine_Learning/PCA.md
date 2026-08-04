---
title: Principal Component Analysis (PCA) —— 主成分分析
published: 2025-07-24
description: 系统讲解主成分分析（PCA）的完整数学原理：从方差最大化与最小化重构误差两个等价视角出发，通过拉格朗日乘子法推导出协方差矩阵的特征方程，揭示特征向量即主成分方向、特征值即主成分方差的本质联系；深入对比EVD与SVD两种实现方式的优劣与适用场景；详细介绍三种主成分数量选择方法；讨论PCA的假设和局限。
cover: "/assets/images/posts/pca_derivation.png"
coverInContent: false
tags: [主成分分析, PCA, 特征值分解, SVD, 降维, 协方差矩阵, 机器学习]
category: Machine_Learning
draft: false
---

# Principal Component Analysis (PCA) —— 主成分分析

## 引言

经过**第二阶段：树模型与集成学习**的完整旅程，我们完成了**监督学习**核心范式的全部拼图 —— 从单棵决策树的生长与剪枝，到随机森林的并行降方差，再到 AdaBoost、GBM、XGBoost、LightGBM 的串行降偏差迭代。这些算法的共同前提是：**训练数据带有明确的标签 $y$，模型的目标是学习从 $x$ 到 $y$ 的映射关系**。

现在，我们将进入一个全新的领域——**无监督学习**。这里的挑战不再是“**如何预测**”，而是 **“如何从无标签数据中发现结构”** 。**主成分分析（PCA）** 正是这个领域最经典、最基础的算法，由 Karl Pearson 于 1901 年提出，迄今已逾百年，却依然是数据预处理和探索性分析中最常用的工具之一。

PCA 的核心思想极为简洁：**找到数据中方差最大的几个正交方向，将数据投影到这些方向上，用更少的维度保留尽可能多的信息**。这一直觉背后有着完整的数学支撑 —— 从**方差最大化**与**最小化重构误差**两个等价视角出发，通过**拉格朗日乘子法**可以严格推导出：**主成分方向即为数据协方差矩阵的特征向量，主成分方差即为对应的特征值**。在实现层面，PCA 既可以通过对协方差矩阵做**特征值分解（EVD）** 完成，也可以通过**奇异值分解（SVD）** 更高效地实现——后者在“样本数远小于特征数”的高维场景下优势尤为明显。

本文还将讨论**累计方差贡献率**、**碎石图肘部法则**等选择主成分数量的实用方法，并明确指出 **PCA 的适用边界** —— 它对**特征尺度敏感**（标准化是必修课）、假设数据关系是**线性**的、对**离群点**敏感，这些局限决定了它并非万能。

> [!note]
> 
> 读完本文，你将掌握 PCA 从几何直觉到数学推导再到工程实现的完整链路。作为**第三阶段：无监督与概率模型**的起点，PCA 将为我们建立“**从数据自身结构出发**”的思维方式。
> 
> 下一篇文章，我们将转向聚类问题——**K-Means** 与 PCA 在目标上不同（**聚类 vs 降维**），但二者在“**利用数据协方差结构**”这一底层逻辑上有着深刻的联系。

---

## 一、PCA的降维直觉

### 1.1 主成分的含义

假设我们有一堆**二维数据点**，它们大致沿着**某个方向拉伸**（比如一条斜线）。如果我们只能用**一个数**来描述每个点，应该选哪个方向？

直觉上，我们应该选**数据变化最大的那个方向** —— 也就是**数据点投影后方差最大的方向**。因为方差越大，说明数据在这个方向上越“**分散**”，保留的信息就**越多**。如果选了一个方差很小的方向，所有点投影后都挤在一起，信息就丢失了。

**主成分（Principal Component）就是数据中方差最大的方向**。**第一个**主成分是**方差最大**的方向，**第二个**主成分是**与第一个正交的、方差次大**的方向，依此类推。

> [!tip] 一个简单的实例
> 
> 想象二维平面上的数据点呈**椭圆形分布**（长轴沿45°方向）。PCA会找到两个主成分：
> 
> - **第一主成分（PC1）** ：沿椭圆**长轴**方向——数据在这个方向上**变化最大**
> - **第二主成分（PC2）** ：沿椭圆**短轴**方向——**与 PC1 正交，变化次大**
> 
> 如果把数据从2维降到1维，就**保留 PC1，丢掉 PC2**。虽然丢掉了一些信息（短轴方向的变化），但**损失最小** —— 因为在所有可能的1维投影中，**PC1 方向的投影保留了最多的方差**。

### 1.2 两个等价视角

PCA可以从两个**完全等价**的角度来理解，两者都指向同一个数学结论。

**视角一：方差最大化**

**1. 核心思想** —— 将数据**投影**到某个方向上，使得**投影后的数据方差最大**。

**方差最大就代表信息保留最**多 —— **方差越大，数据点在这个方向上的散布越广，区分度越高**。如果一个方向上所有数据点的投影都挤在一起（方差小），那么这个方向就**几乎没有区分能力**。

**2. 数学表述** —— 寻找一个**单位向量** $u_1$（$\|u_1\| = 1$），使得数据 $X$ 在 $u_1$ 方向上的**投影方差最大**。

数据点 $x_i$ 在 $u$ 方向上的**投影为 $u^T x_i$**（假设数据已中心化，均值为0）。投影的方差为：

$$
\text{Var}(u^T X) = \frac{1}{n} \sum_{i=1}^{n} (u^T x_i)^2 = u^T \left( \frac{1}{n} \sum_{i=1}^{n} x_i x_i^T \right) u = u^T \Sigma u
$$

其中 $\Sigma = \frac{1}{n} \sum_{i=1}^{n} x_i x_i^T$ 是数据的**协方差矩阵**。

于是问题转化为：

$$
\max_{u} \; u^T \Sigma u, \quad \text{s.t.} \quad u^T u = 1
$$

**视角二：投影距离最小化**

**1. 核心思想** —— 将数据**投影**到某个**低维子空间**上，使得**原始数据点到投影点的距离平方和最小**。

这个视角更接近“**有损压缩**”的直觉 —— **用一个低维的“近似”来代表高维的原始数据，让“近似误差”尽可能小**。

**2. 数学表述** —— 寻找一个 $d$ 维**子空间**（$d < p$），使得**所有数据点到该子空间的垂直距离平方和最小**。

> [!note] 两个视角的等价性
> 
> - **方差最大化**：投影后的点**散布得越开越好** → **信息保留越多**
> - **距离最小化**：投影后的点**离原始点越近越好** → **信息损失越少**
> 
> **毕达哥拉斯定理**告诉我们：对于每个数据点，**原始点到原点的距离平方 = 投影点到原点的距离平方 + 原始点到投影点的垂直距离平方**。
> 
> 当数据已中心化时，“投影点到原点的距离平方”就是**投影的方差**，“原始点到投影点的垂直距离平方”就是**重构误差**。两者之和为常数，因此**最大化方差等价于最小化重构误差**。

---

## 二、数学推导

### 2.1 符号约定与数据预处理

设我们有 $ n $ 个**样本**，每个样本有 $ p $ 个**特征**。**数据矩阵** $ X $ 为 $ n \times p $ 的矩阵，每一行是一个样本，每一列是一个特征。

PCA对数据的**绝对位置**不敏感，只关心**相对变化**。因此，首先对每个特征进行**中心化 —— 减去该特征的均值**：$\tilde{X} = X - \bar{X}$，其中 $\bar{X}$ 是 $ p $ 维**均值向量**。

**中心化后，数据的均值为零**。如果不中心化，**第一主成分会被数据的均值方向“带偏”**，无法正确反映数据的**真实变化方向**。

### 2.2 寻找第一主成分

假设要找一个**单位向量** $ w_1 $（$\|w_1\| = 1$），将数据点**投影**到 $ w_1 $ 方向上。投影后的值为：$z_i = \tilde{x}_i^T w_1$

所有样本投影后的方差为：

$$
\text{Var}(z) = \frac{1}{n} \sum_{i=1}^{n} (\tilde{x}_i^T w_1)^2 = \frac{1}{n} w_1^T \left( \sum_{i=1}^{n} \tilde{x}_i \tilde{x}_i^T \right) w_1
$$

注意到 $\sum_{i=1}^{n} \tilde{x}_i \tilde{x}_i^T$ 正是**协方差矩阵**（乘以 $ n $）：

$$
\Sigma = \frac{1}{n} \sum_{i=1}^{n} \tilde{x}_i \tilde{x}_i^T = \frac{1}{n} \tilde{X}^T \tilde{X}
$$

因此，投影方差为：

$$
\text{Var}(z) = w_1^T \Sigma w_1
$$

目标是**在 $\|w_1\| = 1$ 的约束下，最大化 $ w_1^T \Sigma w_1 $**。这是一个**带约束的优化问题**。构造拉格朗日函数：

$$
\mathcal{L}(w_1, \lambda) = w_1^T \Sigma w_1 - \lambda (w_1^T w_1 - 1)
$$

对 $ w_1 $ 求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial w_1} = 2\Sigma w_1 - 2\lambda w_1 = 0
$$

得到：

$$
\boxed{\Sigma w_1 = \lambda w_1}
$$

这正是**特征方程**。其中 $ \lambda $ 是**特征值**，$ w_1 $ 是**特征向量**。由此可以得到结论：

- **第一主成分的方向**，就是**协方差矩阵 $\Sigma$ 的最大特征值对应的特征向量**。
- **第一主成分的方差**，就等于该**最大特征值**。

### 2.3 所有主成分

推导第二个主成分：在 $ w_2 \perp w_1 $ 的约束下最大化 $ w_2^T \Sigma w_2 $。同样用拉格朗日乘子法，可以得到：$\Sigma w_2 = \lambda_2 w_2$

也就是说，**第 $ k $ 个主成分就是协方差矩阵的第 $ k $ 大特征值对应的特征向量**。

如果我们对协方差矩阵 $\Sigma$ 做**特征值分解**：

$$
\Sigma = W \Lambda W^T
$$

其中：
- $ \Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_p) $ 是**对角矩阵**，$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p \geq 0$
- $ W = [w_1, w_2, ..., w_p] $ 是**正交矩阵**，每一列是一个特征向量

那么有如下结论：
- **第 $ k $ 个主成分的方向 = $ w_k $**
- **第 $ k $ 个主成分的方差 = $ \lambda_k $**
- 所有主成分**两两正交**（因为实对称矩阵的特征向量相互正交）

> [!note] 一个重要且优美的性质
> 
> $$ \sum_{k=1}^{p} \lambda_k = \sum_{j=1}^{p} \text{Var}(x_j) $$
> 
> 也就是说，**所有特征值之和等于原始数据所有特征的方差之和**。
> 
> 这意味着**特征值衡量了每个主成分捕获了多少总方差；且第 $ k $ 个主成分的方差贡献率 = $\lambda_k / \sum_{j=1}^{p} \lambda_j$**

---

## 三、PCA的标准算法流程

**步骤1：数据标准化（中心化 + 缩放）**

**将每个特征减去其均值（中心化），再除以标准差（缩放到单位方差）**：

$$
x_{ij}^{(标准化)} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

**标准化的原因**：如果特征的**量纲**不同，**量纲大的特征会主导协方差矩阵**，导致PCA结果偏向于**量纲大**的特征。**标准化让所有特征在同一尺度上公平竞争**。

**步骤2：计算协方差矩阵**

使用 $n-1$ 是样本**协方差的无偏估计**；也有实现使用 $n$，差异在常数倍上，不影响特征向量的方向。

$$
\Sigma = \frac{1}{n-1} X^T X
$$

**步骤3：特征分解**

求解**特征方程** $\Sigma u = \lambda u$，得到**特征值** $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p$ 和对应的**特征向量** $u_1, u_2, ..., u_p$。

**步骤4：选择主成分**

计算**累计方差贡献率**：

$$
\text{累计贡献率}(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}
$$

选择**最小**的 $k$ 使得**累计贡献率达到某个阈值**（通常为85%或95%）。

**步骤5：投影降维**

取前 $k$ 个特征向量组成**投影矩阵** $U_k = [u_1, u_2, ..., u_k]$，将原始数据**投影到主成分空间**：

$$
Y = U_k^T X
$$

其中 $Y$ 是 $k \times n$ 的**降维后数据**。

**代码实现**：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成二维数据（椭圆形分布）
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=[[3, 1.5]], random_state=42)
X = X @ np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], 
                  [np.sin(np.pi/4), np.cos(np.pi/4)]])

# 中心化
X_centered = X - np.mean(X, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(X_centered, rowvar=False)

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# 按特征值降序排列
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")
print(f"总方差 = {np.sum(eigenvalues):.4f}")
print(f"PC1方差贡献率 = {eigenvalues[0] / np.sum(eigenvalues):.4f}")

# 可视化
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
# 绘制主成分方向（箭头）
for i in range(2):
    # 箭头长度 = 2 * sqrt(特征值)
    length = 2 * np.sqrt(eigenvalues[i])
    plt.arrow(0, 0, eigenvectors[0, i] * length, eigenvectors[1, i] * length,
              head_width=0.2, head_length=0.2, 
              color='red' if i == 0 else 'blue',
              label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.axis('equal')
plt.legend()
plt.title('PCA主成分方向（椭圆数据）')
plt.show()
```

---

## 四、SVD视角：更高效的PCA实现

### 4.1 SVD的引入

第3节通过**特征值分解（Eigenvalue Decomposition, EVD）** 实现了PCA：**先计算协方差矩阵 $\Sigma = \frac{1}{n}X^T X$，再对 $\Sigma$ 做特征值分解**。

但当**特征维度 $ p $ 很大**时（比如图像数据，$ p = 10000 $），协方差矩阵 $ \Sigma $ 是 $ 10000 \times 10000 $ 的矩阵，计算特征值分解的计算量巨大。此时，**奇异值分解（Singular Value Decomposition, SVD）** 提供了一个更高效的替代方案。

### 4.2 SVD与PCA的数学关系

对**中心化后**的数据矩阵 $\tilde{X}$（$ n \times p $）做**SVD**：

$$
\tilde{X} = U \Sigma_{svd} V^T
$$

其中：
- $ U $ 是 $ n \times n $ 的**正交矩阵（左奇异向量）**
- $ \Sigma_{svd} $ 是 $ n \times p $ 的**对角矩阵（奇异值）**
- $ V $ 是 $ p \times p $ 的**正交矩阵（右奇异向量）**

现在计算**协方差矩阵**：

$$
\tilde{X}^T \tilde{X} = (U \Sigma_{svd} V^T)^T (U \Sigma_{svd} V^T) = V \Sigma_{svd}^T \Sigma_{svd} V^T
$$

由于 $\Sigma_{svd}^T \Sigma_{svd}$ 是**对角矩阵**，其对角元素为**奇异值的平方** $\sigma_i^2$。

因此：

$$
\tilde{X}^T \tilde{X} = V \cdot \text{diag}(\sigma_1^2, \sigma_2^2, ..., \sigma_p^2) \cdot V^T
$$

> [!important] 关键结论
> 
> - PCA的**特征向量（主成分方向）**就是SVD的**右奇异向量 $ V $ 的列向量**。
> - PCA的**特征值**等于**SVD奇异值的平方 $\sigma_i^2$ 除以 $n$**。
> 
> 即**主成分方向 = $ V $ 的列；特征值 $ \lambda_i = \sigma_i^2 / n $**

### 4.3 EVD vs SVD

| 对比维度 | EVD方法 | SVD方法 |
|---------|---------|---------|
| **需要计算的矩阵** | **协方差矩阵** $ \Sigma $（$ p \times p $） | **直接分解数据矩阵** $ X $（$ n \times p $） |
| **计算复杂度** | $ O(p^3) $ | $ O(np^2) $（当 $ n < p $ 时更优） |
| **数值稳定性** | 一般 | **更好**（避免显式计算协方差矩阵） |
| **适用场景** | $ p $ 较小时 | **$ p $ 很大时优势明显** |

当 **样本数 $ n $ 远小于特征数 $ p $** 时（如基因数据：$ n=200, p=20000 $），**SVD的优势尤其明显**——直接分解 $ 200 \times 20000 $ 的矩阵，远比先算 $ 20000 \times 20000 $ 的协方差矩阵再分解要高效得多。

**代码实现**：

```python
def pca_via_evd(X, n_components=None):
    """通过特征值分解实现PCA"""
    # 中心化
    X_centered = X - np.mean(X, axis=0)
    # 协方差矩阵
    cov = np.cov(X_centered, rowvar=False)
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 按特征值降序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if n_components:
        eigvecs = eigvecs[:, :n_components]
    # 投影
    X_pca = X_centered @ eigvecs
    return X_pca, eigvals, eigvecs

def pca_via_svd(X, n_components=None):
    """通过奇异值分解实现PCA（更高效）"""
    # 中心化
    X_centered = X - np.mean(X, axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt 的每一行是一个右奇异向量（即主成分方向）
    components = Vt[:n_components].T if n_components else Vt.T
    # 投影
    X_pca = X_centered @ components
    # 特征值 = 奇异值^2 / (n-1)
    eigvals = (S[:n_components] ** 2) / (X.shape[0] - 1) if n_components else (S ** 2) / (X.shape[0] - 1)
    return X_pca, eigvals, components

# 验证两种方法等价
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

X_pca_evd, evals_evd, evecs_evd = pca_via_evd(X, n_components=2)
X_pca_svd, evals_svd, evecs_svd = pca_via_svd(X, n_components=2)

print("EVD vs SVD 投影结果是否一致:", np.allclose(np.abs(X_pca_evd), np.abs(X_pca_svd), atol=1e-10))
print("EVD特征值:", evals_evd)
print("SVD特征值:", evals_svd)
```

---

## 五、选择主成分数量的方法

PCA降维需要决定**保留多少个主成分**。这个问题没有唯一的答案，但有几种常用的启发式方法。

### 5.1 累计方差贡献率法

这是最常用的方法。**计算前 $ k $ 个主成分的累计方差贡献率**：

$$
\text{Cumulative Ratio}(k) = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}
$$

通常选择一个阈值（如 **80%、85%或95%** ），**选择使累计贡献率达到该阈值的最小 $k$**。

### 5.2 碎石图（Scree Plot）与肘部法则

绘制**特征值从大到小**的分布图（**碎石图**），寻找“**肘部（Elbow）**” —— 即**特征值开始急剧下降后趋于平缓的位置**。**肘部之前的成分包含了大部分信息**，之后的成分贡献很小可以舍弃。

**肘部法则的局限**：与**K-Means**中的肘部法则类似，这种方法**主观性较强**，且在**高维噪声数据**中肘部可能**不明显**。

### 5.3 Kaiser准则

**只保留特征值大于1的主成分**。

这个准则基于一个直觉：每个主成分至少应该解释**一个原始变量的方差**（原始变量方差为1，前提是数据已经标准化）。

### 5.4 实际建议

在实践中，**累计方差贡献率法**最常用。对于探索性数据分析，80%-90%通常就足够了；对于后续的机器学习任务，建议用**交叉验证**来评估不同 $ k $ 对下游任务的影响。

**代码实现**：

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 加载手写数字数据（64维）
digits = load_digits()
X = digits.data

# 使用sklearn的PCA
pca = PCA()
pca.fit(X)

# 计算累计方差贡献率
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 碎石图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_)
plt.xlabel('主成分编号')
plt.ylabel('方差贡献率')
plt.title('碎石图（Scree Plot）')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.axhline(y=0.85, color='g', linestyle='--', label='85%阈值')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90%阈值')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
plt.xlabel('主成分数量')
plt.ylabel('累计方差贡献率')
plt.legend()
plt.title('累计方差贡献率')

plt.tight_layout()
plt.show()

# 计算达到不同阈值所需的主成分数
for threshold in [0.80, 0.85, 0.90, 0.95]:
    n = np.argmax(cumulative_variance >= threshold) + 1
    print(f"达到 {threshold*100:.0f}% 方差需要 {n} 个主成分")
```

---

## 六、PCA的假设与局限

### 6.1 PCA的三个关键假设

**假设一：线性关系。** PCA寻找的是**线性**的主成分方向。如果数据中存在**复杂的非线性结构**（如环形、流形），**PCA无法有效捕捉**。

**假设二：方差代表信息。** PCA认为**方差越大的方向越重要**。但如果数据中**存在噪声**，方差大的方向可能只是**噪声**的方向，而不是信号的方向。

**假设三：特征相互独立（在降维后）。** PCA的目标是找到一组**线性无关的主成分**，这意味着它假设**原始特征之间存在线性相关性**（这正是PCA想要去除的）。

### 6.2 PCA的常见局限

**局限一：对特征尺度敏感。** PCA是基于**协方差矩阵**的，如果不同特征的**量纲差异很大**，**量纲大**的特征会**主导**主成分方向。**解决方案**即为在PCA之前对数据进行**标准化（Standardization）** ——使每个特征**均值为0、方差为1**。

**局限二：主成分难以解释。** 主成分是原始特征的**线性组合**，往往不具有直接的物理或业务含义。

**局限三：对离群点敏感。** 方差对**离群点**非常敏感，一个**极端值**可能会严重**扭曲**主成分方向。

**局限四：假设数据服从高斯分布（在协方差矩阵的意义上）。** PCA只利用了数据的**二阶统计量（均值和协方差）**，忽略了更高阶的统计信息。

---

## 七、完整代码实现：从鸢尾花数据到二维可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

print(f"原始数据维度: {X.shape}")  # (150, 4)

# 2. 标准化（对PCA至关重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用sklearn的PCA（使用SVD实现）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"降维后维度: {X_pca.shape}")  # (150, 2)
print(f"PC1方差贡献率: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2方差贡献率: {pca.explained_variance_ratio_[1]:.4f}")
print(f"累计方差贡献率: {np.sum(pca.explained_variance_ratio_):.4f}")

# 4. 可视化降维结果
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
target_names = iris.target_names
for i, color, name in zip(range(3), colors, target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, label=name, alpha=0.7)

plt.xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('鸢尾花数据 PCA 降维可视化')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. 查看主成分的"含义"（载荷）
print("\n主成分载荷（特征向量）:")
for i, comp in enumerate(pca.components_):
    print(f"PC{i+1}:")
    for j, name in enumerate(feature_names):
        print(f"  {name}: {comp[j]:.4f}")
```

---

> [!note] 总结
> 
> | 概念 | 核心内容 |
> |------|---------|
> | **PCA的目标** | 找到数据中**方差最大**的几个**正交方向**，**用更少的维度保留尽可能多的信息** |
> | **数学核心** | 对**协方差矩阵** $\Sigma = \frac{1}{n}X^T X$ 做**特征值分解**，**特征向量 = 主成分方向，特征值 = 主成分方差** |
> | **SVD实现** | 对中心化数据矩阵做**SVD**，**右奇异向量 = 主成分方向，奇异值²/n = 特征值**，**更高效、更稳定** |
> | **主成分选择** | **累计方差贡献率法**（80%-95%）、碎石图肘部法则、Kaiser准则（特征值>1） |
> | **关键假设** | 数据中的关系是**线性的**，**方差代表信息** |
> | **重要提醒** | PCA对**特征尺度敏感**，使用前必须**标准化** |
> 
> 核心要点回顾
> 
> 1. **PCA的本质**是找到数据中**方差最大的方向**。**第一主成分是协方差矩阵最大特征值对应的特征向量，其方差等于该特征值。所有主成分两两正交**。
> 2. **EVD vs SVD**：EVD方法**先算协方差矩阵再分解**，适合**特征数较少**的情况；SVD方法**直接分解数据矩阵**，当**样本数远小于特征数**时**效率更高、数值更稳定**。
> 3. **主成分的数量**通常通过**累计方差贡献率**来选择——达到80%-95%即可。**碎石图的“肘部”**也是一个直观的参考。
> 4. **标准化是PCA的必修课**。PCA基于**协方差矩阵**，**量纲不同**的特征会扭曲主成分方向。使用PCA前，务必**对每个特征进行标准化（均值为0、方差为1）**。
> 5. **PCA的局限**：它是**线性**的，无法捕捉**非线性**结构；对**离群点**敏感；主成分往往难以解释。在这些场景下，可以考虑**核PCA（Kernel PCA）** 等非线性降维方法。