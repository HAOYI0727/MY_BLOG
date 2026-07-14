---
title: K-Nearest Neighbor (KNN) —— K-近邻
published: 2025-07-10
description: 全面剖析 KNN 算法的核心机制、KD-Tree 与 Ball-Tree 的加速原理与适用场景，深入揭示维度灾难导致高维欧氏距离失效的数学根源，并强调特征标准化对 KNN 的必要性。
cover: "/assets/images/posts/knn.png"
coverInContent: false
tags: [KNN, KD-Tree, Ball-Tree, 机器学习]
category: Machine_Learning
draft: false
---

# K-Nearest Neighbor (KNN) —— K-近邻

## 前言

在前面几篇文章中，我们讨论了线性回归、逻辑回归、支持向量机和核方法。这些算法都有一个共同点：**它们都是“急切”的学习者**——在训练阶段，它们会从数据中学习一个明确的模型（比如线性函数的系数、SVM的支持向量），然后在预测阶段直接用这个模型做判断。

**K-近邻（K-Nearest Neighbor, KNN）** 则完全不同。它是最著名的 **“懒惰学习”（Lazy Learning）** 算法——训练阶段几乎什么都不做，只是把数据存下来；等到需要预测新样本时，才去训练数据中“现场”找最近的几个邻居，让邻居们“投票”决定结果。

KNN 由 Cover 和 Hart 于 1968 年提出，是机器学习中最古老也最直观的算法之一。它的思想简单到可以用一句话概括：**“物以类聚，人以群分”**。

这篇文章，我们将从 KNN 的投票与回归机制出发，深入探讨 KD-Tree 与 Ball-Tree 的加速原理，理解维度灾难如何从根本上破坏欧氏距离的有效性，最后讨论为什么特征标准化对 KNN 来说不是可选项而是必选项。

---

## 一、KNN 的核心机制：投票与回归

### 1.1 算法思想

KNN 的工作机制极其简单：给定一个测试样本，在训练集中找出与它**距离最近**的 $k$ 个样本（即 $k$ 个邻居），然后根据这些邻居的信息来做预测。

算法的伪代码如下：

```
对每个待预测的样本点：
    1. 计算它与训练集中所有样本点的距离
    2. 按距离从小到大排序
    3. 选取距离最小的 k 个点
    4. 根据这 k 个点的标签做决策
```

### 1.2 分类任务：投票法

对于**分类问题**，KNN 使用**投票法（Voting）** ——统计 $k$ 个邻居中每个类别出现的次数，将出现次数最多的类别作为预测结果。

数学上，测试样本 $x$ 的预测类别为：

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = c)
$$

其中 $\mathcal{N}_k(x)$ 是 $x$ 的 $k$ 个最近邻的集合，$\mathbb{I}(\cdot)$ 是指示函数。

**K 值的选择至关重要**：
- **K 值过小**（如 K=1）：模型对噪声敏感，容易**过拟合**——一个异常点就可能改变预测结果
- **K 值过大**：模型过于平滑，容易**欠拟合**——远距离的样本也会影响预测
- **极端情况 K=N**：无论输入什么，都预测为训练集中最多的类别，模型完全失效

### 1.3 回归任务：平均法与加权平均法

对于**回归问题**，KNN 使用**平均法（Averaging）** ——计算 $k$ 个邻居的输出值的平均值作为预测结果：

$$
\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
$$

但更精细的做法是**加权平均**——距离越近的邻居权重越大：

$$
\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i \cdot y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}
$$

其中权重 $w_i$ 通常取距离的倒数：$w_i = 1 / d(x, x_i)$。

```python
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

class KNN:
    """从头实现 K-近邻算法（支持分类与回归）"""
    
    def __init__(self, k=5, weights='uniform'):
        """
        k: 邻居数量
        weights: 'uniform' 等权投票/平均, 'distance' 距离加权
        """
        self.k = k
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """KNN 是懒惰学习——训练阶段只存储数据"""
        self.X_train = X
        self.y_train = y
        return self
    
    def _predict_one(self, x):
        """预测单个样本"""
        # 计算到所有训练样本的欧氏距离
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # 获取 k 个最近邻的索引
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        k_labels = self.y_train[k_indices]
        
        # 判断是分类还是回归
        if self.y_train.dtype == np.float64 or self.y_train.dtype == np.float32:
            # 回归：平均或加权平均
            if self.weights == 'uniform':
                return np.mean(k_labels)
            else:
                # 距离加权（加一个小 epsilon 防止除零）
                weights = 1.0 / (k_distances + 1e-8)
                return np.average(k_labels, weights=weights)
        else:
            # 分类：投票或加权投票
            if self.weights == 'uniform':
                counter = Counter(k_labels)
                return counter.most_common(1)[0][0]
            else:
                # 距离加权投票
                weights = 1.0 / (k_distances + 1e-8)
                weight_dict = {}
                for label, w in zip(k_labels, weights):
                    weight_dict[label] = weight_dict.get(label, 0) + w
                return max(weight_dict, key=weight_dict.get)
    
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

# ============ 分类示例 ============
X_clf, y_clf = make_classification(n_samples=300, n_features=4, 
                                    n_informative=3, n_redundant=1,
                                    n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3)

# 标准化（后面会解释为什么这是必须的）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNN(k=5, weights='distance')
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print(f"分类准确率: {accuracy_score(y_test, y_pred):.4f}")

# ============ 回归示例 ============
X_reg, y_reg = make_regression(n_samples=300, n_features=4, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_reg = KNN(k=5, weights='distance')
knn_reg.fit(X_train_scaled, y_train)
y_pred = knn_reg.predict(X_test_scaled)
print(f"回归 MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

### 1.4 距离度量

KNN 的核心操作是计算“距离”。最常用的是**欧氏距离（Euclidean Distance）** ：

$$
d(x, z) = \sqrt{\sum_{j=1}^{p} (x_j - z_j)^2}
$$

其中 $p$ 是特征的维度。

除此之外，曼哈顿距离、切比雪夫距离、马氏距离等也各有适用场景。

---

## 二、KD-Tree 与 Ball-Tree：加速 KNN 搜索

### 2.1 暴力法的困境

最朴素的 KNN 实现是**暴力法（Brute Force）** ：对每个待预测样本，计算它与**所有**训练样本的距离，然后排序取前 $k$ 个。

这种方法的时间复杂度是 $O(n)$，其中 $n$ 是训练样本的数量。当数据集很大时（比如百万级样本），每次预测都要计算百万次距离，这是不可接受的。

### 2.2 KD-Tree：基于超矩形的空间划分

**KD-Tree（K-Dimensional Tree）** 是一种二叉树数据结构，它通过**递归地将空间划分为超矩形（Hyper-rectangles）** 来组织数据点。

**构建过程**：
1. 在当前节点，选择一个维度（通常轮流选择）
2. 在该维度上找到所有数据点的**中位数**
3. 用中位数将数据分成两半：小于中位数的放左子树，大于的放右子树
4. 递归地在左右子树上重复上述过程

**搜索过程**：
1. 从根节点开始，根据目标点在当前维度的值，决定进入左子树还是右子树
2. 到达叶子节点后，计算该节点中样本与目标点的距离，作为当前“最佳”
3. **回溯**到父节点，检查父节点的另一个子树是否可能存在更近的点（通过计算目标点到该子树对应超矩形的最小距离）
4. 如果可能，进入该子树搜索；否则剪枝

**KD-Tree 的优势**在于：它不需要计算目标点到所有样本的距离，而是通过**剪枝**跳过了大量不可能成为最近邻的区域。

**KD-Tree 的局限性**：随着维度 $D$ 的增加，KD-Tree 的性能会急剧下降。当维度较高时（通常认为 $D > 20$），KD-Tree 的效率会退化到接近暴力法。这是因为在高维空间中，数据点变得极其稀疏，超矩形之间的边界模糊，剪枝效果大打折扣。

### 2.3 Ball-Tree：基于超球体的空间划分

**Ball-Tree** 是对 KD-Tree 的改进，它使用**超球体（Hyperspheres）** 而不是超矩形来划分空间。

**构建过程**：
1. 选择一个数据点作为球心
2. 找到距离球心最远的点，用它们确定一个最小包围球
3. 将球内的点分配到两个子球中
4. 递归构建

**搜索过程**与 KD-Tree 类似，但剪枝条件变为：如果目标点到某个球心的距离**减去**该球的半径，仍然大于当前找到的最佳距离，则整个球体都可以被剪枝。

**Ball-Tree 的优势**：
- 在高维空间中**表现优于 KD-Tree**
- 能更好地处理**非均匀分布**的数据
- 球体划分比矩形划分更适应高维空间的几何特性

**KD-Tree vs Ball-Tree 总结**：

| 特性 | KD-Tree | Ball-Tree |
|------|---------|-----------|
| 划分形状 | 超矩形 | 超球体 |
| 低维表现 | 优秀 | 良好 |
| 高维表现（D>20） | 显著下降 | 相对较好 |
| 适用场景 | 低维到中维数据 | 高维数据、非均匀分布 |

```python
from sklearn.neighbors import KNeighborsClassifier
import time

# 生成不同维度的数据
for dim in [2, 5, 10, 20, 50]:
    X, y = make_classification(n_samples=5000, n_features=dim, 
                                n_informative=dim, n_redundant=0,
                                n_classes=2, random_state=42)
    X_train, X_test = X[:4000], X[4000:]
    y_train, y_test = y[:4000], y[4000:]
    
    # KD-Tree
    start = time.time()
    knn_kd = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    knn_kd.fit(X_train, y_train)
    knn_kd.predict(X_test)
    kd_time = time.time() - start
    
    # Ball-Tree
    start = time.time()
    knn_ball = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    knn_ball.fit(X_train, y_train)
    knn_ball.predict(X_test)
    ball_time = time.time() - start
    
    print(f"维度 {dim:2d}: KD-Tree {kd_time:.4f}s, Ball-Tree {ball_time:.4f}s")
# 输出会显示：低维时 KD-Tree 更快，高维时 Ball-Tree 优势明显
```

---

## 三、维度灾难：欧氏距离的失效

### 3.1 什么是维度灾难？

**维度灾难（Curse of Dimensionality）** 是指：随着特征维度的增加，数据在高维空间中的性质会发生根本性的变化，导致许多在低维空间有效的算法在高维空间中失效。

对于 KNN 来说，维度灾难的影响尤为致命。

### 3.2 为什么高维空间中欧氏距离失效？

在低维空间中（如二维平面），我们的直觉是：每个点都有一些“近”的点和一些“远”的点，“最近邻”的概念是有意义的。

但在高维空间中，情况完全不同。**在高维空间中，所有点到查询点的距离几乎都相等**。

我们可以用数学来理解这个现象。

假设数据点均匀分布在一个 $D$ 维单位超立方体 $[0,1]^D$ 中。对于一个查询点（比如在原点），一个随机数据点到查询点的距离平方为：

$$
d^2 = \sum_{j=1}^{D} x_j^2
$$

由于 $x_j \sim \text{Uniform}(0,1)$，$E[x_j^2] = 1/3$，方差为 $4/45$。因此：

$$
E[d^2] = \frac{D}{3}, \quad \text{Var}(d^2) = \frac{4D}{45}
$$

距离的标准差与期望之比为：

$$
\frac{\sqrt{\text{Var}(d^2)}}{E[d^2]} = \frac{\sqrt{4D/45}}{D/3} = \frac{3}{\sqrt{45}} \cdot \frac{1}{\sqrt{D}} \approx \frac{0.447}{\sqrt{D}}
$$

**关键结论**：随着维度 $D$ 增大，距离的相对标准差**趋于 0**。这意味着在高维空间中，**所有点到查询点的距离几乎都相等**。

当所有距离都差不多时，“最近邻”与“最远邻”的区分度消失了。KNN 赖以生存的“邻居”概念变得毫无意义。

### 3.3 维度灾难的实际影响

**样本需求的指数级增长**：为了在高维空间中保持同样的样本密度，所需的样本数量随维度**指数增长**。如果在一维空间中需要 10 个样本才能覆盖一个区间，那么在 10 维空间中就需要 $10^{10}$ 个样本才能达到同样的密度。

**搜索效率的崩溃**：如前面所述，KD-Tree 在高维空间中效率急剧下降。即使使用 Ball-Tree，也只是“缓解”而非“解决”这个问题。

**距离度量的选择**：有研究表明，在高维特征空间中，使用**曼哈顿距离（L1 范数）** 比欧氏距离（L2 范数）更能抵抗维度灾难的影响。这是因为 L1 范数对各个维度的“贡献”是线性的，而 L2 范数是平方的，会进一步放大维度增加带来的效应。

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_curse_of_dimensionality():
    """演示高维空间中距离分布的趋同现象"""
    np.random.seed(42)
    dimensions = [1, 2, 5, 10, 20, 50, 100]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, D in enumerate(dimensions):
        # 在 D 维单位超立方体中生成 1000 个点
        points = np.random.rand(1000, D)
        # 查询点在原点
        query = np.zeros(D)
        # 计算所有点到原点的距离
        distances = np.linalg.norm(points - query, axis=1)
        
        axes[idx].hist(distances, bins=30, alpha=0.7)
        axes[idx].set_title(f'D={D}')
        axes[idx].set_xlabel('Distance')
        axes[idx].set_ylabel('Frequency')
        # 标注均值和标准差
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        axes[idx].axvline(mean_d, color='red', linestyle='--', 
                          label=f'μ={mean_d:.2f}')
        axes[idx].axvline(mean_d - std_d, color='green', linestyle=':')
        axes[idx].axvline(mean_d + std_d, color='green', linestyle=':')
        axes[idx].legend()
    
    plt.suptitle('高维空间中距离分布的趋同现象（维度灾难）', fontsize=14)
    plt.tight_layout()
    plt.show()
    # 观察：随着维度增加，距离分布越来越集中（标准差相对于均值越来越小）

demonstrate_curse_of_dimensionality()
```

---

## 四、特征标准化：KNN 的必修课

### 4.1 为什么 KNN 必须做特征标准化？

KNN 是基于**距离**的算法。如果不同特征的量纲（scale）不同，那么**取值范围大的特征会主导距离计算**，而取值范围小的特征几乎不起作用。

举个例子：假设我们要预测房价，有两个特征：
- **房屋面积**：50-500 平方米
- **卧室数量**：1-5 间

如果直接用原始数据计算欧氏距离，面积特征（范围 450）的差异会完全淹没卧室数量（范围 4）的差异。一个面积相差 100 平方米的房子，和卧室数量相差 2 间的房子，在距离计算中前者会被视为“更远”——但这可能完全不符合实际。

### 4.2 标准化 vs 归一化

**标准化（Standardization）** ：

$$
x' = \frac{x - \mu}{\sigma}
$$

将数据转换为**均值为 0、标准差为 1** 的分布。

**归一化（Normalization）** ：

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

将数据缩放到 **[0, 1]** 区间。

**对于 KNN，标准化通常是更好的选择**，因为：
- 标准化对异常值**不那么敏感**（归一化受最大最小值影响大）
- 标准化保留了数据的分布形状
- 当数据中存在异常值时，标准化的鲁棒性更好

### 4.3 不做标准化的后果

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 生成具有不同量纲特征的数据
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, 
                            n_informative=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)
# 人为制造量纲差异：第一个特征放大100倍
X[:, 0] = X[:, 0] * 100

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 不做标准化
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))

# 做标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))

print(f"未标准化准确率: {acc_raw:.4f}")
print(f"标准化后准确率: {acc_scaled:.4f}")
# 标准化后的准确率通常显著高于未标准化的版本
```

---

## 五、总结

| 概念 | 核心内容 |
|------|---------|
| **KNN 分类** | 投票法：$k$ 个邻居中多数类别作为预测 |
| **KNN 回归** | 平均法或加权平均法（权重为距离倒数） |
| **KD-Tree** | 基于超矩形的空间划分，低维高效，高维退化 |
| **Ball-Tree** | 基于超球体的空间划分，高维表现优于 KD-Tree |
| **维度灾难** | 高维空间中所有距离趋于相等，欧氏距离失效 |
| **特征标准化** | 将各特征缩放到同一量纲，对 KNN 是**必须的** |

### 核心要点回顾

1. **KNN 的本质**：KNN 是最经典的“懒惰学习”算法，训练阶段只存储数据，预测阶段才进行计算。分类用投票，回归用平均（或加权平均）。

2. **KD-Tree 与 Ball-Tree**：两者都是通过空间划分来加速 KNN 搜索的数据结构。KD-Tree 用超矩形划分，在低维（D<20）时表现优异；Ball-Tree 用超球体划分，在高维和分布不均匀的数据上更有优势。

3. **维度灾难的数学本质**：在高维空间中，所有点到查询点的欧氏距离几乎相等。“最近邻”与“最远邻”失去区分度，KNN 的预测能力被严重削弱。要维持同样的预测精度，样本数量需要随维度指数增长。

4. **特征标准化是必修课**：KNN 基于距离，量纲不同的特征会扭曲距离计算。标准化（将每个特征转为均值为 0、标准差为 1）是使用 KNN 前的**必要预处理步骤**。

5. **KNN 的适用场景**：KNN 简单直观、无需训练、对异常值不敏感，适合**低维、样本量适中**的数据集。在高维场景下，建议先做**降维**（如 PCA）或考虑使用其他算法。