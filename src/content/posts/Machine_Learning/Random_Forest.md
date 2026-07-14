---
title: Bagging & Random Forest —— 随机森林
published: 2025-07-16
description: 系统讲解集成学习的核心思想，从Bootstrap自助采样的数学原理出发，深入剖析Bagging的方差降低机制、随机森林的双重随机性（行采样+列采样）、OOB误差的理论基础与实用价值，并对比基尼重要性与排列重要性两种特征重要性计算方法。
cover: "/assets/images/posts/random_forest.png"
coverInContent: false
tags: [Bagging, 随机森林, Bootstrap, 集成学习]
category: Machine_Learning
draft: false
---

# Bagging & Random Forest —— 随机森林

## 前言

在上一篇文章中，我们深入探讨了决策树——这个可解释性最强、最贴近人类决策思维的机器学习模型。但决策树有一个致命的弱点：**它太“脆弱”了**——训练数据的微小变化可能导致一棵完全不同的树，这种**高方差（High Variance）** 特性让决策树在测试集上的表现常常不稳定。

那么，有没有办法既保留决策树的优点（可解释性、非线性建模能力），又克服它的缺点（过拟合、高方差）呢？

**集成学习（Ensemble Learning）** 给出了答案：**“众人拾柴火焰高”** ——与其相信一棵树，不如种一片森林。

**Bagging（Bootstrap Aggregating）** 和**随机森林（Random Forest）** 正是集成学习思想最经典的实践。Bagging由Leo Breiman于1996年提出，而随机森林则是Breiman在2001年将Bagging与Ho提出的随机子空间方法相结合的产物。

这篇文章，我们将从Bootstrap自助采样出发，一步步理解Bagging和随机森林的工作原理，深入分析偏差-方差分解如何解释随机森林的成功，最后系统讲解两种特征重要性计算方法及其数学原理。

---

## 一、Bootstrap自助采样：Bagging的基石

### 1.1 什么是Bootstrap？

**Bootstrap（自助法）** 是一种强大的统计方法，用于从有限的样本中估计总体统计量（如均值、方差等）。

假设我们有一个包含 $n$ 个样本的数据集 $D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$。Bootstrap采样的过程是：

1. 从数据集 $D$ 中**有放回地**随机抽取一个样本
2. 将抽取的样本放入Bootstrap样本集中
3. 重复上述步骤 $n$ 次（即抽取与原始数据集相同数量的样本）

这样我们就得到了一个Bootstrap样本集 $D^*$，它同样包含 $n$ 个样本，但**有些原始样本会出现多次，而有些则一次都不会出现**。

### 1.2 为什么约63.2%的样本会被选中？

这是一个经典而优美的数学结果。对于原始数据集中的**任意一个特定样本**，在每次有放回抽样中**不被选中的概率**为：

$$
P(\text{不被选中}) = 1 - \frac{1}{n}
$$

经过 $n$ 次独立抽样后，该样本**从未被选中的概率**为：

$$
P(\text{从未被选中}) = \left(1 - \frac{1}{n}\right)^n
$$

当 $n$ 足够大时，利用极限 $\lim_{n \to \infty} (1 - 1/n)^n = e^{-1}$：

$$
P(\text{从未被选中}) \approx e^{-1} \approx 0.368
$$

因此，该样本**至少被选中一次的概率**为：

$$
1 - e^{-1} \approx 0.632
$$

**结论**：每个Bootstrap样本集平均包含原始数据集约 **63.2%** 的独特样本，剩下的约 **36.8%** 从未出现在该Bootstrap样本中。

这些未被选中的样本被称为**袋外样本（Out-of-Bag, OOB）** 。

```python
import numpy as np
import matplotlib.pyplot as plt

def bootstrap_sampling_demo(n_samples=1000, n_bootstraps=1000):
    """演示Bootstrap采样中样本被选中的比例"""
    # 模拟：对每个Bootstrap样本，统计原始数据集中有多少个样本被选中
    selected_counts = []
    for _ in range(n_bootstraps):
        # 从0到n_samples-1中有放回地抽取n_samples次
        sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        unique_selected = len(np.unique(sampled_indices))
        selected_counts.append(unique_selected / n_samples)
    
    mean_ratio = np.mean(selected_counts)
    print(f"平均选中比例: {mean_ratio:.4f}")
    print(f"理论值 (1 - 1/e): {1 - np.exp(-1):.4f}")
    
    # 可视化
    plt.hist(selected_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(1 - np.exp(-1), color='red', linestyle='--', label='理论值 ≈ 0.632')
    plt.xlabel('被选中的独特样本比例')
    plt.ylabel('频次')
    plt.legend()
    plt.title('Bootstrap采样中独特样本的比例分布')
    plt.show()

bootstrap_sampling_demo()
# 输出: 平均选中比例 ≈ 0.632, 与理论值高度一致
```

---

## 二、Bagging：Bootstrap + 聚合

### 2.1 Bagging的核心思想

**Bagging（Bootstrap Aggregating）** 的核心思想非常简单：

1. 从原始训练集中通过Bootstrap采样生成 $m$ 个不同的训练子集
2. 在每个子集上独立训练一个基学习器（通常是决策树）
3. 对于新样本，将所有基学习器的预测结果进行**聚合**：
   - **分类**：多数投票（Majority Voting）
   - **回归**：简单平均（Averaging）

### 2.2 Bagging为什么有效？

Bagging的关键优势在于**降低方差（Variance Reduction）** 。让我们通过数学来理解这一点。

假设我们有 $m$ 个独立同分布的基学习器 $h_1, h_2, ..., h_m$，每个学习器的预测方差为 $\sigma^2$。它们的平均预测的方差为：

$$
\text{Var}\left(\frac{1}{m}\sum_{i=1}^{m} h_i\right) = \frac{1}{m^2} \sum_{i=1}^{m} \text{Var}(h_i) = \frac{\sigma^2}{m}
$$

**随着 $m$ 增大，方差线性减小！**

但在现实中，Bootstrap样本之间**并非完全独立**（因为它们是**有放回**地从同一数据集中采样的）。如果基学习器之间的相关性为 $\rho$，则平均预测的方差为：

$$
\text{Var}\left(\bar{h}\right) = \rho\sigma^2 + \frac{1-\rho}{m}\sigma^2
$$

当 $m \to \infty$ 时，方差趋近于 $\rho\sigma^2$，而非零。这说明：**基学习器之间的相关性越低，Bagging的方差降低效果越好**。

这正是随机森林在Bagging基础上**进一步引入列采样（特征随机子空间）** 的原因——**降低树之间的相关性**。

对于回归问题的偏差-方差分解，期望预测误差可以分解为：

$$
\underbrace{\mathbb{E}[(h_D(x) - y)^2]}_{\text{误差}} = 
\underbrace{\mathbb{E}[(h_D(x) - \bar{h}(x))^2]}_{\text{方差}} + 
\underbrace{(\bar{h}(x) - \bar{y}(x))^2}_{\text{偏差}} + 
\underbrace{\mathbb{E}[(\bar{y}(x) - y(x))^2]}_{\text{噪声}}
$$

Bagging通过**平均**多个模型来降低**方差项**，而**不增加偏差**。

---

## 三、随机森林：Bagging + 列采样

### 3.1 随机森林的“双重随机性”

随机森林在Bagging的基础上增加了一个关键的改进：**在每次分裂时，不是从所有特征中选择最佳分裂特征，而是从一个随机选择的特征子集中选择**。

这形成了随机森林的**双重随机性**：

| 随机性来源 | 操作 | 目的 |
|-----------|------|------|
| **行采样（Bootstrap）** | 每棵树使用不同的Bootstrap样本集 | 增加数据多样性 |
| **列采样（Feature Subspace）** | 每次分裂时只考虑随机子集的特征 | 降低树间相关性 |

**列采样**也被称为**随机子空间方法（Random Subspace Method）** 或**特征Bagging（Feature Bagging）** 。

在scikit-learn中：
- **行采样**由 `max_samples` 和 `bootstrap` 控制
- **列采样**由 `max_features` 控制

对于分类任务，默认的 `max_features = sqrt(n_features)`；对于回归任务，默认的 `max_features = n_features`。

### 3.2 列采样的数学动机

列采样通过**降低树之间的相关性**来进一步减小方差。

假设特征总数为 $p$，每次分裂时随机选择 $k$ 个特征（$k \ll p$）。如果两个特征高度相关，它们可能在不同的树中被选为分裂特征，从而产生相似的树结构。列采样**强制**不同树使用不同的特征子集，增加了树之间的**多样性**。

Breiman在原始随机森林论文中指出，**森林的泛化误差取决于两个因素**：
1. **任意两棵树之间的相关性**：相关性越低，误差越低
2. **单棵树的强度（Strength）** ：每棵树本身的预测能力

列采样在**降低相关性**（好）的同时可能**略微降低单棵树的强度**（坏），但总体效果是**降低泛化误差**。

### 3.3 从数学上看：随机森林 vs Bagging

最近的研究表明，随机森林不仅降低方差，在某些情况下还能**同时降低偏差**：

> 当数据中存在某些模式时，随机森林能够捕捉到Bagging集成无法捕捉的模式，从而在降低方差的同时也降低偏差。

特别是当**特征之间存在相关性**时，随机森林的效果更为显著。

---

## 四、袋外误差（OOB Error）：免费的验证集

### 4.1 什么是OOB误差？

由于每个Bootstrap样本只包含约63.2%的原始样本，剩下的36.8%是**袋外样本（Out-of-Bag, OOB）** 。

对于第 $i$ 个样本，如果它在第 $t$ 棵树的Bootstrap样本中**从未出现**，那么第 $t$ 棵树就可以用来验证第 $i$ 个样本——**这相当于免费的交叉验证！**

### 4.2 OOB误差的数学定义

对于样本 $x_n$，定义：

$$
G_n^-(x_n) = \text{average}\left( g_{i_1}(x_n), g_{i_2}(x_n), ..., g_{i_T}(x_n) \right)
$$

其中 $i_1, i_2, ..., i_T$ 是那些**没有使用**样本 $x_n$ 进行训练的树的索引。

OOB误差为：

$$
E_{oob}(G) = \frac{1}{N} \sum_{n=1}^{N} \text{err}\left(y_n, G_n^-(x_n)\right)
$$

其中 $\text{err}$ 是损失函数（分类用0-1损失，回归用MSE）。

OOB误差是**测试误差的无偏估计**，而且**不需要额外的验证集**——这是随机森林的一个巨大优势。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")
# OOB Score 与 Test Accuracy 通常非常接近
```

---

## 五、特征重要性计算

### 5.1 基于基尼减少量的重要性（Gini Importance）

这是随机森林**默认**的特征重要性计算方法。

**数学原理**：

在决策树的每个节点，分裂会带来**不纯度的减少**。对于分类树，不纯度用**基尼指数（Gini Index）** 衡量：

$$
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
$$

其中 $p_k$ 是节点 $D$ 中第 $k$ 类样本的比例。

对于一个节点 $D$ 按特征 $j$ 分裂为左子节点 $D_L$ 和右子节点 $D_R$，**基尼减少量**为：

$$
\Delta\text{Gini} = \text{Gini}(D) - \frac{|D_L|}{|D|}\text{Gini}(D_L) - \frac{|D_R|}{|D|}\text{Gini}(D_R)
$$

**特征 $j$ 的重要性**就是：**在所有树中，所有使用特征 $j$ 进行分裂的节点的基尼减少量之和**。

$$
\text{Importance}(j) = \sum_{t=1}^{T} \sum_{\text{node } s \text{ splits on } j} \Delta\text{Gini}(s)
$$

最后将所有特征的重要性归一化到 $[0, 1]$ 区间。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 生成数据：只有3个特征是有信息的
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=3,
    n_redundant=0, n_repeated=0, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 基尼重要性
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

# 可视化
plt.figure(figsize=(10, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], capsize=5)
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices])
plt.xlabel('特征')
plt.ylabel('基尼重要性')
plt.title('随机森林特征重要性（基于基尼减少量）')
plt.show()
```

**基尼重要性的局限性**：
- 对**高基数特征**（取值多的特征）有偏好
- 可能**高估相关特征**的重要性

### 5.2 基于排列的重要性（Permutation Importance）

为了克服基尼重要性的偏差，另一种方法是**排列重要性（Permutation Importance）** 。

**数学原理**：

1. 在训练好的模型上，计算原始数据集上的性能指标（如准确率或MSE）
2. 对于特征 $j$，**随机打乱**该特征在所有样本中的取值（破坏特征与标签的关联）
3. 重新计算打乱后的性能指标
4. **重要性 = 原始性能 - 打乱后的性能**

如果打乱某个特征后性能**显著下降**，说明该特征对模型很重要；如果性能**几乎不变**，说明该特征不重要。

排列重要性的优点：
- **模型无关**：适用于任何模型
- **无偏**：不依赖于特定的分裂准则
- **更可信**：直接反映特征对预测的实际贡献

```python
from sklearn.inspection import permutation_importance

# 计算排列重要性
result = permutation_importance(
    rf, X, y, 
    n_repeats=10,  # 每个特征打乱10次取平均
    random_state=42
)

perm_importances = result.importances_mean
perm_std = result.importances_std

# 对比两种重要性
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(X.shape[1]), importances)
plt.title('基尼重要性')

plt.subplot(1, 2, 2)
plt.bar(range(X.shape[1]), perm_importances)
plt.title('排列重要性')

plt.tight_layout()
plt.show()
```

### 5.3 两种方法的对比

| 对比维度 | 基尼重要性 | 排列重要性 |
|---------|-----------|-----------|
| **计算速度** | 快（训练时顺便计算） | 慢（需要额外计算） |
| **偏差** | 对高基数特征有偏好 | 无偏 |
| **模型依赖** | 仅适用于树模型 | 适用于任何模型 |
| **可解释性** | 间接（基于不纯度减少） | 直接（基于性能下降） |
| **scikit-learn实现** | `feature_importances_` | `permutation_importance` |

---

## 六、偏差-方差分解：理解随机森林为何有效

### 6.1 偏差-方差权衡回顾

在机器学习中，泛化误差可以分解为三个部分：

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

- **偏差（Bias）** ：模型预测的平均值与真实值之间的差异——衡量模型的**表达能力**
- **方差（Variance）** ：模型在不同训练集上的预测波动——衡量模型的**稳定性**
- **噪声（Noise）** ：数据本身的不可约误差

### 6.2 单棵决策树的问题

单棵**完全生长的决策树**：
- **低偏差**：能够完美拟合训练数据
- **高方差**：训练数据的微小变化会导致完全不同的树

这就是决策树容易**过拟合**的根本原因。

### 6.3 Bagging如何起作用

Bagging通过**平均**多棵树的预测来降低方差：

$$
\text{Var}(\text{Bagging}) = \rho\sigma^2 + \frac{1-\rho}{m}\sigma^2
$$

其中 $\rho$ 是树之间的相关性，$\sigma^2$ 是单棵树的方差。

- 当 $m \to \infty$ 时，方差趋近于 $\rho\sigma^2$
- **树之间的相关性越低（$\rho$ 越小），方差降低越多**

**Bagging不改变偏差**——如果单棵树有偏差，Bagging后的偏差基本不变。

### 6.4 随机森林：超越Bagging

随机森林通过**列采样**进一步降低树之间的相关性（$\rho \downarrow$），从而比Bagging获得更低的方差。

更令人惊讶的是，近年来的研究表明，随机森林在某些情况下还能**降低偏差**：

> 随机森林能够捕捉到Bagging集成无法捕捉的数据模式，在降低方差的同时也降低偏差。

特别是在**信噪比（SNR）较高**或**特征之间存在相关性**时，随机森林的这种优势更为明显。

### 6.5 直观理解

| 模型 | 偏差 | 方差 | 总体 |
|------|------|------|------|
| **单棵决策树** | 低 | **极高** | 过拟合 |
| **Bagging** | 不变（低） | **降低** | 改善 |
| **随机森林** | **可能更低** | **进一步降低** | 最优 |

**一句话总结**：随机森林 = Bagging（降低方差）+ 列采样（进一步降低相关性，可能同时降低偏差）。

---

## 七、完整示例：从数据到森林

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. 训练随机森林（调参示例）
rf = RandomForestClassifier(
    n_estimators=100,          # 树的数量
    max_features='sqrt',       # 列采样：sqrt(n_features)
    max_depth=10,              # 预剪枝：限制深度
    min_samples_split=10,      # 预剪枝：最小分裂样本数
    oob_score=True,            # 计算OOB误差
    random_state=42,
    n_jobs=-1                  # 并行计算
)
rf.fit(X_train, y_train)

# 3. 评估
print(f"训练集准确率: {rf.score(X_train, y_train):.4f}")
print(f"测试集准确率: {rf.score(X_test, y_test):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")

# 4. 特征重要性分析
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 重要特征:")
for i in range(10):
    print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 5. 交叉验证
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\n5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. 排列重要性（可选，计算较慢）
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
print("\n排列重要性 Top 5:")
top5_perm = np.argsort(result.importances_mean)[::-1][:5]
for i in top5_perm:
    print(f"  {feature_names[i]}: {result.importances_mean[i]:.4f} (+/- {result.importances_std[i]:.4f})")
```

---

## 八、总结

| 概念 | 核心内容 |
|------|---------|
| **Bootstrap** | 有放回抽样，每个样本被选中的概率 ≈ 63.2% |
| **Bagging** | Bootstrap + 聚合（投票/平均），**降低方差** |
| **随机森林** | Bagging + 列采样（特征子空间），**进一步降低相关性** |
| **OOB误差** | 利用未被选中的样本做验证，**免费的测试集** |
| **基尼重要性** | 累加所有树中特征分裂带来的基尼减少量，**快速但有偏** |
| **排列重要性** | 打乱特征后观察性能下降，**无偏但较慢** |
| **偏差-方差分解** | 随机森林降低方差，某些情况下也降低偏差 |

### 核心要点回顾

1. **Bootstrap自助采样**是Bagging的基石。每个Bootstrap样本包含约63.2%的独特样本，剩下的36.8%成为**袋外样本（OOB）** ，可用于免费验证。

2. **Bagging**通过对多个高方差模型（如决策树）的预测进行平均来**降低方差**，且不增加偏差。树之间的相关性越低，方差降低效果越好。

3. **随机森林** = Bagging + **列采样（特征子空间）** 。列采样进一步降低了树之间的相关性，使得方差降低效果更显著。在某些情况下，随机森林还能**同时降低偏差**。

4. **OOB误差**是随机森林的“免费午餐”——无需额外的验证集就能获得测试误差的无偏估计。

5. **特征重要性**有两种主流计算方法：
   - **基尼重要性**：累加特征在所有树中分裂带来的不纯度减少，计算快速但可能对高基数特征有偏好
   - **排列重要性**：打乱特征后观察性能下降，计算较慢但更可靠、模型无关

6. **偏差-方差分解**揭示了随机森林成功的根源：通过Bootstrap和列采样的双重随机性，随机森林在保持低偏差的同时大幅降低了方差。