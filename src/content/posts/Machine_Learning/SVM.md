---
title: Support Vector Machine (SVM) —— 支持向量机
published: 2025-07-06
description: 系统讲解支持向量机的完整数学推导与核心原理。涵盖函数间隔与几何间隔的定义及差异、硬间隔SVM的原始问题、拉格朗日对偶推导与KKT条件、软间隔SVM通过松弛变量与惩罚参数C处理线性不可分数据、Hinge Loss作为SVM损失函数的等价视角，以及SVM与逻辑回归在离群点鲁棒性上的本质差异。
cover: "/assets/images/posts/svm.png"
coverInContent: false
tags: [SVM, KKT条件, 核方法]
category: Machine_Learning
draft: false
---

# Support Vector Machine (SVM) —— 支持向量机

## 前言

在前两篇文章中，我们分别讨论了线性回归和逻辑回归。如果说线性回归是机器学习的“Hello World”，逻辑回归是二分类的“标配”，那么**支持向量机（Support Vector Machine, SVM）** 就是传统机器学习时代最具理论深度和美学魅力的算法之一。

俗话说，**SVM有三宝：间隔、对偶、核技巧**。这句话精准地概括了SVM最精髓的三个部分：
- **间隔**：SVM的核心思想是最大化分类超平面与最近样本点之间的距离
- **对偶**：通过拉格朗日对偶性将原始问题转化为更易求解的对偶问题
- **核技巧**：通过核函数将数据映射到高维空间，处理非线性分类问题

这篇文章，我们会从函数间隔与几何间隔的定义出发，一步步推导硬间隔SVM的原始问题、对偶问题及KKT条件，然后引入软间隔和松弛变量处理线性不可分数据，最后深入探讨SVM与逻辑回归在离群点鲁棒性上的本质差异。

---

## 一、函数间隔与几何间隔

### 1.1 超平面与分类决策

在二分类问题中，我们有一组训练样本 $\{(x_i, y_i)\}_{i=1}^{N}$，其中 $x_i \in \mathbb{R}^p$，$y_i \in \{+1, -1\}$。

我们希望找到一个**超平面** $w^T x + b = 0$ 将两类样本分开。对于任意样本点 $x_i$，分类决策为：

$$
\text{sign}(w^T x_i + b) =
\begin{cases}
+1, & w^T x_i + b > 0 \\
-1, & w^T x_i + b < 0
\end{cases}
$$

### 1.2 函数间隔（Functional Margin）

**函数间隔**的定义为：

$$ \hat{\gamma}_i = y_i (w^T x_i + b) $$

其含义是：
- 如果分类正确，$y_i(w^T x_i + b) > 0$，函数间隔为正
- $|w^T x_i + b|$ 越大，分类的**确信度**越高
- 但函数间隔有一个严重的问题：**如果同时将 $w$ 和 $b$ 放大 2 倍，超平面本身完全不变，但函数间隔也会放大 2 倍**

因此，函数间隔不适合直接作为优化目标。

### 1.3 几何间隔（Geometric Margin）

**几何间隔**定义为样本点到超平面的**真实距离**：

$$ \gamma_i = \frac{y_i(w^T x_i + b)}{\|w\|} $$

几何间隔就是**点到超平面的距离**。它的绝对值与点到直线的距离公式完全一致。

函数间隔与几何间隔的关系为：

$$ \gamma_i = \frac{\hat{\gamma}_i}{\|w\|} $$

**关键区别**：几何间隔对 $w$ 和 $b$ 的缩放是**不变的**——同时放大 $w$ 和 $b$，几何间隔保持不变。

### 1.4 为什么最大化几何间隔？

对于一个包含 $N$ 个点的数据集，分类的**确信度**与间隔大小正相关。我们希望找到的超平面，不仅要能正确分类，还要让所有样本点都**尽可能远离**超平面。这样，当新样本到来时，即使有轻微的扰动，也不容易被误分类。

因此，SVM 的目标是**最大化最小几何间隔**：

$$ \max_{w,b} \gamma, \quad \text{s.t.} \quad \frac{y_i(w^T x_i + b)}{\|w\|} \geq \gamma, \quad i = 1, 2, ..., N $$

其中 $\gamma = \min_i \gamma_i$，即所有样本点中**最小的**几何间隔。

---

## 二、硬间隔SVM：原始问题

### 2.1 从最大化间隔到最小化 $\|w\|$

令 $\hat{\gamma} = \gamma \|w\|$，上述优化目标可改写为：

$$ \max_{w,b} \frac{\hat{\gamma}}{\|w\|}, \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq \hat{\gamma}, \quad i = 1, 2, ..., N $$

由于我们可以通过成比例地缩放 $w$ 和 $b$ 来任意调节 $\hat{\gamma}$，不妨令 $\hat{\gamma} = 1$。于是优化问题变为：

$$ \max_{w,b} \frac{1}{\|w\|}, \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \quad i = 1, 2, ..., N $$

最大化 $\frac{1}{\|w\|}$ 等价于最小化 $\|w\|$，进一步等价于最小化 $\frac{1}{2}\|w\|^2$（平方和 1/2 是为了后续求导方便）。

于是得到**硬间隔SVM的原始问题（Primal Problem）** ：

$$ \boxed{\min_{w,b} \frac{1}{2}\|w\|^2} $$

$$ \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \quad i = 1, 2, ..., N $$

### 2.2 支持向量（Support Vectors）

使约束条件取等号的样本点，即 $y_i(w^T x_i + b) = 1$ 的点，被称为**支持向量**。

两个异类支持向量到超平面的距离之和为：

$$ \text{Margin} = \frac{2}{\|w\|} $$

最大化间隔 $\frac{2}{\|w\|}$ 等价于最小化 $\|w\|$，这与我们的优化目标一致。

---

## 三、拉格朗日对偶与KKT条件

### 3.1 为什么要转化为对偶问题？

直接求解原始问题虽然可行，但转化为对偶问题有三个重要优势：
1. **更容易求解**：对偶问题的约束更简单
2. **自然引入核函数**：对偶形式中只出现样本的内积 $x_i^T x_j$，可以用核函数替换
3. **揭示支持向量的本质**：只有支持向量对应的拉格朗日乘子非零

### 3.2 构造拉格朗日函数

对于原始问题，引入拉格朗日乘子 $\alpha_i \geq 0$（$i = 1, 2, ..., N$），构造拉格朗日函数：

$$ L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^{N} \alpha_i \left[ y_i(w^T x_i + b) - 1 \right] $$

展开为：

$$ L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^{N} \alpha_i y_i(w^T x_i + b) + \sum_{i=1}^{N} \alpha_i $$

### 3.3 对偶问题的推导

拉格朗日对偶问题为：

$$ \max_{\alpha} \min_{w,b} L(w, b, \alpha) $$

**第一步：对 $w$ 和 $b$ 求极小**

令 $L(w, b, \alpha)$ 对 $w$ 和 $b$ 的偏导为零：

$$ \frac{\partial L}{\partial w} = w - \sum_{i=1}^{N} \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^{N} \alpha_i y_i x_i $$

$$ \frac{\partial L}{\partial b} = -\sum_{i=1}^{N} \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^{N} \alpha_i y_i = 0 $$

将这两个结果代回拉格朗日函数：

$$ L(w, b, \alpha) = \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i^T x_j) $$

**第二步：对 $\alpha$ 求极大**

得到**对偶问题（Dual Problem）** ：

$$ \boxed{\max_{\alpha} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j x_i^T x_j} $$

$$ \text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad i = 1, 2, ..., N $$

### 3.4 KKT条件

对于SVM这个凸优化问题，**KKT条件是原始问题与对偶问题取得最优解的充分必要条件**。

KKT条件包含以下四个部分：

**（1）原始可行性（Primal Feasibility）** ：

$$ y_i(w^T x_i + b) - 1 \geq 0, \quad i = 1, 2, ..., N $$

**（2）对偶可行性（Dual Feasibility）** ：

$$ \alpha_i \geq 0, \quad i = 1, 2, ..., N $$

**（3）互补松弛性（Complementary Slackness）** ：

$$ \alpha_i \left[ y_i(w^T x_i + b) - 1 \right] = 0, \quad i = 1, 2, ..., N $$

这个条件极其重要！它意味着：
- 如果 $\alpha_i > 0$，则 $y_i(w^T x_i + b) = 1$ —— 该样本点是**支持向量**
- 如果 $y_i(w^T x_i + b) > 1$，则 $\alpha_i = 0$ —— 该样本点对模型**没有贡献**

**（4）梯度为零（Stationarity）** ：

$$ w = \sum_{i=1}^{N} \alpha_i y_i x_i, \quad \sum_{i=1}^{N} \alpha_i y_i = 0 $$

### 3.5 从对偶解回到原始解

求解对偶问题得到最优的 $\alpha_i^*$ 后，可以恢复原始参数：

$$ w^* = \sum_{i=1}^{N} \alpha_i^* y_i x_i $$

对于 $b^*$，选择任意一个支持向量（$\alpha_s > 0$），由互补松弛条件：

$$ y_s(w^{*T} x_s + b^*) = 1 $$

由于 $y_s^2 = 1$，两边同乘 $y_s$：

$$ b^* = y_s - w^{*T} x_s $$

在实际应用中，为了更鲁棒，通常取所有支持向量的平均值：

$$ b^* = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( y_s - \sum_{i \in \mathcal{S}} \alpha_i y_i x_i^T x_s \right) $$

其中 $\mathcal{S}$ 是所有支持向量的下标集合。

```python
import numpy as np
from cvxopt import matrix, solvers

class HardMarginSVM:
    """硬间隔SVM（使用CVXOPT求解对偶问题）"""
    
    def __init__(self):
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 构建对偶问题的二次规划形式
        # 最大化: sum(alpha_i) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * x_i^T x_j)
        # 约束: sum(alpha_i * y_i) = 0, alpha_i >= 0
        
        # P矩阵: P_ij = y_i * y_j * x_i^T x_j
        P = np.outer(y, y) * (X @ X.T)
        P = matrix(P.astype(np.float64))
        
        # q向量: q_i = -1 (因为cvxopt求解的是最小化)
        q = matrix(-np.ones(n_samples).astype(np.float64))
        
        # G矩阵和h向量: -alpha_i <= 0 => alpha_i >= 0
        G = matrix(-np.eye(n_samples).astype(np.float64))
        h = matrix(np.zeros(n_samples).astype(np.float64))
        
        # A矩阵和b向量: sum(alpha_i * y_i) = 0
        A = matrix(y.reshape(1, -1).astype(np.float64))
        b = matrix(np.zeros(1).astype(np.float64))
        
        # 求解QP
        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x']).flatten()
        
        # 计算w
        self.w = np.sum(self.alpha[:, None] * y[:, None] * X, axis=0)
        
        # 找出支持向量 (alpha > 1e-5)
        sv_idx = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[sv_idx]
        self.support_labels = y[sv_idx]
        
        # 计算b (取支持向量的平均值)
        self.b = np.mean([
            self.support_labels[i] - self.w @ self.support_vectors[i]
            for i in range(len(self.support_vectors))
        ])
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
```

---

## 四、软间隔SVM：松弛变量与Hinge Loss

### 4.1 为什么要引入软间隔？

硬间隔SVM要求所有样本都能被正确分类且满足 $y_i(w^T x_i + b) \geq 1$。这在现实数据中往往**过于苛刻**：
- 数据可能**线性不可分**（即使在高维空间）
- 可能存在**噪声或离群点**，强制完美分类会导致过拟合

### 4.2 松弛变量的引入

为了解决这个问题，软间隔SVM为每个样本引入一个**松弛变量（Slack Variable）** $\xi_i \geq 0$。

约束条件被放松为：

$$ y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 $$

松弛变量 $\xi_i$ 的几何含义是：样本点 $x_i$ 到边界 $w^T x + b = \pm 1$ 的距离。

- $\xi_i = 0$：样本点在正确一侧且满足间隔要求
- $0 < \xi_i < 1$：样本点在间隔内但在正确一侧
- $\xi_i = 1$：样本点在超平面上
- $\xi_i > 1$：样本点被**误分类**（位于超平面的错误一侧）

### 4.3 软间隔的原始问题

软间隔SVM的优化目标变为：

$$ \boxed{\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{N} \xi_i} $$

$$ \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i = 1, 2, ..., N $$

其中 $C > 0$ 是**惩罚参数**，控制着“最大化间隔”和“最小化分类错误”之间的权衡：
- $C \to \infty$：迫使所有样本满足约束，退化为硬间隔
- $C$ 较小时：允许更多样本违反约束，间隔更大，但分类错误可能更多

### 4.4 软间隔的对偶问题

构造拉格朗日函数（引入 $\alpha_i \geq 0$ 和 $\mu_i \geq 0$）：

$$ L(w, b, \xi, \alpha, \mu) = \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{N} \xi_i - \sum_{i=1}^{N} \alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] - \sum_{i=1}^{N} \mu_i \xi_i $$

对 $w$、$b$、$\xi_i$ 求偏导并令其为零：

$$ \frac{\partial L}{\partial w} = w - \sum_{i=1}^{N} \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^{N} \alpha_i y_i x_i $$

$$ \frac{\partial L}{\partial b} = -\sum_{i=1}^{N} \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^{N} \alpha_i y_i = 0 $$

$$ \frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \quad \Rightarrow \quad \alpha_i + \mu_i = C $$

由于 $\mu_i \geq 0$，可得 **$0 \leq \alpha_i \leq C$**。

代入后得到**软间隔SVM的对偶问题**：

$$ \boxed{\max_{\alpha} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j x_i^T x_j} $$

$$ \text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, ..., N $$

**与硬间隔的唯一区别**：$\alpha_i$ 的上限从 $+\infty$ 变成了 $C$。

### 4.5 Hinge Loss：SVM的另一种视角

软间隔SVM可以等价地写成一个**无约束优化问题**：

$$ \min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(w^T x_i + b)) $$

其中 $\max(0, 1 - y_i(w^T x_i + b))$ 就是著名的 **Hinge Loss（合页损失）** 。

Hinge Loss 的数学定义为：

$$ L_{\text{hinge}}(z) = \max(0, 1 - z), \quad z = y_i(w^T x_i + b) $$

**Hinge Loss 的特性**：
- 当 $y_i(w^T x_i + b) \geq 1$ 时，损失为 0（样本已经被正确分类且距离足够远）
- 当 $y_i(w^T x_i + b) < 1$ 时，损失线性增加
- 它是**分段线性的**，在 $z = 1$ 处不可导

松弛变量 $\xi_i$ 与 Hinge Loss 的关系为：

$$ \xi_i = \max(0, 1 - y_i(w^T x_i + b)) $$

因此，软间隔SVM的原始问题可以理解为：**L2正则化 + Hinge Loss**。

```python
def hinge_loss(y_true, y_pred):
    """Hinge Loss: max(0, 1 - y * y_pred)"""
    return np.maximum(0, 1 - y_true * y_pred)

# 示例
y_true = np.array([1, -1, 1, -1])
y_pred = np.array([0.8, -0.9, 1.5, 0.5])
losses = hinge_loss(y_true, y_pred)
print(f"Hinge Losses: {losses}")
# 输出: [0.2, 0.1, 0. , 1.5]
# 解释: 
# - 第1个样本: 1*0.8=0.8<1, 损失0.2
# - 第2个样本: (-1)*(-0.9)=0.9<1, 损失0.1
# - 第3个样本: 1*1.5=1.5>=1, 损失0
# - 第4个样本: (-1)*0.5=-0.5<1, 损失1.5
```

---

## 五、SVM与逻辑回归：离群点鲁棒性的本质差异

### 5.1 损失函数的差异

SVM和逻辑回归的核心差异在于**损失函数不同**：

| 模型 | 损失函数 | 数学形式 |
|------|---------|---------|
| **逻辑回归** | Log Loss（交叉熵） | $-\log(\sigma(y \cdot z))$ |
| **SVM** | Hinge Loss | $\max(0, 1 - y \cdot z)$ |

其中 $z = w^T x + b$，$\sigma$ 是 Sigmoid 函数。

### 5.2 为什么SVM对离群点更鲁棒？

**Log Loss（逻辑回归）的特性**：
- 即使样本被正确分类且距离超平面很远，Log Loss **仍然会持续减小**（但不会到零）
- 这意味着**每个样本都对模型有影响**，离群点会持续“拉扯”决策边界
- 逻辑回归对噪声和异常点比较敏感，容易过拟合

**Hinge Loss（SVM）的特性**：
- 一旦样本满足 $y_i(w^T x_i + b) \geq 1$，损失**直接降为 0**
- 这意味着**远离决策边界的样本对模型完全没有影响**
- 只有**支持向量**（位于间隔边界上或间隔内的样本）才决定决策边界
- SVM在决策边界附近的支持向量上更加鲁棒，对噪声的容忍性更好

### 5.3 几何直觉

可以这样理解两者的差异：

- **逻辑回归**像一个**完美主义者**：每个样本都在尽力“说服”决策边界向自己这边移动。即使一个点已经离边界很远，它仍然在施加影响。一个极端的离群点会把决策边界向自己这边“拉”很远。

- **SVM**像一个**实用主义者**：它只关心“临界地带”的样本。一旦一个样本离边界足够远（超过间隔），SVM就说“你安全了，我不再管你了”。因此，远离边界的离群点**根本不影响**SVM的决策边界。

### 5.4 惩罚参数C的作用

SVM的鲁棒性可以通过参数 $C$ 进一步调节：
- **较小的 $C$** ：允许更多样本违反间隔约束，对离群点更宽容，决策边界更平滑
- **较大的 $C$** ：强迫更多样本满足约束，对离群点更敏感，决策边界更曲折

```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 生成包含离群点的数据
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)
# 添加一个离群点
X_outlier = np.array([[3.5, -2.5]])
y_outlier = np.array([1])
X_aug = np.vstack([X, X_outlier])
y_aug = np.hstack([y, y_outlier])

# 训练逻辑回归和SVM
lr = LogisticRegression(C=1.0)
svm = SVC(kernel='linear', C=1.0)
lr.fit(X_aug, y_aug)
svm.fit(X_aug, y_aug)

# 打印系数差异
print("逻辑回归系数:", lr.coef_)
print("SVM系数:", svm.coef_)
# 通常SVM的决策边界受离群点影响更小
```

### 5.5 总结对比

| 特性 | 逻辑回归 | SVM |
|------|---------|-----|
| **损失函数** | Log Loss（交叉熵） | Hinge Loss |
| **所有样本都有影响** | ✅ 是 | ❌ 否（只有支持向量有影响） |
| **对离群点敏感度** | 高 | 低 |
| **鲁棒性** | 较弱 | 较强 |
| **输出概率** | ✅ 天然输出概率 | ❌ 需要额外校准（Platt scaling） |
| **适用场景** | 需要概率解释、数据较干净 | 数据有噪声、需要强泛化能力 |

---

## 六、总结

| 概念 | 数学形式 | 核心作用 |
|------|---------|---------|
| **函数间隔** | $\hat{\gamma}_i = y_i(w^T x_i + b)$ | 衡量分类确信度，但对缩放敏感 |
| **几何间隔** | $\gamma_i = \hat{\gamma}_i / \|w\|$ | 点到超平面的真实距离，缩放不变 |
| **硬间隔原始问题** | $\min \frac{1}{2}\|w\|^2$，s.t. $y_i(w^T x_i + b) \geq 1$ | 线性可分时的最大化间隔 |
| **硬间隔对偶问题** | $\max \sum \alpha_i - \frac{1}{2}\sum\sum \alpha_i\alpha_j y_i y_j x_i^T x_j$，s.t. $\sum \alpha_i y_i = 0, \alpha_i \geq 0$ | 更易求解，引入核函数 |
| **KKT条件** | 可行性 + 对偶可行性 + 互补松弛 + 梯度为零 | 原始与对偶最优解的充要条件 |
| **软间隔** | $\min \frac{1}{2}\|w\|^2 + C\sum \xi_i$，s.t. $y_i(w^T x_i + b) \geq 1 - \xi_i$ | 处理线性不可分数据 |
| **Hinge Loss** | $\max(0, 1 - y_i(w^T x_i + b))$ | SVM的损失函数，产生稀疏解 |
| **SVM vs LR** | Hinge Loss vs Log Loss | SVM只关注支持向量，对离群点更鲁棒 |

### 核心要点回顾

1. **函数间隔 vs 几何间隔**：函数间隔对 $w$ 和 $b$ 的缩放敏感，而几何间隔是点到超平面的真实距离，是SVM优化的真正目标。

2. **硬间隔SVM**：要求所有样本满足 $y_i(w^T x_i + b) \geq 1$，通过最大化间隔（等价于最小化 $\|w\|$）找到最优超平面。

3. **拉格朗日对偶**：将原始问题转化为对偶问题，对偶形式中只出现样本内积 $x_i^T x_j$，为核技巧铺平了道路。

4. **KKT条件**：互补松弛条件 $\alpha_i [y_i(w^T x_i + b) - 1] = 0$ 揭示了SVM的稀疏性——只有支持向量（$\alpha_i > 0$）影响模型。

5. **软间隔与Hinge Loss**：通过松弛变量 $\xi_i$ 允许部分样本违反间隔约束，等价于在L2正则化上使用Hinge Loss。参数 $C$ 控制着间隔宽度与分类误差的权衡。

6. **SVM vs 逻辑回归**：SVM使用Hinge Loss，一旦样本满足间隔要求损失即为0，因此只有支持向量决定决策边界，对离群点更加鲁棒。逻辑回归使用Log Loss，所有样本都持续影响模型，对离群点更敏感。