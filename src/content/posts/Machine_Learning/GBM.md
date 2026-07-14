---
title: Gradient Boosting Machine (GBM) —— 梯度提升机与加法模型
published: 2025-07-18
description: 系统推导梯度提升机（GBM）的数学原理，从加法模型与前向分步算法出发，深入剖析“拟合负梯度”在函数空间梯度下降中的核心地位，通过偏差-方差分解对比GBM与随机森林的本质差异，并分析GBM对异常值敏感的根本原因及缓解策略（鲁棒损失函数、学习率、子采样）。
cover: "/assets/images/posts/gbm.png"
coverInContent: false
tags: [梯度提升机, GBM, 加法模型, 机器学习]
category: Machine_Learning
draft: false
---

# Gradient Boosting Machine (GBM) —— 梯度提升机与加法模型

## 前言

在上一篇文章中，我们深入探讨了随机森林——这个通过 **“并行投票”** 降低方差的集成学习算法。随机森林的精髓在于：**让多棵树独立生长，然后让它们“民主投票”** 。每棵树都有“话语权”，但任何一棵树都不能主导最终决策。

**梯度提升机（Gradient Boosting Machine, GBM）** 则采用了完全不同的哲学。它由Jerome Friedman于1999年提出，其核心思想可以概括为 **“知错就改，循序渐进”** ——每一棵新树不是为了“发表独立意见”，而是专门用来**修正前面所有树的集体错误**。

如果说随机森林是 **“众人的智慧”** ，那么GBM就是 **“一个人的持续进步”** ——每一步都比上一步更好一点。

这篇文章，我们将从加法模型和前向分步算法出发，一步步推导GBM的完整数学框架，深入理解 **“拟合负梯度”** 为何是GBM的核心，通过偏差-方差分解对比GBM与随机森林的本质差异，最后分析GBM对异常值的敏感性及其背后的数学原因。

---

## 一、加法模型与前向分步算法

### 1.1 加法模型：Boosting家族的通用框架

Boosting类算法（包括AdaBoost和GBM）都可以统一描述为**加法模型（Additive Model）** ：

> **加法模型**：最终模型由多个基函数的**加权和**组成。

数学上，加法模型的形式为：

$$
f(x) = \sum_{m=1}^{M} \beta_m b(x; \gamma_m)
$$

其中：
- $b(x; \gamma_m)$ 是第 $m$ 个基函数（在GBM中就是决策树）
- $\gamma_m$ 是基函数的参数
- $\beta_m$ 是基函数的权重系数

在给定训练数据 $T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$ 和损失函数 $L(y, f(x))$ 的条件下，学习加法模型 $f(x)$ 就变成了一个经验风险极小化问题：

$$
\min_{\beta_m, \gamma_m} \sum_{i=1}^{N} L\left(y_i, \sum_{m=1}^{M} \beta_m b(x_i; \gamma_m)\right)
$$

这是一个**极其复杂的优化问题**——我们需要同时优化所有基函数的参数和权重。当 $M$ 较大时，直接求解几乎是不可能的。

### 1.2 前向分步算法：化整为零的求解策略

**前向分步算法（Forward Stagewise Algorithm）** 是解决上述问题的巧妙策略：

> **核心思想**：**从前向后，每一步只学习一个基函数及其系数**，逐步逼近目标函数。

具体来说，在第 $m$ 步，我们**固定**前 $m-1$ 个基函数不变，只优化第 $m$ 个：

$$
(\beta_m, \gamma_m) = \arg\min_{\beta, \gamma} \sum_{i=1}^{N} L\left(y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma)\right)
$$

其中 $f_{m-1}(x) = \sum_{j=1}^{m-1} \beta_j b(x; \gamma_j)$ 是前 $m-1$ 步已经得到的模型。

更新模型：

$$
f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)
$$

前向分步算法的完整流程如下：

```
输入：训练数据集 T，损失函数 L，基函数集合 {b(x; γ)}
输出：加法模型 f(x)

1. 初始化 f₀(x) = 0
2. 对 m = 1, 2, ..., M：
   (a) 极小化损失函数，求解：
       (β_m, γ_m) = argmin_{β,γ} Σᵢ L(yᵢ, f_{m-1}(xᵢ) + β·b(xᵢ; γ))
   (b) 更新：f_m(x) = f_{m-1}(x) + β_m·b(x; γ_m)
3. 得到加法模型：f(x) = f_M(x) = Σ_{m=1}^M β_m·b(x; γ_m)
```

**前向分步算法的优势**在于：它将一个复杂的全局优化问题，转化为 $M$ 个相对简单的**局部优化问题**，每一步只需要优化一个基函数。

---

## 二、从前向分步到梯度提升

### 2.1 前向分步的困境

前向分步算法虽然思路清晰，但每一步仍然需要求解一个优化问题：

$$
\min_{\beta, \gamma} \sum_{i=1}^{N} L(y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma))
$$

对于**任意**损失函数 $L$，这个优化问题可能仍然很困难。特别是当损失函数不是平方损失或指数损失时，没有简单的解析解。

**梯度提升（Gradient Boosting）** 的核心贡献就在于：**它用“拟合负梯度”代替了“直接优化损失函数”** ，从而将前向分步算法推广到了**任意可微损失函数**。

### 2.2 函数空间的梯度下降

要理解GBM为什么“拟合负梯度”，我们需要跳出**参数空间**，进入**函数空间**。

**回顾参数空间的梯度下降**：

在传统的参数优化中，我们最小化损失函数 $L(\theta)$，参数更新公式为：

$$
\theta = \theta - \alpha \cdot \frac{\partial L(\theta)}{\partial \theta}
$$

**函数空间的梯度下降**：

在GBM中，我们把**函数 $f(x)$ 本身当作优化变量**。在第 $m$ 步，前 $m-1$ 个基函数已经固定，即 $f_{m-1}(x)$ 已知。我们的目标是最小化：

$$
L(f) = \sum_{i=1}^{N} L(y_i, f_m(x_i))
$$

其中 $f_m(x) = f_{m-1}(x) + \beta_m h_m(x)$。

如果我们把 $f(x)$ 当作“参数”，用梯度下降的思路来更新：

$$
f_m(x) = f_{m-1}(x) - \rho_m \cdot \left. \frac{\partial L(y, f(x))}{\partial f(x)} \right|_{f=f_{m-1}}
$$

**关键洞察**：对比 $f_m(x) = f_{m-1}(x) + \beta_m h_m(x)$ 和上面的梯度下降更新式，可以发现：

$$
\boxed{h_m(x) \approx -\frac{\partial L(y, f_{m-1}(x))}{\partial f_{m-1}(x)}}
$$

也就是说，**第 $m$ 棵树的训练目标，就是去拟合前一轮模型损失函数的负梯度**。

这就是GBM中 **“拟合负梯度”** 的完整数学推导。

### 2.3 负梯度的直观理解

负梯度也被称为 **“伪残差（Pseudo Residual）”** 。

- 当损失函数是**平方损失** $L(y, f) = \frac{1}{2}(y - f)^2$ 时：

$$
-\frac{\partial L}{\partial f} = y - f = \text{残差}
$$

此时GBM退化为**拟合残差**——每一棵树学习的是“真实值与当前预测值的差”。

- 当损失函数是**绝对损失** $L(y, f) = |y - f|$ 时，负梯度是 **$\text{sign}(y - f)$** ——只告诉新树“方向”，不告诉“幅度”。

- 当损失函数是**对数损失**（分类任务）时，负梯度是**概率误差**。

**直观理解**：残差 $y - f(x)$ 越大，说明前一轮模型 $f(x)$ 与真实值 $y$ 相差越大。下一轮学习器通过拟合残差（或更一般地，负梯度），就能**重点纠正前一轮犯错较大的地方**。

---

## 三、GBM的完整算法流程

### 3.1 回归任务的GBM

**输入**：训练集 $\{(x_i, y_i)\}_{i=1}^{N}$，可微损失函数 $L(y, f(x))$，迭代次数 $M$

**输出**：最终模型 $f_M(x)$

**算法步骤**：

**步骤1：初始化**

$$
f_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma)
$$

即用一个常数 $\gamma$ 来初始化模型。对于平方损失，$\gamma$ 就是所有 $y_i$ 的均值。

**步骤2：对 $m = 1, 2, ..., M$ 迭代**

**(a) 计算负梯度（伪残差）** ：

$$
r_{im} = -\left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f=f_{m-1}}, \quad i = 1, 2, ..., N
$$

**(b) 用基学习器（通常是CART回归树）拟合负梯度**：

训练一棵回归树，以 $\{x_i\}$ 为输入，$\{r_{im}\}$ 为目标值。得到第 $m$ 棵树的叶节点区域 $\{R_{jm}\}_{j=1}^{J_m}$。

**(c) 计算每个叶节点的最优输出值**：

$$
\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)
$$

**(d) 更新模型**：

$$
f_m(x) = f_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} \cdot \mathbb{I}(x \in R_{jm})
$$

其中 $\nu \in (0, 1]$ 是**学习率（Learning Rate）** ，也叫**收缩系数（Shrinkage）** ，用于控制每棵树的贡献，防止过拟合。

**步骤3：输出最终模型** $f_M(x)$。

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    """从零实现梯度提升回归器"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None
    
    def fit(self, X, y):
        # 步骤1：初始化
        self.initial_pred = np.mean(y)
        f = np.full(len(y), self.initial_pred)
        
        # 步骤2：迭代M轮
        for _ in range(self.n_estimators):
            # (a) 计算负梯度（对于平方损失，负梯度 = y - f）
            residuals = y - f
            
            # (b) 用决策树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # (c) 更新预测（学习率控制步长）
            f += self.learning_rate * tree.predict(X)
        
        return self
    
    def predict(self, X):
        f = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        return f

# 示例：拟合正弦函数
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
gbm.fit(X, y)
y_pred = gbm.predict(X)

print(f"训练集MSE: {np.mean((y - y_pred) ** 2):.4f}")
```

### 3.2 分类任务的GBM

对于二分类问题，GBM使用**对数损失（Log Loss）** ：

$$
L(y, f(x)) = \log(1 + e^{-y f(x)}), \quad y \in \{-1, +1\}
$$

负梯度为：

$$
r_{im} = \frac{y_i}{1 + e^{y_i f_{m-1}(x_i)}}
$$

其余步骤与回归任务相同。最终通过 **$\text{sign}(f_M(x))$** 做出分类决策。

---

## 四、GBM与随机森林：偏差-方差视角

### 4.1 偏差-方差分解回顾

在机器学习中，泛化误差可以分解为三个部分：

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

- **偏差（Bias）** ：模型预测的平均值与真实值之间的差异——反映模型的**表达能力**
- **方差（Variance）** ：模型在不同训练集上的预测波动——反映模型的**稳定性**
- **噪声（Noise）** ：数据本身的不可约误差

### 4.2 随机森林：降低方差

随机森林通过**并行构建多棵独立树**并**平均它们的预测**来降低方差。

在上一篇文章中我们推导过：如果 $m$ 棵树之间的相关性为 $\rho$，单棵树的方差为 $\sigma^2$，则平均预测的方差为：

$$
\text{Var}(\text{RF}) = \rho\sigma^2 + \frac{1-\rho}{m}\sigma^2
$$

当 $m \to \infty$ 时，方差趋近于 $\rho\sigma^2$。随机森林通过**行采样（Bootstrap）** 和**列采样（特征子空间）** 来降低 $\rho$，从而进一步降低方差。

**随机森林的本质**：**通过“平均”来平滑掉单棵树的噪声，降低方差**。

### 4.3 GBM：降低偏差

GBM则走了完全不同的路径。它通过**串行地拟合残差（负梯度）** ，每一轮都在**修正前一轮的错误**。

从偏差-方差的角度来看：

> **随机森林降低方差，GBM降低偏差**。

**GBM的本质**：**通过“持续纠错”来逼近真实函数，降低偏差**。

这解释了为什么：
- **随机森林**在噪声数据上更稳健，不容易过拟合
- **GBM**在干净数据上通常能达到更高的精度，但需要精细调参

### 4.4 核心对比总结

| 对比维度 | **随机森林** | **GBM** |
|---------|------------|---------|
| **构建方式** | 并行构建独立树 | 串行构建依赖树 |
| **核心目标** | **降低方差** | **降低偏差** |
| **树的特点** | 完全生长的深树 | 浅层弱学习器 |
| **过拟合风险** | 较低（内置正则化） | 较高（需精细调参） |
| **训练速度** | 快（可并行） | 慢（需串行） |
| **对噪声敏感度** | 低 | 高 |
| **适用场景** | 快速原型、噪声数据 | 追求极致精度 |

**一句话总结**：

> **随机森林是“安全的”，GBM是“锋利的”** 。

---

## 五、GBM对异常值的敏感性

### 5.1 为什么GBM对异常值敏感？

GBM对异常值的敏感性源于其**“串行纠错”** 的机制。

**原因一：平方损失放大了异常值的影响**

当使用平方损失时，负梯度就是残差 $y - f(x)$。如果一个样本是异常值（即 $y$ 与正常值相差很大），它的**残差会非常大**。由于平方损失对**大误差的惩罚是二次的**，这个异常值会在梯度中产生**不成比例的巨大影响**。

在下一轮迭代中，新树会**重点拟合这个巨大的残差**，从而导致模型被异常值“带偏”。

**原因二：串行传播放大了异常值的影响**

在随机森林中，一棵树受异常值影响，其他树可能不受影响，最终投票可以“稀释”掉这个影响。但在GBM中，**每一棵树都建立在前面所有树的基础上**——如果早期的一棵树被异常值带偏，这个偏差会**在后续的迭代中被不断放大**。

### 5.2 如何缓解GBM对异常值的敏感性？

**方法一：使用鲁棒的损失函数**

GBM的一大优势是**可以使用任意可微的损失函数**。如果我们担心异常值的影响，可以用**Huber损失**或**绝对损失**代替平方损失。

- **Huber损失**：在误差较小时使用平方损失（光滑可微），在误差较大时使用绝对损失（对异常值不敏感）

$$
L_{\delta}(y, f) = 
\begin{cases}
\frac{1}{2}(y - f)^2, & |y - f| \leq \delta \\
\delta |y - f| - \frac{1}{2}\delta^2, & |y - f| > \delta
\end{cases}
$$

**方法二：降低学习率**

较小的学习率意味着每棵树的贡献更小，模型更新更“谨慎”，从而降低了异常值的影响。

**方法三：子采样（Subsampling）**

在每次迭代中，只使用部分样本（如80%）来训练树。这可以降低异常值被选中的概率，类似于随机森林的行采样思想。

**方法四：使用XGBoost/LightGBM等现代实现**

这些实现内置了**L1/L2正则化**，可以有效控制模型的复杂度，减轻异常值的影响。

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import numpy as np

# 生成数据并加入异常值
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
y[0] = y[0] * 10  # 制造一个异常值

# 使用平方损失（默认）
gbm_mse = GradientBoostingRegressor(loss='squared_error', n_estimators=100, 
                                      learning_rate=0.1, random_state=42)
gbm_mse.fit(X, y)

# 使用Huber损失（对异常值更鲁棒）
gbm_huber = GradientBoostingRegressor(loss='huber', n_estimators=100,
                                       learning_rate=0.1, random_state=42)
gbm_huber.fit(X, y)

print(f"MSE损失模型在异常值样本上的预测: {gbm_mse.predict([X[0]])[0]:.2f}")
print(f"Huber损失模型在异常值样本上的预测: {gbm_huber.predict([X[0]])[0]:.2f}")
print(f"真实值: {y[0]:.2f}")
# Huber损失通常能更好地抵抗异常值的影响
```

---

## 六、完整代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 生成非线性数据
np.random.seed(42)
X = np.linspace(0, 10, 300).reshape(-1, 1)
y = np.sin(X).ravel() + 0.3 * np.random.randn(300)
# 加入异常值
y[0] = 5.0
y[10] = -4.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. 训练不同参数的GBM
results = {}
for lr in [0.01, 0.1, 0.5]:
    for depth in [2, 4, 6]:
        gbm = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=lr,
            max_depth=depth,
            random_state=42
        )
        gbm.fit(X_train, y_train)
        y_pred = gbm.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[(lr, depth)] = mse
        print(f"lr={lr:.2f}, depth={depth}: MSE={mse:.4f}")

# 3. 可视化最佳模型
best_params = min(results, key=results.get)
best_gbm = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=best_params[0],
    max_depth=best_params[1],
    random_state=42
)
best_gbm.fit(X_train, y_train)

X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
y_plot = best_gbm.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='训练集')
plt.scatter(X_test, y_test, alpha=0.5, label='测试集')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='GBM预测')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title(f'GBM: lr={best_params[0]}, depth={best_params[1]}')
plt.show()
```

---

## 七、总结

| 概念 | 核心内容 |
|------|---------|
| **加法模型** | $f(x) = \sum \beta_m b(x; \gamma_m)$，Boosting家族的通用框架 |
| **前向分步算法** | 从前向后，每一步只优化一个基函数 |
| **GBM的核心思想** | 用基学习器拟合前一轮损失函数的**负梯度** |
| **函数空间梯度下降** | 将 $f(x)$ 本身当作优化变量，在函数空间中做梯度下降 |
| **伪残差** | 负梯度的别名，是“残差”在任意损失函数下的推广 |
| **GBM vs 随机森林** | GBM降低**偏差**，随机森林降低**方差** |
| **异常值敏感性** | 平方损失放大异常值影响，串行机制放大偏差 |
| **缓解方法** | 鲁棒损失函数、降低学习率、子采样、正则化 |

### 核心要点回顾

1. **加法模型 + 前向分步算法**是Boosting家族的通用框架。前向分步将复杂的全局优化转化为 $M$ 个简单的局部优化。

2. **GBM的核心创新**在于：用 **“拟合负梯度”** 代替了 **“直接优化损失函数”** ，从而将Boosting推广到了**任意可微损失函数**。负梯度是“残差”在任意损失函数下的推广。

3. **函数空间的梯度下降**是理解GBM的关键视角——我们把 $f(x)$ 本身当作优化变量，每一轮都在函数空间中沿着负梯度方向走一小步。

4. **GBM vs 随机森林**：随机森林通过**并行平均**降低**方差**，GBM通过**串行纠错**降低**偏差**。随机森林 **“安全”** ，GBM **“锋利”** 。

5. **GBM对异常值敏感**的根本原因在于：平方损失放大异常值的影响，而串行机制会把这个影响在后续迭代中不断放大。可以通过**Huber损失**、**降低学习率**、**子采样**和**正则化**来缓解。