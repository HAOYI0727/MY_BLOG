---
title: Linear Regression —— 线性回归
published: 2025-07-02
description: 系统讲解线性回归模型的核心原理与演进路径，涵盖最小二乘法（OLS）的矩阵推导与正规方程、Ridge（L2）正则化的闭式解与收缩特性、Lasso（L1）正则化的稀疏性来源、ElasticNet 的组合优势，以及 MSE/RMSE/MAE 评价指标的适用场景与异常值敏感性对比。
cover: "/assets/images/posts/linear_regression.png"
coverInContent: false
tags: [线性回归, 最小二乘法, 正则化, 机器学习]
category: Machine_Learning
draft: false
---

# Linear Regression —— 线性回归

## 前言

线性回归是机器学习中最古老、最基础，也最重要的模型之一。它简单却蕴含着机器学习的核心思想：**从数据中学习一条“最佳”的线，用来做预测**。

很多教程会直接告诉你：“调用 scikit-learn 的 `LinearRegression` 类，一行代码就能搞定！”这当然没错，但如果止步于此，算法对我们来说就永远是一个“黑箱”。我们不知道它为何有效，更不知道它在什么情况下会失效。

这篇博客的目标，就是打破这个黑箱。我们会从最小二乘法的数学推导开始，逐步深入到 Ridge、Lasso 和 ElasticNet 正则化，最后讨论模型评价指标

---

## 一、线性回归与最小二乘法

### 1.1 什么是线性回归？

线性回归试图找到一个线性方程，来描述自变量 $X$ 与因变量 $y$ 之间的关系。对于单变量情况，就是一条直线：

$$ y = w_1 x + w_0 $$

其中 $w_1$ 是斜率，$w_0$ 是截距。在机器学习中，我们统称为模型的**权重（weights）** 。

对于多元情况，可以写成矩阵形式：

$$ y = Xw $$

其中 $X$ 是 $n \times p$ 的设计矩阵（每行一个样本，每列一个特征），$w$ 是 $p \times 1$ 的权重向量。

### 1.2 如何定义“最佳”？——最小二乘法

给定一条直线，每个数据点都有一个**残差（residual）**：真实值与预测值之差。我们需要一个指标来衡量“总体残差”的大小。

最经典的做法是：**最小化残差平方和**。这就是**最小二乘法（Ordinary Least Squares, OLS）** 。

损失函数为：

$$ L(w) = \sum_{i=1}^{n} (y_i - X_i w)^2 = \|y - Xw\|_2^2 $$

### 1.3 矩阵求导推导闭式解

现在我们来手推这个最小化问题的解析解。

首先展开损失函数：

$$ L(w) = (y - Xw)^T(y - Xw) = y^T y - y^T X w - w^T X^T y + w^T X^T X w $$

由于 $y^T X w$ 和 $w^T X^T y$ 都是标量且互为转置（相等），所以：

$$ L(w) = y^T y - 2w^T X^T y + w^T X^T X w $$

对 $w$ 求梯度：

$$ \nabla_w L(w) = -2X^T y + 2X^T X w $$

令梯度为零：

$$ -2X^T y + 2X^T X w = 0 $$

$$ X^T X w = X^T y $$

这就是著名的**正规方程（Normal Equation）** 。当 $X^T X$ 可逆时：

$$ \boxed{w = (X^T X)^{-1} X^T y} $$

**注意**：这个闭式解要求 $X^T X$ 是满秩的（即特征之间不存在完全共线性）。如果特征数量大于样本数量（$p > n$），或者存在多重共线性，$X^T X$ 就是奇异矩阵，无法求逆。

### 1.4 Python 手动实现

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """手动实现线性回归（最小二乘法）"""
    
    def __init__(self):
        self.w = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        """
        # 添加一列全1，用于处理截距项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # (n, p+1)
        
        # 正规方程求解
        # w = (X^T X)^(-1) X^T y
        self.w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.w

# 生成示例数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 训练模型
model = LinearRegression()
model.fit(X, y.flatten())
print(f"截距: {model.intercept_:.4f}")
print(f"系数: {model.coef_[0]:.4f}")
# 输出: 截距: 4.2221, 系数: 2.7709

# 可视化
X_test = np.array([[0], [2]])
y_pred = model.predict(X_test)
plt.scatter(X, y, alpha=0.6)
plt.plot(X_test, y_pred, 'r-', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

---

## 二、Ridge回归（L2正则化）

### 2.1 为什么要正则化？

当 $X^T X$ 不可逆时（特征数 > 样本数，或存在多重共线性），OLS 无法求解。即使可逆，模型也可能**过拟合**——在训练集上表现很好，但在新数据上很差。

**正则化（Regularization）** 的核心思想是：在损失函数中加入一个惩罚项，限制模型参数的大小，从而防止过拟合。

### 2.2 Ridge 的数学形式

Ridge 回归（又称 L2 正则化）在 OLS 损失函数上增加了 **权重平方和** 作为惩罚项：

$$ L_{\text{ridge}}(w) = \|y - Xw\|_2^2 + \lambda \|w\|_2^2 $$

其中 $\lambda \geq 0$ 是正则化强度超参数。

写成展开形式：

$$ L_{\text{ridge}}(w) = y^T y - 2y^T X w + w^T X^T X w + \lambda w^T w $$

对 $w$ 求梯度：

$$ \nabla_w L_{\text{ridge}}(w) = -2X^T y + 2X^T X w + 2\lambda w $$

令梯度为零：

$$ (X^T X + \lambda I) w = X^T y $$

$$ \boxed{w = (X^T X + \lambda I)^{-1} X^T y} $$

**关键洞察**：即使 $X^T X$ 不可逆，加上 $\lambda I$ 后矩阵一定可逆（因为 $\lambda > 0$ 时，$X^T X + \lambda I$ 是正定矩阵）。这就是 Ridge 能处理病态矩阵的根本原因。

### 2.3 Ridge 的几何解释

Ridge 等价于在以下约束下最小化残差平方和：

$$ \sum_{j=1}^{p} w_j^2 \leq t $$

在二维空间中，L2 约束区域是一个**圆形**。圆形的边界是光滑的，最优解可能落在圆周的任何位置，但很难正好落在坐标轴上，因此 Ridge 通常不会将系数压缩到 0——它做的是**收缩（shrinkage）**，而不是**选择（selection）**。

### 2.4 Python 实现

```python
class RidgeRegression:
    """手动实现Ridge回归（L2正则化）"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 即 lambda
        self.w = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]
        
        # Ridge闭式解: (X^T X + alpha * I)^(-1) X^T y
        # 注意：截距项通常不参与正则化，这里为了简便对所有参数都加了惩罚
        I = np.eye(n_features)
        self.w = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.w

# 对比不同alpha的效果
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=50, n_features=10, noise=0.1, random_state=42)

for alpha in [0, 0.1, 1, 10]:
    model = RidgeRegression(alpha=alpha)
    model.fit(X, y)
    coef_norm = np.linalg.norm(model.w[1:])  # 忽略截距
    print(f"alpha={alpha:4.1f}, 系数L2范数={coef_norm:.4f}")
# 输出: alpha越大，系数范数越小（收缩越强）
```

---

## 三、Lasso回归（L1正则化）

### 3.1 Lasso 的数学形式

Lasso（Least Absolute Shrinkage and Selection Operator）使用 **L1 正则化**，惩罚项是权重的绝对值之和：

$$ L_{\text{lasso}}(w) = \|y - Xw\|_2^2 + \lambda \|w\|_1 = \|y - Xw\|_2^2 + \lambda \sum_{j=1}^{p} |w_j| $$

与 Ridge 不同，Lasso 的损失函数在 $w_j=0$ 处不可微，因此**没有简单的闭式解**。

### 3.2 Lasso 为什么能产生稀疏解？

#### 几何解释

Lasso 等价于在约束 $\sum_{j=1}^{p} |w_j| \leq t$ 下最小化残差平方和。

在二维空间中，L1 约束区域是一个**菱形**（diamond）。菱形的顶点在坐标轴上。当目标函数的等高线（椭圆）与约束区域相切时，**切点很可能落在菱形的顶点上**，此时对应的系数就被压缩为 0。

随着维度增加，多维菱形的“角”越来越多，系数被置零的概率就越大。这就是 Lasso 能够**自动进行特征选择**的原因——它会产生稀疏解。

#### 为什么 L1 产生稀疏而 L2 不产生？

- **L2（圆形）**：边界光滑，切点落在坐标轴上的概率为 0
- **L1（菱形）**：边界有尖角，切点很容易落在尖角上（即坐标轴）

**一句话总结：L1 的几何形状有“角”，所以容易产生零系数**。

### 3.3 软阈值算子（Soft-Thresholding Operator）

虽然 Lasso 没有闭式解，但在**单变量**情况下，我们可以推导出解析解。

考虑单变量 Lasso 问题（没有截距）：

$$ \min_{w} \frac{1}{2}(y - Xw)^2 + \lambda |w| $$

其中 $X$ 和 $y$ 都是标量（为简单起见，假设 $X^T X = 1$，即特征已标准化）。

目标是：

$$ \min_{w} \frac{1}{2}(y - Xw)^2 + \lambda |w| $$

展开并求**次梯度（subgradient）**：

- 当 $w > 0$ 时：$-X(y - Xw) + \lambda = 0$，得 $w = (Xy - \lambda)/X^2$
- 当 $w < 0$ 时：$-X(y - Xw) - \lambda = 0$，得 $w = (Xy + \lambda)/X^2$
- 当 $w = 0$ 时：次梯度包含 $[-Xy - \lambda, -Xy + \lambda]$，当 $|Xy| \leq \lambda$ 时成立

综合三种情况，得到**软阈值算子（Soft-Thresholding Operator）** ：

$$ \boxed{S(z, \lambda) = \text{sign}(z) \cdot \max(|z| - \lambda, 0)} $$

其中 $z = X^T y$（在标准化特征下）。

**软阈值的含义**：
- 如果 $|z| \leq \lambda$，输出为 0（**稀疏性来源**）
- 如果 $|z| > \lambda$，将 $z$ 向 0 收缩 $\lambda$ 的量

```python
def soft_threshold(z, lam):
    """软阈值算子"""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

# 演示
z_values = np.linspace(-3, 3, 100)
for lam in [0, 0.5, 1, 2]:
    s_values = [soft_threshold(z, lam) for z in z_values]
    plt.plot(z_values, s_values, label=f'λ={lam}')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('z')
plt.ylabel('S(z, λ)')
plt.legend()
plt.title('软阈值算子')
plt.show()
```

### 3.4 坐标下降法求解 Lasso

对于多维 Lasso，通常使用**坐标下降法（Coordinate Descent）** 求解。

核心思想：**每次只优化一个参数，固定其他所有参数**。

对于 Lasso 的第 $j$ 个坐标更新：

$$ w_j^{(new)} = S\left(\frac{1}{n} \sum_{i=1}^{n} X_{ij} (y_i - \sum_{k \neq j} X_{ik} w_k), \lambda\right) $$

即：计算当前残差与第 $j$ 个特征的相关性，然后应用软阈值。

scikit-learn 的 Lasso 实现默认就使用坐标下降法。

```python
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=20, n_informative=5, noise=0.1, random_state=42)

# 注意：alpha越大，稀疏性越强
for alpha in [0.01, 0.1, 1]:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-6)
    print(f"alpha={alpha:.2f}, 非零系数个数={n_nonzero}")
# alpha越大，非零系数越少（越稀疏）
```

---

## 四、ElasticNet（L1 + L2 组合）

### 4.1 为什么要结合 L1 和 L2？

- **Lasso（L1）**：能做特征选择，产生稀疏解。但在特征高度相关时，Lasso 会随机选择其中一个，导致**不稳定**。
- **Ridge（L2）**：系数收缩但不置零，在相关特征上表现稳定。

**ElasticNet** 结合了两者的优点。

### 4.2 ElasticNet 的数学形式

ElasticNet 的损失函数为：

$$ L_{\text{elastic}}(w) = \frac{1}{2n}\|y - Xw\|_2^2 + \alpha \cdot \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2}\|w\|_2^2 $$

其中：
- $\alpha$：整体正则化强度
- $\rho$（即 `l1_ratio`）：L1 和 L2 的混合比例

特殊情形：
- $\rho = 0$：纯 Ridge（L2）
- $\rho = 1$：纯 Lasso（L1）
- $0 < \rho < 1$：两者的组合

ElasticNet 也可以用坐标下降法求解。scikit-learn 的 `ElasticNet` 默认使用坐标下降。

```python
from sklearn.linear_model import ElasticNet

X, y = make_regression(n_samples=100, n_features=20, n_informative=5, noise=0.1, random_state=42)

for l1_ratio in [0, 0.5, 1]:
    enet = ElasticNet(alpha=0.1, l1_ratio=l1_ratio)
    enet.fit(X, y)
    n_nonzero = np.sum(np.abs(enet.coef_) > 1e-6)
    print(f"l1_ratio={l1_ratio:.1f}, 非零系数个数={n_nonzero}")
# l1_ratio=0 (Ridge): 全部非零
# l1_ratio=0.5: 部分稀疏
# l1_ratio=1 (Lasso): 最稀疏
```

---

## 五、评价指标：MSE、RMSE 与 MAE

### 5.1 MSE（均方误差）

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

MSE 是线性回归最常用的损失函数。它对**大误差给予更大的惩罚**（平方放大）。

### 5.2 RMSE（均方根误差）

$$ \text{RMSE} = \sqrt{\text{MSE}} $$

RMSE 只是 MSE 开方，**量纲与原始数据一致**，更易解释。比如房价预测中，MSE 的单位是“万元的平方”，而 RMSE 的单位直接是“万元”。

### 5.3 MAE（平均绝对误差）

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

MAE 是**线性**的，所有误差被同等对待。

### 5.4 异常值敏感性对比

这是三者最核心的区别：

| 指标 | 对异常值的敏感性 | 原因 |
|------|-----------------|------|
| **MSE/RMSE** | **非常敏感** | 平方运算放大了大误差的影响 |
| **MAE** | **相对鲁棒** | 绝对值不放大误差 |

**直观解释**：
- 最小化 MSE 等价于预测**均值**——均值对异常值敏感
- 最小化 MAE 等价于预测**中位数**——中位数对异常值鲁棒

**实际建议**：
- 如果异常值代表**重要的业务场景**（如欺诈检测），用 MSE/RMSE
- 如果异常值只是**噪声/脏数据**，用 MAE
- 如果想兼顾两者，可以用 **Huber 损失**（在误差小时用平方，误差大时用绝对值）

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 模拟正常数据和异常值
y_true = np.array([1, 2, 3, 4, 5])
y_pred_normal = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
y_pred_outlier = np.array([1.1, 1.9, 3.2, 3.8, 50])  # 最后一个点是异常预测

print("=== 正常情况 ===")
print(f"MSE: {mean_squared_error(y_true, y_pred_normal):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_normal)):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred_normal):.4f}")

print("\n=== 存在异常预测 ===")
print(f"MSE: {mean_squared_error(y_true, y_pred_outlier):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_outlier)):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred_outlier):.4f}")
# 可以看到：MSE/RMSE被异常值大幅拉高，而MAE变化相对温和
```

---

## 六、总结

| 方法 | 正则项 | 闭式解 | 稀疏性 | 适用场景 |
|------|--------|--------|--------|----------|
| **OLS** | 无 | ✅ $ (X^TX)^{-1}X^Ty $ | ❌ | 特征数 < 样本数，无共线性 |
| **Ridge** | L2: $\lambda\|w\|_2^2$ | ✅ $ (X^TX + \lambda I)^{-1}X^Ty $ | ❌ | 多重共线性，特征相关 |
| **Lasso** | L1: $\lambda\|w\|_1$ | ❌（坐标下降） | ✅ | 特征选择，高维数据 |
| **ElasticNet** | L1 + L2 | ❌（坐标下降） | ✅（可控） | 相关特征分组，兼顾选择与稳定 |

### 核心要点回顾

1. **最小二乘法**的闭式解 $w = (X^TX)^{-1}X^Ty$ 是线性回归的基石，但要求 $X^TX$ 可逆。

2. **Ridge（L2）** 通过在 $X^TX$ 对角线加 $\lambda$ 保证矩阵可逆，系数收缩但不归零。

3. **Lasso（L1）** 的稀疏性来自两个层面：
   - **几何**：L1 约束是菱形，更容易在坐标轴上相切
   - **算法**：软阈值算子 $S(z, \lambda) = \text{sign}(z)\max(|z|-\lambda, 0)$ 会将小系数直接置零

4. **ElasticNet** 结合 L1 和 L2，在特征选择的同时保持分组稳定性。

5. **MSE/RMSE 对异常值敏感**（平方放大误差），**MAE 相对鲁棒**（绝对值不放大）。