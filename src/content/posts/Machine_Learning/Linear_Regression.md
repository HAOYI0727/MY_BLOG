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

## 引言

**线性回归**是机器学习系列的第一个模型，也是整个机器学习大厦的"第一块基石"。它看起来简单——不过是一条"**最佳拟合直线**"——但其背后蕴含的思想却贯穿了后续所有模型：**定义损失函数、优化目标、处理过拟合**，这三步曲将在**逻辑回归、SVM、神经网络乃至梯度提升树**中反复重现。

本文的核心脉络遵循"**从纯净到修正**"的演进逻辑。我们首先从**最小二乘法（OLS）**出发，推导其**矩阵形式的闭式解**，并明确揭示其致命缺陷——当特征存在多重共线性或特征数超过样本数时，$X^TX$ 不可逆，模型崩塌。紧接着，我们引入**Ridge回归（L2正则化）**，通过在 $X^TX$ 对角线加上 $\lambda I$，不仅使矩阵永远**可逆**，还实现了**系数收缩**；随后是**Lasso回归（L1正则化）**，其**几何特性和软阈值算子**使其能够将无关特征系数精确置零，实现**特征选择**；最后，**ElasticNet** 融合二者优势，在相关特征分组场景中表现稳健。

在模型评价环节，我们对比了 **MSE/RMSE** 与 **MAE 对异常值的敏感性差异**——这不仅是评价指标的选择问题，其背后"**平方 vs 绝对值**"的数学本质，将直接决定模型在具体业务场景中的行为偏好。

> [!note]
> 
> 读完本文，你将彻底理解"**线性模型家族**"的完整谱系。
> 
> 而本文铺垫的"**损失函数设计 + 正则化修正**"这一分析框架，将是我们后续解读**逻辑回归和 SVM** 等模型时反复使用的核心方法论。

---

## 一、线性回归与最小二乘法

### 1.1 线性回归的含义

线性回归试图找到一个线性方程，来描述**自变量 $X$ 与因变量 $y$** 之间的关系。对于**单变量**情况，就是一条**直线**：

$$ y = w_1 x + w_0 $$

其中 $w_1$ 是斜率，$w_0$ 是截距。在机器学习中，我们统称为模型的**权重（weights）** 。

对于**多元**情况，可以写成**矩阵**形式：

$$ y = Xw $$

其中 $X$ 是 $n \times p$ 的**设计矩阵**（**每行一个样本，每列一个特征**），$w$ 是 $p \times 1$ 的**权重向量**。

### 1.2 最小二乘法

给定一条直线，每个数据点都有一个**残差（residual）：真实值与预测值之差**。我们需要一个指标来衡量“总体残差”的大小。

最经典的做法是：**最小化残差平方和**。这就是**最小二乘法（Ordinary Least Squares, OLS）**。

**损失函数**为：

$$ L(w) = \sum_{i=1}^{n} (y_i - X_i w)^2 = \|y - Xw\|_2^2 $$

### 1.3 矩阵求导推导闭式解

现在我们来手推这个最小化问题的解析解。

首先展开**损失函数**：

$$ L(w) = (y - Xw)^T(y - Xw) = y^T y - y^T X w - w^T X^T y + w^T X^T X w $$

由于 $y^T X w$ 和 $w^T X^T y$ 都是**标量且互为转置（相等）**，所以：

$$ L(w) = y^T y - 2w^T X^T y + w^T X^T X w $$

对 $w$ 求**梯度**：

$$ \nabla_w L(w) = -2X^T y + 2X^T X w $$

> [!tip] 关于矩阵的梯度（求导）：
> 
> 对于整理后的损失函数 $ L(w) = y^T y - 2w^T X^T y + w^T X^T X w $
> 
> - **第1项** $y^T y$：与 $w$ 无关，梯度为 **0**。
> - **第2项** $-2w^T X^T y$：这可以看成 $-2 \cdot w^T a$ 的形式（其中 $a = X^T y$）。使用公式 $\nabla_w (w^T a) = a$，所以这一项梯度为 $-2a = -2X^T y$。
> - **第3项** $w^T X^T X w$：这是典型的**二次型** $w^T A w$，其中 $A = X^T X$，且 $A$ 是对称矩阵。使用公式 $\nabla_w (w^T A w) = (A + A^T)w = 2A w$，所以这一项梯度为 $2X^T X w$。
> 
> 三项相加，立即得到：$ \nabla_w L(w) = -2X^T y + 2X^T X w $

令梯度为零：

$$ -2X^T y + 2X^T X w = 0 $$

$$ X^T X w = X^T y $$

这就是著名的**正规方程（Normal Equation）** 。**当 $X^T X$ 可逆时**：

$$ \boxed{w = (X^T X)^{-1} X^T y} $$

> [!note] 关于闭式解的存在前提
> 
> 这个闭式解要求 **$X^T X$ 是满秩的**（即特征之间不存在完全共线性）。如果特征数量**大于**样本数量（$p > n$），或者存在**多重共线性**，$X^T X$ 就是**奇异矩阵**，无法求逆。

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

### 2.1 正则化

当 **$X^T X$ 不可逆**时（特征数 > 样本数，或存在多重共线性），OLS 无法求解。即使可逆，模型也可能**过拟合** —— 在训练集上表现很好，但在新数据上很差。

**正则化（Regularization）** 的核心思想是：**在损失函数中加入一个惩罚项，限制模型参数的大小，从而防止过拟合**。

### 2.2 Ridge 的数学形式

**Ridge 回归**（又称 **L2 正则化**）在 **OLS 损失函数**上增加了**权重平方和**作为惩罚项：

$$ L_{\text{ridge}}(w) = \|y - Xw\|_2^2 + \lambda \|w\|_2^2 $$

其中 $\lambda \geq 0$ 是**正则化强度超参数**。

写成展开形式：

$$ L_{\text{ridge}}(w) = y^T y - 2y^T X w + w^T X^T X w + \lambda w^T w $$

对 $w$ 求梯度：

$$ \nabla_w L_{\text{ridge}}(w) = -2X^T y + 2X^T X w + 2\lambda w $$

令梯度为零：

$$ (X^T X + \lambda I) w = X^T y $$

$$ \boxed{w = (X^T X + \lambda I)^{-1} X^T y} $$

> [!note] Ridge 能处理不可逆矩阵的根本原因
> 
> 即使 $X^T X$ **不可逆**，加上 $\lambda I$ 后矩阵**一定可逆**（因为 $\lambda > 0$ 时，$X^T X + \lambda I$ 是**正定矩阵**）。这就是 Ridge 能处理**不可逆矩阵**的根本原因。

### 2.3 Ridge 的几何解释

Ridge 等价于在以下约束下**最小化残差平方和**：

$$ \sum_{j=1}^{p} w_j^2 \leq t $$

在二维空间中，L2 约束区域是一个**圆形**。圆形的边界是光滑的，**最优解可能落在圆周的任何位置**，但很难正好落在坐标轴上，因此 Ridge 通常不会将系数压缩到 0 —— 它做的是**收缩（shrinkage）**，而不是**选择（selection）**。

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

**Lasso**（Least Absolute Shrinkage and Selection Operator）使用 **L1 正则化**，惩罚项是**权重的绝对值之和**：

$$ L_{\text{lasso}}(w) = \|y - Xw\|_2^2 + \lambda \|w\|_1 = \|y - Xw\|_2^2 + \lambda \sum_{j=1}^{p} |w_j| $$

与 Ridge 不同，**Lasso 的损失函数在 $w_j=0$ 处不可微**，因此**没有简单的闭式解**。

### 3.2 Lasso 产生稀疏解的原因

**1. 几何解释**

**Lasso 等价于在约束 $\sum_{j=1}^{p} |w_j| \leq t$ 下最小化残差平方和**。

在二维空间中，L1 约束区域是一个**菱形**。菱形的顶点在坐标轴上。当目标函数的**等高线（椭圆）与约束区域相切**时，**切点很可能落在菱形的顶点上**，此时对应的系数就被压缩为 **0**。

随着维度增加，多维菱形的“角”越来越多，**系数被置零的概率就越大**。这就是 Lasso 能够**自动进行特征选择**的原因——它会**产生稀疏解**。

**2. L1 产生稀疏解而 L2 不产生的原因**

- **L2（圆形）**：边界**光滑**，切点落在坐标轴上的**概率为 0**
- **L1（菱形）**：边界**有尖角**，切点很**容易**落在尖角上（即**坐标轴**）

**总结：L1 的几何形状有“角”，所以容易产生零系数**。


在3.3节现有推导基础上，我从专业角度为你补充了三块核心内容，旨在将“跳跃的数学步骤”彻底展开，并赋予其直观的统计含义。请查收修订后的完整 **3.3节**：

---

### 3.3 软阈值算子（Soft-Thresholding Operator）

虽然 Lasso 没有闭式解，但在**单变量**情况下，我们可以推导出**解析解**。

考虑**单变量 Lasso 问题**（没有截距）：

$$ \min_{w} \frac{1}{2}(y - Xw)^2 + \lambda |w| $$

其中 **$X$ 和 $y$ 都是标量**（为简单起见，假设 $X^T X = 1$，即**特征已标准化**）。

目标是：

$$ \min_{w} \frac{1}{2}(y - Xw)^2 + \lambda |w| $$

**次梯度（Subgradient）条件的完整展开**

由于**绝对值函数在 $w=0$ 处不可导**，我们需要引入**次梯度** $\partial |w|$。对于绝对值函数，其次导数为：

$$
\partial |w| = \begin{cases} 
\{1\}, & w > 0 \\
[-1, 1], & w = 0 \\
\{-1\}, & w < 0 
\end{cases} 
$$

**令 $z = X^T y$ 且利用标准化条件 $X^T X = 1$**，损失函数的**次梯度**为：

$$ \partial_w L_{\text{lasso}} = -X(y - Xw) + \lambda \partial |w| = (w - z) + \lambda \partial |w| $$

根据**凸优化的最优性条件**（$0 \in \partial f(w)$），分三种情况讨论：

1. **当 $w > 0$ 时**：$\partial |w| = 1$，令 $w - z + \lambda = 0$，得 $w = z - \lambda$。为保证 $w>0$，须有 $z > \lambda$。
2. **当 $w < 0$ 时**：$\partial |w| = -1$，令 $w - z - \lambda = 0$，得 $w = z + \lambda$。为保证 $w<0$，须有 $z < -\lambda$。
3. **当 $w = 0$ 时**：$\partial |w| = [-1, 1]$，令 $0 \in (0 - z) + \lambda[-1, 1]$，即 $z \in [-\lambda, \lambda]$ 时成立，等价于 $|z| \leq \lambda$。

综合三种情况，得到**软阈值算子（Soft-Thresholding Operator）**：

$$ \boxed{S(z, \lambda) = \text{sign}(z) \cdot \max(|z| - \lambda, 0)} $$

其中 **$z = X^T y$（在标准化特征下，$z$ 也等于普通最小二乘的单变量解 $w_{OLS}$）**。

> [!note] z 的统计直觉与“软阈值”的物理意义
> 
> **$z$ 的本质**：在**标准化**特征下，$z = X^T y$ 实际上就是**特征 $X$ 与响应变量 $y$ 的协方差（相关性）**。它衡量了当前特征对预测结果的边际贡献强度。
> 
> **阈值的逻辑**：
> - 如果相关性 $|z| \leq \lambda$，说明该特征的贡献强度**被正则化惩罚完全“淹没”**，模型判定其为**噪声**，系数直接置零（**实现特征筛选**）。
> - 如果相关性 $|z| > \lambda$，说明该特征**有效**，但系数**不是直接取 $z$，而是向零收缩** $\lambda$ 的量（减去 $\lambda$）。这种“收缩”不同于 Ridge 的等比缩放，它是**绝对值上的硬性削减**，因此得名“**软阈值**”（Soft-Thresholding）。

**代码实现**：

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

对于多维 Lasso，通常使用**坐标下降法（Coordinate Descent）** 求解。核心思想是**每次只优化一个参数，固定其他所有参数**。这样一来，原本复杂的 $p$ 维优化问题，就被拆解成了无数个**一维的单变量 Lasso 问题**。

在**第 $t$ 轮迭代**中，当我们准备更新**第 $j$ 个特征对应的权重 $w_j$** 时：

- 第一步：计算**当前残差与第 $j$ 个特征的“相关系数”**（专业叫法叫**偏残差**）。这个值就相当于 3.3 节里的 $z$，它衡量了**剔除了其他特征影响后，当前这个特征还有多大的贡献潜力**。
- 第二步：把贡献潜力套入 3.3 节的**软阈值公式** $S(z, \lambda)$ 里。
  - 如果潜力值**很小**（$|z| \le \lambda$），软阈值公式直接输出 **0** —— **把这个特征踢出模型**；
  - 如果潜力值**足够大**（$|z| > \lambda$），软阈值公式就输出 $z - \lambda$ 或 $z + \lambda$ —— **保留特征，但强制砍掉 $\lambda$ 的冗余分量**。

即在坐标下降法中，软阈值算子会逐坐标地对特征进行“**相关性截断**”，从而**在多维场景中高效地产生稀疏解**。这使得 Lasso 在**超高维数据**（$p \gg n$）中依然具备强大的**特征选择**能力。

对于 Lasso 的第 $j$ 个坐标更新：

$$ w_j^{(new)} = S\left(\frac{1}{n} \sum_{i=1}^{n} X_{ij} (y_i - \sum_{k \neq j} X_{ik} w_k), \lambda\right) $$

即**计算当前残差与第 $j$ 个特征的相关性**，然后应用**软阈值**。

> [!note] 和 Ridge 回归的直观对比：
> - **Ridge** 的坐标更新是给系数乘以一个**小于 1 的固定比例**（如 \(0.9 \times w_j\)），系数只会**无限逼近 0**，但**永远到不了 0**。
> - **Lasso** 的坐标更新是硬生生的“**阈值截断**”，只要贡献潜力不够大，系数**直接精确等于 0**。
> 这个“绝对零”的特性，就是**坐标下降法让 Lasso 产生稀疏解**的工程实现。

scikit-learn 的 Lasso 实现默认就使用**坐标下降法**。

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

### 4.1 结合 L1 和 L2 的原因

- **Lasso（L1）**：能做**特征选择**，产生**稀疏解**。但在特征高度相关时，Lasso 会随机选择其中一个，导致**不稳定**。
- **Ridge（L2）**：**系数收缩但不置零**，在**相关**特征上表现**稳定**。

**ElasticNet** 结合了两者的优点。

### 4.2 ElasticNet 的数学形式

ElasticNet 的损失函数为：

$$ L_{\text{elastic}}(w) = \frac{1}{2n}\|y - Xw\|_2^2 + \alpha \cdot \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2}\|w\|_2^2 $$

其中：
- $\alpha$：**整体正则化强度**
- $\rho$（即 `l1_ratio`）：L1 和 L2 的**混合比例**
  - $\rho = 0$：纯 Ridge（L2）
  - $\rho = 1$：纯 Lasso（L1）
  - $0 < \rho < 1$：两者的组合

ElasticNet 也可以用**坐标下降法**求解。scikit-learn 的 `ElasticNet` 默认使用坐标下降。

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

**MSE 是线性回归最常用的损失函数**。它对**大误差给予更大的惩罚**（平方放大）。

### 5.2 RMSE（均方根误差）

$$ \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

RMSE 只是 MSE **开方**，**量纲与原始数据一致**，更易解释。

### 5.3 MAE（平均绝对误差）

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

**MAE 是线性的**，所有误差被同等对待。

### 5.4 异常值敏感性对比

这是三者最核心的区别：

| 指标 | 对异常值的敏感性 | 原因 |
|------|-----------------|------|
| **MSE/RMSE** | **非常敏感** | **平方运算**放大了大误差的影响 |
| **MAE** | **相对鲁棒** | **绝对值**不放大误差 |

- **直观解释**：最小化 **MSE** 等价于预测**均值 —— 均值对异常值敏感**；最小化 **MAE** 等价于预测**中位数 —— 中位数对异常值鲁棒**
- **实际建议**：如果异常值代表**重要的业务场景**（如欺诈检测），用 **MSE/RMSE**；如果异常值只是**噪声/脏数据**，用 **MAE**；如果想兼顾两者，可以用 **Huber 损失**（在误差小时用平方，误差大时用绝对值）

**代码实现**：

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

> [!note] 总结
> 
> | 方法 | 正则项 | 闭式解 | 稀疏性 | 适用场景 |
> |------|--------|--------|--------|----------|
> | **OLS** | 无 | ✅ $ (X^TX)^{-1}X^Ty $ | ❌ | **特征数 < 样本数，无共线性** |
> | **Ridge** | L2: $\lambda\|w\|_2^2$ | ✅ $ (X^TX + \lambda I)^{-1}X^Ty $ | ❌ | **多重共线性，特征相关** |
> | **Lasso** | L1: $\lambda\|w\|_1$ | ❌（坐标下降） | ✅ | **特征选择，高维数据** |
> | **ElasticNet** | L1 + L2 | ❌（坐标下降） | ✅（可控） | **相关特征分组，兼顾选择与稳定** |
> 
> 核心要点回顾
> 
> 1. **最小二乘法**的闭式解 $w = (X^TX)^{-1}X^Ty$ 是线性回归的基石，但要求 $X^TX$ **可逆**。
> 2. **Ridge（L2）** 通过在 $X^TX$ 对角线加 $\lambda$ 保证**矩阵可逆**，系数收缩但不归零。
> 3. **Lasso（L1）** 的**稀疏性**来自两个层面：
>   - **几何**：L1 约束是**菱形**，更容易在坐标轴上相切
>   - **算法**：**软阈值算子** $S(z, \lambda) = \text{sign}(z)\max(|z|-\lambda, 0)$ 会将小系数直接置零
> 4. **ElasticNet** 结合 L1 和 L2，**在特征选择的同时保持分组稳定性**。
> 5. **MSE/RMSE 对异常值敏感**（平方放大误差），**MAE 相对鲁棒**（绝对值不放大）。