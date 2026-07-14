---
title: Logistic Regression —— 逻辑回归与Softmax多分类
published: 2025-07-04
description: 系统讲解逻辑回归与Softmax多分类的核心原理与完整数学推导。涵盖Sigmoid函数的概率映射与导数特性、对数几率与线性决策边界、最大似然估计推导二分类交叉熵损失及梯度下降求解、Softmax回归从二分类到多分类的自然推广、Log-Sum-Exp数值稳定性技巧，以及宏平均与微平均F1-Score在多分类评价中的适用场景与代码实现。
cover: "/assets/images/posts/logistic_regression.png"
coverInContent: false
tags: [逻辑回归, Softmax, 交叉熵损失, 梯度下降]
category: Machine_Learning
draft: false
---

# Logistic Regression —— 逻辑回归与Softmax多分类

## 前言

在上一篇文章中，我们深入探讨了线性回归及其正则化变体。线性回归解决的是**回归**问题——预测一个连续数值。但现实世界中大量的机器学习问题其实是**分类**问题：这封邮件是垃圾邮件还是正常邮件？这个肿瘤是良性还是恶性？这张图片里是猫、狗还是鸟？

**逻辑回归（Logistic Regression）** 就是解决二分类问题最经典的算法。尽管名字里带着“回归”，它本质上是一个分类模型。而当我们需要处理**三个或更多类别**时，逻辑回归的自然推广就是 **Softmax回归**（也称为多项逻辑回归）。

这篇文章，我们会从 Sigmoid 函数出发，一步步推导逻辑回归的损失函数（最大似然估计），理解“对数几率”的含义和决策边界，然后扩展到 Softmax 多分类，最后讨论多分类模型的评价指标。

---

## 一、从线性回归到逻辑回归

### 1.1 为什么线性回归不能做分类？

线性回归的输出是一个**连续值**，取值范围是 $(-\infty, +\infty)$。而二分类问题的输出是离散的类别标签，比如 $y \in \{0, 1\}$。

如果我们直接用线性回归 $z = w^T x$ 去拟合 0/1 标签，会遇到两个问题：

1. **预测值可能远超出 [0, 1] 区间**，无法解释为概率
2. 模型对异常值极其敏感

因此，我们需要一个**映射函数**，将线性回归的输出 $z \in (-\infty, +\infty)$ **压缩**到 $[0, 1]$ 区间，使其可以解释为概率。

### 1.2 Sigmoid 函数

最常用的映射函数就是 **Sigmoid 函数**（也叫 Logistic 函数）：

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

它的图像是一个优美的 **S 形曲线**，具有以下重要性质：

1. **值域为 (0, 1)**：可以将任意实数映射为概率
2. **单调递增**：$z$ 越大，输出越接近 1
3. **中心对称**：$\sigma(0) = 0.5$，关于点 (0, 0.5) 对称
4. **处处可导、光滑连续**
5. **导数形式极其简洁**：

$$ \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z)) $$

这个导数特性非常关键——它意味着我们可以**根据函数值直接计算导数值**，而不需要额外的计算，在梯度下降中效率极高。

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z), label='σ(z)')
plt.plot(z, sigmoid_derivative(z), label="σ'(z)", linestyle='--')
plt.axhline(0.5, color='gray', linestyle=':')
plt.axvline(0, color='gray', linestyle=':')
plt.xlabel('z')
plt.ylabel('value')
plt.legend()
plt.title('Sigmoid 函数及其导数')
plt.show()
```

### 1.3 逻辑回归的模型形式

逻辑回归的模型假设是：

$$ h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}} $$

其中 $h_\theta(x)$ 表示在给定参数 $\theta$ 和输入 $x$ 的条件下，样本属于**正类（y=1）** 的概率：

$$ P(y=1 \mid x; \theta) = h_\theta(x) $$

$$ P(y=0 \mid x; \theta) = 1 - h_\theta(x) $$

将两个式子统一写成：

$$ P(y \mid x; \theta) = h_\theta(x)^y \cdot (1 - h_\theta(x))^{1-y}, \quad y \in \{0, 1\} $$

---

## 二、对数几率（Log Odds）与决策边界

### 2.1 什么是对数几率？

**几率（Odds）** 是指事件发生的概率与不发生的概率之比：

$$ \text{Odds} = \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = \frac{h_\theta(x)}{1 - h_\theta(x)} $$

对几率取自然对数，得到**对数几率（Log Odds）** ，也叫 **Logit**：

$$ \text{Logit} = \log\left(\frac{h_\theta(x)}{1 - h_\theta(x)}\right) $$

将 $h_\theta(x) = \sigma(\theta^T x)$ 代入：

$$ \log\left(\frac{h_\theta(x)}{1 - h_\theta(x)}\right) = \log\left(\frac{\frac{1}{1+e^{-\theta^T x}}}{\frac{e^{-\theta^T x}}{1+e^{-\theta^T x}}}\right) = \log(e^{\theta^T x}) = \theta^T x $$

**这就是逻辑回归被称为“对数几率回归”的原因**：模型实际上是用线性回归 $\theta^T x$ 去逼近真实标签的**对数几率**。

### 2.2 决策边界

从上面的推导可以看出：

$$ h_\theta(x) > 0.5 \iff \theta^T x > 0 $$

$$ h_\theta(x) < 0.5 \iff \theta^T x < 0 $$

因此，**决策边界**就是 $\theta^T x = 0$ 这条线（或超平面）。

**关键洞察**：逻辑回归的决策边界**本质上是线性的**。如果数据本身不是线性可分的，我们需要通过**特征工程**（如添加多项式特征、交互特征）来构造非线性决策边界。

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 生成二分类数据
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

# 训练逻辑回归
model = LogisticRegression()
model.fit(X, y)

# 决策边界: θ0 + θ1*x1 + θ2*x2 = 0 => x2 = -(θ0 + θ1*x1) / θ2
coef = model.coef_[0]
intercept = model.intercept_[0]
# 绘制决策边界...
```

---

## 三、最大似然估计：推导逻辑回归的损失函数

### 3.1 为什么要用最大似然估计？

在线性回归中，我们通过**最小化均方误差（MSE）** 来求解参数。但在逻辑回归中，**MSE 不再是凸函数**，用梯度下降容易陷入局部最优。

因此，逻辑回归采用**最大似然估计（Maximum Likelihood Estimation, MLE）** 来估计参数。

### 3.2 似然函数

给定训练集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^{n}$，其中 $y^{(i)} \in \{0, 1\}$。

根据模型假设，每个样本的概率为：

$$ P(y^{(i)} \mid x^{(i)}; \theta) = h_\theta(x^{(i)})^{y^{(i)}} \cdot (1 - h_\theta(x^{(i)}))^{1-y^{(i)}} $$

假设样本之间**独立同分布**，则**似然函数**为所有样本概率的乘积：

$$ L(\theta) = \prod_{i=1}^{n} h_\theta(x^{(i)})^{y^{(i)}} \cdot (1 - h_\theta(x^{(i)}))^{1-y^{(i)}} $$

### 3.3 对数似然与损失函数

直接最大化乘积不方便（容易数值下溢），取对数将乘积变为求和：

$$ \ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)})) \right] $$

机器学习中习惯**最小化损失函数**，所以在对数似然前加负号，得到**交叉熵损失（Cross-Entropy Loss）** ：

$$ \boxed{J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)})) \right]} $$

### 3.4 梯度下降求解

对 $J(\theta)$ 求偏导：

$$ \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} $$

这个形式**与线性回归的梯度形式惊人地相似**！区别仅在于线性回归的预测值是 $\theta^T x$，而逻辑回归的预测值是 $h_\theta(x) = \sigma(\theta^T x)$。

梯度下降更新公式：

$$ \theta_j := \theta_j - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} $$

```python
class LogisticRegressionGD:
    """使用梯度下降的逻辑回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]  # 添加截距项
        self.theta = np.zeros(n_features + 1)
        
        for _ in range(self.n_iterations):
            z = X_b @ self.theta
            h = self.sigmoid(z)
            gradient = (1 / n_samples) * (X_b.T @ (h - y))
            self.theta -= self.lr * gradient
        
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b @ self.theta)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 四、Softmax 回归：从二分类到多分类

### 4.1 为什么要用 Softmax？

当类别数 $K > 2$ 时，二分类逻辑回归无法直接使用。有两种常见的处理策略：

1. **One-vs-Rest（OvR）** ：为每个类别训练一个二分类器，将该类 vs 其他所有类
2. **Softmax 回归**（多项逻辑回归）：直接建模多分类问题

Softmax 回归是逻辑回归在多分类问题上的**自然推广**。它假设类别之间**互斥**（每个样本只属于一个类别）。

### 4.2 Softmax 函数的定义

给定 $K$ 个类别的**得分**（logits）$z_1, z_2, ..., z_K$，Softmax 函数将其转换为概率分布：

$$ \text{Softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k = 1, 2, ..., K $$

Softmax 的输出满足概率分布的两个性质：
- 每个输出在 (0, 1) 之间
- 所有输出之和为 1

在 Softmax 回归中，每个类别 $k$ 有自己的权重向量 $\theta_k$，得分 $z_k = \theta_k^T x$：

$$ P(y=k \mid x; \Theta) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^{K} e^{\theta_j^T x}} $$

### 4.3 最大似然推导

对于多分类问题，标签 $y \in \{1, 2, ..., K\}$。用**独热编码（One-Hot Encoding）** 表示：如果样本属于类别 $k$，则 $y_k = 1$，其余为 0。

似然函数为：

$$ L(\Theta) = \prod_{i=1}^{n} \prod_{k=1}^{K} P(y=k \mid x^{(i)}; \Theta)^{\mathbb{I}(y^{(i)}=k)} $$

取负对数，得到**多分类交叉熵损失**：

$$ \boxed{J(\Theta) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{I}(y^{(i)}=k) \cdot \log\left(\frac{e^{\theta_k^T x^{(i)}}}{\sum_{j=1}^{K} e^{\theta_j^T x^{(i)}}}\right)} $$

对 $\theta_k$ 求梯度：

$$ \frac{\partial J(\Theta)}{\partial \theta_k} = \frac{1}{n} \sum_{i=1}^{n} (P(y=k \mid x^{(i)}) - \mathbb{I}(y^{(i)}=k)) \cdot x^{(i)} $$

这个形式与二分类逻辑回归的梯度**完全一致**——只是从 2 类扩展到了 K 类。

```python
class SoftmaxRegression:
    """使用梯度下降的Softmax回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
    
    def softmax(self, Z):
        # Z: (n_samples, K)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 数值稳定
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        # 将标签转为独热编码
        y_onehot = np.eye(self.K)[y]
        
        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.theta = np.zeros((n_features + 1, self.K))
        
        for _ in range(self.n_iterations):
            scores = X_b @ self.theta  # (n_samples, K)
            probs = self.softmax(scores)
            gradient = (1 / n_samples) * (X_b.T @ (probs - y_onehot))
            self.theta -= self.lr * gradient
        
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.softmax(X_b @ self.theta)
    
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
```

---

## 五、Softmax 的数值稳定性：Log-Sum-Exp Trick

### 5.1 数值不稳定的原因

直接计算 Softmax 时，需要计算 $e^{z_k}$。当 $z_k$ 很大时（比如 $z_k = 1000$），$e^{1000}$ 会**溢出（overflow）** ，Python 会返回 `inf`。当所有 $z_k$ 都是很大的负数时，所有 $e^{z_k}$ 都趋近于 0，导致**下溢（underflow）** 。

```python
# 不安全的实现 —— 会溢出！
def bad_softmax(scores):
    exp_scores = np.exp(scores)  # 如果 scores 包含 1000，这里就爆了
    return exp_scores / exp_scores.sum()
```

### 5.2 Log-Sum-Exp Trick

解决方案是利用一个恒等变换：

$$ \log\sum_{k=1}^{K} e^{z_k} = m + \log\sum_{k=1}^{K} e^{z_k - m} $$

其中 $m = \max_k z_k$。这个等式对**任意** $m$ 都成立。

取 $m$ 为所有得分中的最大值后，$z_k - m \leq 0$，所以 $e^{z_k - m} \in (0, 1]$，**绝对不会溢出**。

对于 **Log-Softmax**（即 $\log(\text{Softmax})$，在计算交叉熵时常用）：

$$ \log(\text{Softmax}(z)_k) = z_k - \log\sum_{j=1}^{K} e^{z_j} = z_k - m - \log\sum_{j=1}^{K} e^{z_j - m} $$

```python
def stable_softmax(scores):
    """数值稳定的 Softmax 实现"""
    m = np.max(scores, axis=-1, keepdims=True)
    safe_scores = scores - m
    exp_scores = np.exp(safe_scores)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

def stable_log_softmax(scores):
    """数值稳定的 Log-Softmax 实现"""
    m = np.max(scores, axis=-1, keepdims=True)
    safe_scores = scores - m
    log_sum_exp = m + np.log(np.sum(np.exp(safe_scores), axis=-1, keepdims=True))
    return safe_scores - log_sum_exp

# 测试
scores = np.array([1000, 1001, 1002])
print("Stable softmax:", stable_softmax(scores))
print("Stable log_softmax:", stable_log_softmax(scores))
# 输出: 概率分布合理，不会溢出
```

### 5.3 实际应用中的最佳实践

在实际的深度学习框架（PyTorch、TensorFlow）中，**永远不要单独计算 Softmax 再取 Log**，而是直接使用 `log_softmax` 或 `CrossEntropyLoss`。

- `CrossEntropyLoss` 内部已经集成了 `LogSoftmax` + `NLLLoss`，数值稳定且高效
- 自己实现时，务必使用 Log-Sum-Exp Trick

---

## 六、多分类评价指标：宏平均与微平均 F1-Score

### 6.1 从二分类到多分类的混淆矩阵

在二分类中，我们有 TP、FP、TN、FN。在多分类中，**每个类别**都有自己的 TP、FP、FN：

- **TP$_k$**：真实类别为 $k$，预测也为 $k$ 的样本数（混淆矩阵主对角线）
- **FP$_k$**：真实类别不是 $k$，但预测为 $k$ 的样本数
- **FN$_k$**：真实类别为 $k$，但预测不是 $k$ 的样本数

每个类别的 Precision 和 Recall 为：

$$ \text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k} $$

每个类别的 F1-Score 为：

$$ F1_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} $$

### 6.2 宏平均（Macro-Average）

宏平均**对每个类别平等对待**，先计算每个类别的指标，再取算术平均：

$$ F1_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} F1_k $$

**特点**：
- 每个类别权重相同，**不受类别不平衡影响**
- 少数类的表现会被同等重视
- 适合**每个类别都很重要**的场景

### 6.3 微平均（Micro-Average）

微平均**先汇总所有类别的 TP、FP、FN**，再统一计算：

$$ \text{Precision}_{\text{micro}} = \frac{\sum_k TP_k}{\sum_k TP_k + \sum_k FP_k} $$

$$ \text{Recall}_{\text{micro}} = \frac{\sum_k TP_k}{\sum_k TP_k + \sum_k FN_k} $$

$$ F1_{\text{micro}} = 2 \cdot \frac{\text{Precision}_{\text{micro}} \cdot \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}} $$

**注意**：在多分类单标签问题中，**微平均 F1 等于准确率（Accuracy）** 。

**特点**：
- 每个**样本**被平等对待，大类别主导指标
- 对**类别不平衡**不敏感（实际上是被大类主导了）
- 适合**整体准确性**比每个类别的公平性更重要的场景

### 6.4 代码实现

```python
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

def macro_f1(y_true, y_pred, num_classes):
    """手动计算宏平均 F1"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    f1s = []
    for k in range(num_classes):
        TP = cm[k, k]
        FP = cm[:, k].sum() - TP
        FN = cm[k, :].sum() - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)

def micro_f1(y_true, y_pred, num_classes):
    """手动计算微平均 F1"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    total_TP = np.trace(cm)
    total_FP = cm.sum() - total_TP  # 在多分类中，总FP = 总FN
    total_FN = cm.sum() - total_TP
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# 示例
y_true = np.array([0, 0, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 1, 2, 2, 0, 1])

print(f"Macro F1 (手动): {macro_f1(y_true, y_pred, 3):.4f}")
print(f"Micro F1 (手动): {micro_f1(y_true, y_pred, 3):.4f}")
print(f"Macro F1 (sklearn): {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"Micro F1 (sklearn): {f1_score(y_true, y_pred, average='micro'):.4f}")
```

### 6.5 什么时候用哪个？

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 类别均衡 | Macro 或 Micro 差别不大 | 两者结果相近 |
| 类别严重不平衡，关心少数类 | **Macro** | 平等对待每个类别 |
| 类别严重不平衡，关心整体表现 | **Micro** | 受多数类主导，反映整体准确性 |
| 不知道选哪个 | 两个都报告 | 全面展示模型表现 |

---

## 七、总结

| 概念 | 二分类（逻辑回归） | 多分类（Softmax 回归） |
|------|-------------------|----------------------|
| **激活函数** | Sigmoid: $\sigma(z) = 1/(1+e^{-z})$ | Softmax: $p_k = e^{z_k}/\sum e^{z_j}$ |
| **输出** | 1 个概率（正类） | K 个概率（和为 1） |
| **损失函数** | 二分类交叉熵 | 多分类交叉熵 |
| **参数数量** | $p+1$ | $K \times (p+1)$ |
| **梯度形式** | $(h-y)x$ | $(p_k - \mathbb{I}_{y=k})x$ |

### 核心要点回顾

1. **Sigmoid 函数**将线性输出映射到 (0, 1) 区间，其导数 $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 形式简洁，是梯度下降高效计算的关键。

2. **对数几率（Log Odds）** 揭示了逻辑回归的本质：用线性模型 $\theta^T x$ 去逼近真实标签的对数几率。决策边界 $\theta^T x = 0$ 是线性的。

3. **最大似然估计**推导出交叉熵损失函数，梯度形式与线性回归惊人地相似，区别仅在于预测函数从 $\theta^T x$ 变成了 $\sigma(\theta^T x)$。

4. **Softmax 回归**是逻辑回归在多分类（类别互斥）场景下的自然推广，使用 **Log-Sum-Exp Trick** 保证数值稳定性——永远不要单独计算 Softmax 再取 Log。

5. **宏平均（Macro）** 平等对待每个类别，适合不平衡数据；**微平均（Micro）** 平等对待每个样本，适合关注整体准确性的场景。