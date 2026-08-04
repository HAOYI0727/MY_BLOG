---
title: Adaptive Boosting (AdaBoost) —— 自适应提升
published: 2025-07-18
description: 系统讲解AdaBoost算法的完整数学原理：从Boosting的串行纠错思想出发，推导前向分步加法模型与指数损失函数的等价性，通过最优化求解揭示弱分类器权重和样本权重更新公式的数学来源；深入分析指数损失与0-1损失的一致性及其对异常值的敏感性；阐明AdaBoost是GBM在指数损失下的特例这一底层联系。
cover: "/assets/images/posts/adaboost.png"
coverInContent: false
tags: [AdaBoost, Boosting, 前向分步加法模型, 弱分类器, 集成学习, 机器学习]
category: Machine_Learning
draft: false
---

# Adaptive Boosting (AdaBoost) —— 自适应提升

## 引言

上一篇文章中，我们讨论了 **Bagging 与随机森林** —— 通过 **Bootstrap 采样**构建**多棵并行**的决策树，以**投票**方式降低方差。这种**并行集成**的策略有效驯服了单棵决策树的高方差问题，其核心在于“**让多棵树独立生长，平等表决**”。

**AdaBoost** 则走向了另一条截然不同的路径 —— **串行集成**。它不再让基学习器独立生长，而是让它们**按顺序依次生成**，每一棵新树都聚焦于**前序模型犯错的样本**。这种“**串行纠错**”的直觉，可以用一个生动的比喻来概括：**“错题本机制”** —— 做错的题标记出来重点复习，再做新题，再标记新错题，反复迭代，最终**将一系列“表现平平”的弱分类器组合成一个强分类器**。

**AdaBoost** 由 Freund 和 Schapire 于 1997 年正式提出，是 Boosting 家族最具影响力的奠基之作。其数学框架可被精炼地概括为“**前向分步加法模型 + 指数损失函数**” —— 每一轮通过**解析优化指数损失**，同时推导出两个核心更新公式：**弱分类器的权重 $\alpha_t$**（错误率越低、话语权越大）和**样本权重 $D_t$**（被错分的样本下一轮获得更高关注）。

我们将从这一框架出发，完成**从损失函数到两个权重更新公式**的完整推导，并揭示其与后续 **GBM** 之间的深层联系 —— AdaBoost 本质上是 GBM 在**指数损失**下的特例，而 GBM 则将损失函数从指数泛化为**任意可微损失**，将优化方式从解析求解扩展为**函数空间中的梯度下降**。

> [!note]
> 
> 读完本文，你将彻底理解“**串行纠错**”的数学本质，并厘清 **AdaBoost 与 GBM** 之间的继承与泛化关系。
> 
> 下一篇文章，我们将进入 **Gradient Boosting Machine（GBM）** ——它将 AdaBoost 的**指数损失**泛化为**任意可微损失**，把“错题本”的直觉升级为“**梯度下降**”的通用框架，为 **XGBoost 与 LightGBM** 等工程级实现奠定理论基础。


---

## 一、从决策树到Boosting

### 1.1 单棵决策树的困境

在上一篇文章中，我们详细讨论了**决策树的构建与剪枝**。单棵**完全生长的决策树**具有以下特点：
- **低偏差**：能够**完美拟合**训练数据
- **高方差**：训练数据的**微小变化**会导致完全不同的树

这就是决策树容易**过拟合**的根本原因。如果我们直接用一棵完整的决策树去做分类，在训练集上可能表现完美，但在测试集上往往“原形毕露”。

### 1.2 两个集成方向：并行 vs 串行

如何克服单棵决策树的不稳定性？集成学习给出了两个不同的答案：

| | **并行集成（Bagging）** | **串行集成（Boosting）** |
|---|---|---|
| **代表算法** | **随机森林** | **AdaBoost、GBM** |
| **构建方式** | 多棵树**独立**生长 | 多棵树**串行**生成 |
| **核心思想** | 平等投票，**降低方差** | 串行纠错，**降低偏差** |
| **树的特点** | **完全生长的深树** | **浅层弱学习器** |

**Bagging（随机森林）** ：让多棵树**并行**生长，每棵树都是“**独立的专家**”，最后**平等投票**。就像专家会诊 —— 多位专家独立诊断后投票决定。

**Boosting（AdaBoost）** ：让多棵树**串行**生长，后一棵树专门用来**修正前一棵树的错误**。就像渐进式诊疗 —— 先做初步诊断，发现错误后针对性复查，不断修正。

AdaBoost属于后者——**串行集成**。它的核心思想是：用**一系列弱分类器**（每棵树的深度都很浅，分类能力仅比随机猜测稍好一点点），通过**串行地调整样本权重**，**让每个新的弱分类器都重点关注前一轮被分错的样本**，最终将这些弱分类器**加权组合成一个强分类器**。

### 1.3 “错题本”的直觉

AdaBoost最直观的理解方式就是 **“错题本”机制**。

想象你在准备一场考试：
- **第一轮**：你做完一套模拟卷（训练一个弱分类器），发现某些题做错了
- **第二轮**：你翻开错题本，**重点复习**那些做错的题（提高错题的权重），再做一套新卷子
- **第三轮**：又产生了新的错题，继续重点复习……
- **最终**：你把所有做过的卷子的经验**综合起来**（加权组合），形成一套完整的知识体系

AdaBoost的每一轮迭代都在做同样的事情 —— **增加前一个基学习器在训练过程中预测错误样本的权重，使得后续基学习器更加关注这些被错误标注的训练样本，尽可能纠正这些错误。**

但需要注意的是，AdaBoost**并不是只关注错题**。它只是**更偏向错题**，而不是只看错题。就像复习时不能只看错题，也得看看做对的题 —— 只是错题的权重大一些。

---

## 二、AdaBoost的算法流程

### 2.1 符号约定

在正式进入算法之前，首先需要约定符号：

- 训练集：$\{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$，其中 $y_i \in \{-1, +1\}$
- $T$：**迭代次数（弱分类器的数量）**
- $D_t(i)$：第 $t$ 轮中第 $i$ 个样本的**权重**
- $h_t(x)$：第 $t$ 轮训练出的**弱分类器**
- $\alpha_t$：第 $t$ 个弱分类器在最终集成模型中的**权重**
- $\epsilon_t$：第 $t$ 个弱分类器的**加权错误率**

### 2.2 完整算法步骤

**步骤1：初始化样本权重**

将所有样本的权重设为相等：$D_1(i) = \frac{1}{N}, \quad i = 1, 2, ..., N$

**步骤2：对 $t = 1, 2, ..., T$ 迭代**

**(a) 训练弱分类器**：在当前样本权重分布 $D_t$ 下，训练一个弱分类器 $h_t(x)$。

**(b) 计算加权错误率**：

$$
\epsilon_t = \sum_{i=1}^{N} D_t(i) \cdot \mathbb{I}(h_t(x_i) \neq y_i)
$$

其中 $\mathbb{I}(\cdot)$ 是指示函数，条件成立时为1，否则为0。

**(c) 计算弱分类器的权重**：

$$
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
$$

**(d) 更新样本权重**：

$$
D_{t+1}(i) = \frac{D_t(i) \cdot \exp(-\alpha_t y_i h_t(x_i))}{Z_t}
$$

其中 $Z_t = \sum_{i=1}^{N} D_t(i) \cdot \exp(-\alpha_t y_i h_t(x_i))$ 是归一化因子，保证 $\sum_i D_{t+1}(i) = 1$。

**步骤3：输出最终强分类器**：

$$
H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)
$$

**代码实现**：

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    """从零实现AdaBoost算法"""
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.weak_classifiers = []
    
    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,), 取值 {-1, +1}
        """
        n_samples = X.shape[0]
        
        # 步骤1：初始化样本权重
        D = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            # 步骤2(a)：训练弱分类器（决策树桩，深度为1）
            weak_clf = DecisionTreeClassifier(max_depth=1)
            weak_clf.fit(X, y, sample_weight=D)
            predictions = weak_clf.predict(X)
            
            # 步骤2(b)：计算加权错误率
            misclassified = (predictions != y)
            epsilon = np.sum(D * misclassified)
            
            # 防止除零或数值不稳定
            if epsilon == 0:
                epsilon = 1e-10
            if epsilon == 1:
                epsilon = 1 - 1e-10
            
            # 步骤2(c)：计算弱分类器的权重
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            
            # 步骤2(d)：更新样本权重
            # 正确分类: 权重乘以 exp(-alpha)，错误分类: 权重乘以 exp(alpha)
            D = D * np.exp(-alpha * y * predictions)
            D = D / np.sum(D)  # 归一化
            
            # 保存结果
            self.alphas.append(alpha)
            self.weak_classifiers.append(weak_clf)
        
        return self
    
    def predict(self, X):
        """预测：加权投票"""
        # 计算所有弱分类器的加权预测之和
        weighted_sum = np.zeros(X.shape[0])
        for alpha, clf in zip(self.alphas, self.weak_classifiers):
            weighted_sum += alpha * clf.predict(X)
        return np.sign(weighted_sum)
```

---

## 三、前向分步加法模型：AdaBoost的数学框架

### 3.1 加法模型

AdaBoost最终得到的模型是一个**加法模型（Additive Model）—— 多个基学习器的线性组合**：

$$
H(x) = \sum_{t=1}^{T} \alpha_t h_t(x)
$$

其中 $h_t(x)$ 是第 $t$ 个**弱分类器**，$\alpha_t$ 是其**权重**。

这个形式看起来简单，但问题在于**如何确定每一轮的 $\alpha_t$ 和 $h_t$？**

### 3.2 前向分步算法

**前向分步算法（Forward Stagewise Algorithm）** 是解决这个问题的核心策略。此算法的核心思想是：**从前向后，每一步只优化当前这一个基分类器，固定之前已经选好的所有基分类器不变**。

具体来说：

1. 初始化 $H_0(x) = 0$
2. 对 $t = 1, 2, ..., T$：
   - 固定 $H_{t-1}(x)$ 不变
   - 选择 $(\alpha_t, h_t)$ 使得损失函数最小：$$ (\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_{i=1}^{N} L(y_i, H_{t-1}(x_i) + \alpha h(x_i)) $$
   - 更新：$H_t(x) = H_{t-1}(x) + \alpha_t h_t(x)$

由此可见：前向分步算法将一个复杂的**全局优化问题**（同时优化所有 $\alpha_t$ 和 $h_t$）转化为 $T$ 个相对简单的**局部优化问题**（每一步只优化一个 $\alpha_t$ 和 $h_t$）。

### 3.3 指数损失函数

AdaBoost使用的损失函数 —— **指数损失函数（Exponential Loss）** ：

$$
L(y, H(x)) = e^{-y H(x)}
$$

其中 $y \in \{-1, +1\}$，$H(x)$ 是模型的**预测值**（实数，不一定是 ±1）。

**使用指数损失的原因**：
1. **数学性质好**：指数函数**连续可微**，便于优化
2. **与0-1损失一致**：最小化指数损失等价于**最小化分类错误率**
3. **推导简洁**：指数损失天然导出了**AdaBoost的权重更新公式**

**定理：AdaBoost算法是前向分步加法算法在以指数函数为损失函数时的特例**。

---

## 四、完整推导：从指数损失到AdaBoost的更新公式

### 4.1 推导弱分类器的权重 $\alpha_t$

假设在第 $t$ 轮，我们已经有了前 $t-1$ 轮的**集成模型** $H_{t-1}(x)$。现在要选择**新的弱分类器** $h_t(x)$ 和**权重** $\alpha_t$，使得**指数损失最小**：

$$
(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_{i=1}^{N} \exp\left(-y_i \left(H_{t-1}(x_i) + \alpha h(x_i)\right)\right)
$$

令 $w_i^{(t)} = \exp(-y_i H_{t-1}(x_i))$（上一轮**更新后的样本权重**，未归一化），则目标函数变为：

$$
\sum_{i=1}^{N} w_i^{(t)} \exp(-\alpha y_i h(x_i))
$$

对于固定的 $\alpha$，最小化上式等价于最小化加权错误率：

$$
\epsilon_t = \frac{\sum_{i=1}^{N} w_i^{(t)} \mathbb{I}(h(x_i) \neq y_i)}{\sum_{i=1}^{N} w_i^{(t)}}
$$

**现在推导 $\alpha_t$ 的表达式**：

将样本分为两类：被 $h$ **正确分类**的（$y_i h(x_i) = 1$）和**错误分类**的（$y_i h(x_i) = -1$）。

目标函数可以写成：

$$
\sum_{i: y_i = h(x_i)} w_i^{(t)} e^{-\alpha} + \sum_{i: y_i \neq h(x_i)} w_i^{(t)} e^{\alpha} = (1 - \epsilon_t) e^{-\alpha} + \epsilon_t e^{\alpha}
$$

（这里假设**权重已经归一化**，即 $\sum w_i^{(t)} = 1$）

对 $\alpha$ 求导并令其为零：

$$
\frac{d}{d\alpha} \left[(1 - \epsilon_t) e^{-\alpha} + \epsilon_t e^{\alpha}\right] = -(1 - \epsilon_t) e^{-\alpha} + \epsilon_t e^{\alpha} = 0
$$

解得：

$$
\boxed{\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)}
$$

这就是AdaBoost中**弱分类器权重**的计算公式。

### 4.2 推导样本权重的更新公式

下列推导样本权重 $w_i^{(t)} = \exp(-y_i H_{t-1}(x_i))$ 的更新规律

在得到 $h_t$ 和 $\alpha_t$ 后：

$$
w_i^{(t+1)} = \exp(-y_i H_t(x_i)) = \exp(-y_i (H_{t-1}(x_i) + \alpha_t h_t(x_i))) = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))
$$

- 当 $h_t(x_i) = y_i$（分类正确）时：$ w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t} $

- 当 $h_t(x_i) \neq y_i$（分类错误）时：$ w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t} $

由于 $\alpha_t > 0$（因为 $\epsilon_t < 0.5$），所以**分类正确的样本权重减小（乘以 $e^{-\alpha_t} < 1$），分类错误的样本权重增大（乘以 $e^{\alpha_t} > 1$）**。

这正是AdaBoost **“错题本”机制**的数学本质 —— **被分错的样本在下一轮获得更高的权重**。

统一写成：

$$
\boxed{w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))}
$$

加上**归一化因子** $Z_t$ 后：

$$
\boxed{D_{t+1}(i) = \frac{D_t(i) \cdot \exp(-\alpha_t y_i h_t(x_i))}{Z_t}}
$$

其中 $Z_t = \sum_{i=1}^{N} D_t(i) \exp(-\alpha_t y_i h_t(x_i))$。

**代码实现**：

```python
# 验证权重更新公式的正确性
def verify_weight_update():
    np.random.seed(42)
    N = 10
    y = np.random.choice([-1, 1], N)
    h = np.random.choice([-1, 1], N)
    epsilon = 0.3
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)
    
    D = np.ones(N) / N
    
    # 更新权重
    D_new_raw = D * np.exp(-alpha * y * h)
    D_new = D_new_raw / np.sum(D_new_raw)
    
    # 检查：错误分类的样本权重是否增大
    misclassified = (h != y)
    correct = (h == y)
    
    print(f"错误分类样本的平均权重变化: {np.mean(D_new[misclassified] / D[misclassified]):.4f}")
    print(f"正确分类样本的平均权重变化: {np.mean(D_new[correct] / D[correct]):.4f}")
    # 错误分类的权重变化 > 1，正确分类的权重变化 < 1

verify_weight_update()
```

### 4.3 指数损失与0-1损失的一致性

考虑**期望指数损失** $\mathbb{E}[e^{-y H(x)}]$，对其关于 $H(x)$ 求偏导：

$$
\frac{\partial}{\partial H} \mathbb{E}[e^{-y H(x)}] = \mathbb{E}[-y e^{-y H(x)}] = 0
$$

展开期望公式得：

$$
P(y=1) \cdot e^{-H} - P(y=-1) \cdot e^{H} = 0
$$

整理得：

$$
\frac{P(y=1)}{P(y=-1)} = e^{2H} \quad \xrightarrow{\text{两边取自然对数}} \quad H(x) = \frac{1}{2} \ln \frac{P(y=1)}{P(y=-1)}
$$

因此：

$$
\text{sign}(H(x)) = 
\begin{cases}
+1, & P(y=1) > P(y=-1) \\
-1, & P(y=1) < P(y=-1)
\end{cases}
$$

这意味着：**最小化指数损失得到的分类器，恰好是贝叶斯最优分类器**。指数损失是0-1损失的一个**一致的替代损失函数（Surrogate Loss Function）** 。

---

## 五、AdaBoost与GBM的底层联系

### 5.1 GBM是AdaBoost的泛化

在后续文章中，我们会详细讨论了**梯度提升机（GBM）** 。GBM的核心思想是**每一轮用基学习器去拟合损失函数的负梯度**。

AdaBoost和GBM之间有着深刻的联系 —— **AdaBoost可以被视为GBM在“指数损失函数 + 特定的坐标下降优化”下的一个特例**。更准确地说，**如果GBM选择了指数损失函数 $L(y, f(x)) = e^{-y f(x)}$，那么GBM就退化成了AdaBoost算法。**

这个联系可以从两个角度理解：

- **角度一：损失函数**。GBM允许使用**任意可微的损失函数**，而AdaBoost固定使用**指数损失函数**。从这个意义上说，GBM是AdaBoost在损失函数维度上的**泛化**——它把AdaBoost的指数损失替换成了**任意可微损失**。

- **角度二：优化算法**。AdaBoost使用**前向分步加法模型**（每一步解析地求解**最优**的 $\alpha_t$ 和 $h_t$），而GBM使用**函数空间中的梯度下降**（每一步用基学习器**拟合负梯度**）。从这个意义上说，GBM是AdaBoost在优化算法维度上的**泛化**——它把解析求解替换成了**梯度下降**。

### 5.2 指数损失的优缺点

**优点**：
- **数学性质好**，推导简洁
- 与**0-1损失**一致
- 天然导出AdaBoost的**权重更新公式**

**缺点**：
- **对异常点非常敏感**。指数损失对**大误差**的惩罚是**指数级**的——如果一个样本被严重分错（$y H(x)$ 是一个很大的负数），它的损失会**爆炸式增长**。
- 这也是为什么AdaBoost在**噪声较多的数据集**上表现可能不如GBM。

### 5.3 从AdaBoost到GBM：一条清晰的脉络

我们可以把从AdaBoost到GBM的演进看作一条清晰的脉络：

| 阶段 | 算法 | 损失函数 | 优化方法 |
|------|------|---------|---------|
| **第一阶段** | AdaBoost | **指数损失** | **前向分步**（解析求解） |
| **第二阶段** | GBM | **任意可微损失** | **函数空间梯度下降** |

**AdaBoost** 证明了“**串行纠错**”这个思路是有效的，并用**指数损失**给出了一个优雅的数学框架。

**GBM** 则把这个框架**泛化**了 —— 把“指数损失”换成“**任意可微损失**”，把“解析求解”换成“**梯度下降**”，从而把AdaBoost从一个具体的算法变成了一个**通用的算法框架**。

---

## 六、完整代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    """完整的AdaBoost实现（含训练过程记录）"""
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.weak_classifiers = []
        self.training_errors = []
        self.epsilon_history = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        D = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            # 训练弱分类器
            weak_clf = DecisionTreeClassifier(max_depth=1)
            weak_clf.fit(X, y, sample_weight=D)
            predictions = weak_clf.predict(X)
            
            # 计算加权错误率
            misclassified = (predictions != y)
            epsilon = np.sum(D * misclassified)
            
            if epsilon == 0:
                epsilon = 1e-10
            if epsilon == 1:
                epsilon = 1 - 1e-10
            
            # 计算弱分类器权重
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            
            # 更新样本权重
            D = D * np.exp(-alpha * y * predictions)
            D = D / np.sum(D)
            
            # 记录历史
            self.alphas.append(alpha)
            self.weak_classifiers.append(weak_clf)
            self.epsilon_history.append(epsilon)
            
            # 计算当前集成的训练误差
            train_pred = self.predict(X)
            self.training_errors.append(np.mean(train_pred != y))
        
        return self
    
    def predict(self, X):
        weighted_sum = np.zeros(X.shape[0])
        for alpha, clf in zip(self.alphas, self.weak_classifiers):
            weighted_sum += alpha * clf.predict(X)
        return np.sign(weighted_sum)

# ============ 生成数据并训练 ============
np.random.seed(42)
X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)
y = np.where(y == 0, -1, 1)  # 转换为 {-1, +1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练AdaBoost
adaboost = AdaBoost(n_estimators=100)
adaboost.fit(X_train, y_train)

# 评估
train_acc = accuracy_score(y_train, adaboost.predict(X_train))
test_acc = accuracy_score(y_test, adaboost.predict(X_test))

print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# ============ 可视化：训练过程 ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 弱分类器权重 α_t 的变化
axes[0].plot(adaboost.alphas, 'b-')
axes[0].set_xlabel('迭代轮次 t')
axes[0].set_ylabel('弱分类器权重 α_t')
axes[0].set_title('弱分类器权重的变化')
axes[0].grid(True)

# 2. 加权错误率 ε_t 的变化
axes[1].plot(adaboost.epsilon_history, 'r-')
axes[1].set_xlabel('迭代轮次 t')
axes[1].set_ylabel('加权错误率 ε_t')
axes[1].set_title('加权错误率的变化')
axes[1].axhline(y=0.5, color='gray', linestyle='--', label='随机猜测 (0.5)')
axes[1].legend()
axes[1].grid(True)

# 3. 训练误差的下降
axes[2].plot(adaboost.training_errors, 'g-')
axes[2].set_xlabel('迭代轮次 t')
axes[2].set_ylabel('训练集错误率')
axes[2].set_title('训练误差的下降')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ============ 可视化：决策边界 ============
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.show()

# 对比：单个弱分类器 vs AdaBoost集成
first_clf = adaboost.weak_classifiers[0]
plot_decision_boundary(X_train, y_train, first_clf, 
                       f'第1个弱分类器的决策边界 (准确率: {accuracy_score(y_train, first_clf.predict(X_train)):.3f})')
plot_decision_boundary(X_train, y_train, adaboost, 
                       f'AdaBoost集成的决策边界 (准确率: {accuracy_score(y_train, adaboost.predict(X_train)):.3f})')
```

---

> [!note] 总结
> 
> | 概念 | 核心内容 |
> |------|---------|
> | **Boosting思想** | **串行训练**，后一个模型**修正**前一个模型的**错误** |
> | **错题本机制** | **被分错的样本权重增大，被分对的样本权重减小** |
> | **加法模型** | $H(x) = \sum \alpha_t h_t(x)$，**基学习器的线性组合** |
> | **前向分步算法** | **每一步只优化当前一个基分类器**，固定之前的结果 |
> | **指数损失** | $L(y, H) = e^{-yH}$，AdaBoost的优化目标 |
> | **弱分类器权重** | $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ |
> | **样本权重更新** | $D_{t+1}(i) \propto D_t(i) \cdot \exp(-\alpha_t y_i h_t(x_i))$ |
> | **AdaBoost与GBM** | AdaBoost是GBM在**指数损失**下的特例 |
> 
> 核心要点回顾
> 
> 1. **AdaBoost的核心思想是“错题本”**：每一轮训练后，**提高被分错样本的权重，降低被分对样本的权重**，让下一个弱分类器**重点关注**上一轮的错误。
> 2. **AdaBoost是前向分步加法模型的特例**：模型是**基学习器的线性组合**（加法模型），学习策略是**每一步只优化当前一个基学习器**（前向分步），损失函数是**指数损失**。
> 3. **指数损失函数** $L(y, H) = e^{-yH}$ 是AdaBoost的数学核心。它有两个关键性质：**连续可微**便于优化，且与0-1损失**一致**（最小化指数损失等价于**最小化分类错误率**）。
> 4. **两个权重的推导**：
>    - **弱分类器权重** $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$：**错误率越低权重越大**
>    - **样本权重更新** $D_{t+1}(i) \propto D_t(i) \cdot \exp(-\alpha_t y_i h_t(x_i))$：**被分错的样本权重增大**
> 5. **AdaBoost与GBM的底层联系**：AdaBoost是GBM在**指数损失函数**下的特例。GBM将AdaBoost的“指数损失”泛化为“**任意可微损失**”，将“解析求解”泛化为“**函数空间梯度下降**”。