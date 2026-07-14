---
title: Decision Tree —— 决策树
published: 2025-07-14
description: 系统讲解决策树家族的三大经典算法（ID3、C4.5、CART）及其分裂准则的数学原理（信息增益、信息增益率、基尼指数、MSE），深入剖析预剪枝与后剪枝（CCP成本复杂度剪枝）的策略与权衡，并解释决策树对特征尺度不敏感的深层原因。
cover: "/assets/images/posts/decision_tree.png"
coverInContent: false
tags: [决策树, 信息增益, 基尼指数]
category: Machine_Learning
draft: false
---

# Decision Tree —— 决策树

## 前言

在前面几篇文章中，我们讨论了线性模型、支持向量机、KNN和朴素贝叶斯。这些算法各有千秋，但有一个共同的“缺点”——它们或多或少都像一个**黑箱**：你输入数据，它输出结果，但中间发生了什么，很难向非技术人员解释清楚。

**决策树（Decision Tree）** 则完全不同。它的决策过程与人类思维高度相似——**通过一系列“是/否”的问题，逐步缩小范围，最终得出结论**。正因如此，决策树被誉为机器学习中**可解释性最强**的算法之一。

决策树的历史可以追溯到上世纪60年代的概念学习系统（CLS），经过Quinlan在1986年提出的ID3、1993年的C4.5，再到Breiman等人在1984年提出的CART，决策树家族已经发展出了一套完整的理论体系。

这篇文章，我们将从ID3、C4.5、CART三种经典算法的区别出发，深入推导信息增益、基尼指数和MSE的数学公式，系统讲解预剪枝与后剪枝的策略，最后解释为什么决策树对特征尺度不敏感。

---

## 一、ID3、C4.5与CART：三棵树的恩怨情仇

决策树家族中有三位“元老”——**ID3、C4.5和CART**。它们之间的核心区别可以概括为一句话：**特征选择的标准不同**。

| 算法 | 提出年份 | 树类型 | 分裂准则 | 适用任务 |
|------|---------|--------|---------|---------|
| **ID3** | 1986 | 多叉树 | 信息增益 | 分类 |
| **C4.5** | 1993 | 多叉树 | 信息增益率 | 分类 |
| **CART** | 1984 | 二叉树 | 基尼指数（分类）/ MSE（回归） | 分类 + 回归 |

### 1.1 ID3：信息增益的开创者

ID3（Iterative Dichotomiser 3）由Ross Quinlan于1986年提出，是决策树算法的开山之作。它的核心思想是**以信息增益作为特征选择的准则，选择信息增益最大的特征进行分裂**。

**ID3的局限性**：
- **只能处理离散特征**，无法处理连续值
- **不能处理缺失值**
- **没有剪枝策略**，容易过拟合
- **偏好取值多的特征**——比如“编号”这样的特征，信息增益会接近1，但毫无意义

### 1.2 C4.5：ID3的全面升级

C4.5是Quinlan对ID3的改进版本，主要解决了ID3的三大痛点：

1. **用信息增益率代替信息增益**，克服了对取值多特征的偏好
2. **引入了连续特征离散化**：将连续特征排序后，取相邻两样本值的平均数作为候选切分点
3. **引入悲观剪枝策略**进行后剪枝
4. **能够处理缺失值**

**C4.5的缺点**：需要多次扫描和排序数据，效率较低；只能处理分类任务，不能做回归。

### 1.3 CART：二叉树的全能选手

CART（Classification And Regression Tree）由Breiman等人在1984年提出，是决策树家族中最强大的成员。

**CART与ID3/C4.5的关键区别**：

- **二叉树 vs 多叉树**：CART每次只将数据分成**两份**，生成的是二叉树，而ID3和C4.5是多叉树
- **分类 + 回归**：CART既可以做分类（用基尼指数），也可以做回归（用MSE）
- **基尼指数代替熵**：基尼指数只涉及平方运算，避免了熵模型中的大量对数运算，计算效率更高

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression

# CART分类树（默认使用基尼指数）
X_clf, y_clf = make_classification(n_samples=300, n_features=4, random_state=42)
clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
clf.fit(X_clf, y_clf)

# CART回归树（使用MSE）
X_reg, y_reg = make_regression(n_samples=300, n_features=4, noise=10, random_state=42)
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
reg.fit(X_reg, y_reg)
```

---

## 二、分裂准则的数学原理

决策树的核心问题只有一个：**每次分裂时，应该选择哪个特征？在哪个阈值上分裂？** 

不同的算法给出了不同的答案。下面我们逐一推导。

### 2.1 信息熵与信息增益（ID3）

**信息熵（Entropy）** 是信息论中衡量**不确定性**的指标。不确定性越大，熵越大。

对于样本集合 $D$，假设共有 $K$ 个类别，第 $k$ 类样本所占比例为 $p_k$，则信息熵定义为：

$$
H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

**熵的性质**：
- 当所有样本都属于同一类别时，$H(D) = 0$（纯度最高）
- 当各类别样本均匀分布时，$H(D)$ 最大（纯度最低）

**条件熵** $H(D|A)$ 表示在已知特征 $A$ 的条件下，数据集 $D$ 的不确定性：

$$
H(D|A) = \sum_{v=1}^{V} \frac{|D^v|}{|D|} H(D^v)
$$

其中 $V$ 是特征 $A$ 的取值个数，$D^v$ 是特征 $A$ 取第 $v$ 个值的样本子集。

**信息增益** 就是分裂前后熵的减少量：

$$
\boxed{\text{Gain}(D, A) = H(D) - H(D|A)}
$$

ID3选择**信息增益最大**的特征进行分裂。

```python
import numpy as np

def entropy(y):
    """计算信息熵"""
    classes = np.unique(y)
    probs = [np.sum(y == c) / len(y) for c in classes]
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(X, y, feature_idx):
    """计算某个特征的信息增益"""
    total_entropy = entropy(y)
    values = np.unique(X[:, feature_idx])
    weighted_entropy = 0
    for v in values:
        mask = X[:, feature_idx] == v
        subset_y = y[mask]
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)
    return total_entropy - weighted_entropy

# 示例：计算信息增益
X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
y = np.array([0, 0, 1, 1])
print(f"特征0的信息增益: {information_gain(X, y, 0):.4f}")
print(f"特征1的信息增益: {information_gain(X, y, 1):.4f}")
```

### 2.2 信息增益率（C4.5）

信息增益有一个致命缺陷：**偏好取值数量多的特征**。比如“ID”这样的特征，每个样本取值都不同，条件熵为0，信息增益最大，但这样的分裂毫无意义。

C4.5引入了**信息增益率（Gain Ratio）** 来修正这一问题：

$$
\boxed{\text{GainRatio}(D, A) = \frac{\text{Gain}(D, A)}{\text{IV}(A)}}
$$

其中 $\text{IV}(A)$ 称为**分裂信息（Intrinsic Value）** ，衡量特征 $A$ 自身的信息量：

$$
\text{IV}(A) = -\sum_{v=1}^{V} \frac{|D^v|}{|D|} \log_2 \frac{|D^v|}{|D|}
$$

**直观理解**：特征取值越多，$\text{IV}(A)$ 越大，信息增益率就被“惩罚”得越狠。

C4.5选择**信息增益率最大**的特征进行分裂。

### 2.3 基尼指数（CART分类）

CART分类树使用**基尼指数（Gini Index）** 作为分裂准则。

基尼指数衡量的是**从数据集中随机抽取两个样本，其类别不一致的概率**。基尼指数**越小，纯度越高**。

对于样本集合 $D$，基尼指数定义为：

$$
\boxed{\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2}
$$

对于**二分类问题**，如果正类概率为 $p$，则：

$$
\text{Gini}(D) = 1 - p^2 - (1-p)^2 = 2p(1-p)
$$

对于特征 $A$ 的某个划分（CART是二叉树，每次只二分）：

$$
\text{Gini}(D, A) = \frac{|D_1|}{|D|}\text{Gini}(D_1) + \frac{|D_2|}{|D|}\text{Gini}(D_2)
$$

CART选择**基尼指数最小**的特征和阈值进行分裂。

**为什么CART用基尼指数代替熵？**

基尼指数与熵在数学上非常接近。对于二分类问题，基尼指数 $2p(1-p)$ 与熵之半 $-p\log_2 p - (1-p)\log_2(1-p)$ 的曲线几乎重合。但基尼指数**只涉及平方运算**，避免了大量的对数运算，计算效率更高。

```python
def gini(y):
    """计算基尼指数"""
    classes = np.unique(y)
    probs = [np.sum(y == c) / len(y) for c in classes]
    return 1 - np.sum([p**2 for p in probs])

def gini_split(X, y, feature_idx, threshold):
    """计算在某个阈值上二分的基尼指数"""
    mask = X[:, feature_idx] <= threshold
    left_y, right_y = y[mask], y[~mask]
    left_weight = len(left_y) / len(y)
    right_weight = len(right_y) / len(y)
    return left_weight * gini(left_y) + right_weight * gini(right_y)
```

### 2.4 均方误差（CART回归）

当目标变量是**连续值**时，CART回归树使用**均方误差（Mean Squared Error, MSE）** 作为分裂准则。

对于节点 $m$，其预测值 $\hat{y}_m$ 为该节点所有样本目标值的均值，MSE为：

$$
\text{MSE}(D) = \frac{1}{|D|}\sum_{i \in D} (y_i - \hat{y}_D)^2
$$

其中 $\hat{y}_D = \frac{1}{|D|}\sum_{i \in D} y_i$。

对于某个分裂，分裂后的MSE为左右子节点MSE的加权和：

$$
\text{MSE}_{\text{split}} = \frac{|D_1|}{|D|}\text{MSE}(D_1) + \frac{|D_2|}{|D|}\text{MSE}(D_2)
$$

CART回归树选择**使分裂后MSE最小**的特征和阈值。

---

## 三、剪枝策略：防止过拟合的手术刀

决策树有一个著名的“缺点”：**如果不加限制，它可以生长到完美拟合每一个训练样本**——直到每个叶子节点都只包含一个样本。这样的树在训练集上准确率100%，但在测试集上表现极差——这就是**过拟合**。

**剪枝（Pruning）** 就是为了解决这个问题而生的。剪枝策略分为两大类：**预剪枝**和**后剪枝**。

### 3.1 预剪枝（Pre-Pruning）：防患于未然

预剪枝是在**决策树生成过程中**就提前停止树的生长。

**常见的预剪枝条件**：

| 参数 | 含义 |
|------|------|
| `max_depth` | 树的最大深度 |
| `min_samples_split` | 节点分裂所需的最小样本数 |
| `min_samples_leaf` | 叶子节点所需的最小样本数 |
| `max_leaf_nodes` | 最大叶子节点数 |
| `min_impurity_decrease` | 分裂所需的最小不纯度下降 |

**预剪枝的优点**：计算效率高，简单直接。

**预剪枝的缺点**：基于“贪心”本质，过早停止可能导致**欠拟合**。比如某个节点当前分裂看起来“不划算”，但再往下分裂一层后可能会有更好的效果——预剪枝无法看到这种“长远收益”。

```python
from sklearn.tree import DecisionTreeClassifier

# 预剪枝：通过参数限制树的生长
clf = DecisionTreeClassifier(
    max_depth=5,              # 最大深度
    min_samples_split=10,     # 最少分裂样本数
    min_samples_leaf=5,       # 最少叶子样本数
    max_leaf_nodes=20,        # 最大叶子节点数
    random_state=42
)
clf.fit(X_train, y_train)
```

### 3.2 后剪枝（Post-Pruning）：亡羊补牢

后剪枝是**先让树充分生长**，然后再从底部向上修剪。

常见的后剪枝方法包括：
- **成本复杂度剪枝（Cost-Complexity Pruning, CCP）** ：CART采用的方法
- **错误率降低剪枝（Reduced Error Pruning, REP）**
- **悲观剪枝（Pessimistic Error Pruning, PEP）** ：C4.5采用的方法

#### 3.2.1 成本复杂度剪枝（CCP）

CCP是CART算法采用的剪枝方法，其核心思想是定义一个**损失函数**，在**预测误差**和**树复杂度**之间做权衡。

对于一棵树 $T$，定义其**成本复杂度**为：

$$
\boxed{C_\alpha(T) = R(T) + \alpha \cdot |T|}
$$

其中：
- $R(T)$：树的**误差**（如分类错误率或MSE）
- $|T|$：树的**叶子节点数量**（衡量复杂度）
- $\alpha \geq 0$：**惩罚参数**，控制复杂度在损失函数中的权重

**$\alpha$ 的作用**：
- $\alpha = 0$：只关心误差，不关心复杂度 → 树最大
- $\alpha \to \infty$：只关心复杂度 → 树退化为根节点

CCP的剪枝过程是：

1. 从完整树 $T_0$ 开始
2. 计算每个内部节点被剪枝后的损失函数变化
3. 每次剪掉**使损失函数增加最小**的节点
4. 得到一系列嵌套的子树 $T_0 \supset T_1 \supset T_2 \supset ... \supset \{根节点\}$
5. 用**交叉验证**选择最优的 $\alpha$ 值，从而选择最优子树

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. 先让树充分生长（不设限制或设很松的限制）
clf = DecisionTreeClassifier(
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
clf.fit(X_train, y_train)

# 2. 使用ccp_alpha进行后剪枝
# 可以尝试不同的ccp_alpha值，用验证集选择最优的
for alpha in [0.0, 0.001, 0.005, 0.01, 0.05]:
    clf_pruned = DecisionTreeClassifier(
        ccp_alpha=alpha,
        random_state=42
    )
    clf_pruned.fit(X_train, y_train)
    acc = clf_pruned.score(X_val, y_val)
    print(f"ccp_alpha={alpha:.3f}, 验证集准确率={acc:.4f}, 叶子数={clf_pruned.tree_.n_leaves}")
```

### 3.3 预剪枝 vs 后剪枝：如何选择？

| 对比维度 | 预剪枝 | 后剪枝 |
|---------|--------|--------|
| **时机** | 树生长过程中 | 树生长完成后 |
| **计算开销** | 小 | 大（需要先生成完整树） |
| **风险** | 欠拟合 | 过拟合（如果剪枝不充分） |
| **效果** | 通常较差 | 通常更好 |
| **常用程度** | 常用（因为简单高效） | 更常用（因为效果更好） |

**实际建议**：先用预剪枝参数（如 `max_depth`、`min_samples_split`）快速得到一个“差不多”的模型，如果效果不理想，再考虑使用 `ccp_alpha` 做后剪枝。

---

## 四、决策树对特征尺度不敏感的原因

如果你用过KNN或逻辑回归，一定知道**特征标准化（Standardization）** 或**归一化（Normalization）** 是必不可少的预处理步骤。但如果你用决策树，**完全不需要做特征缩放**。为什么？

### 4.1 核心原因：决策树基于“排序”而非“距离”

决策树的分裂机制决定了它对特征尺度不敏感：

> **决策树寻找最佳分裂点时，仅依赖特征值的排序关系，而非原始数值大小**。

具体来说：

1. **特征选择**：信息增益、信息增益率、基尼指数都基于**概率分布**计算，与特征的具体数值无关。无论特征值是从0到1还是从0到1000，只要排序关系不变，这些指标的计算结果就不变。

2. **分裂点选择**：对于连续特征，决策树将特征值排序后，在相邻值之间尝试切分。归一化只是把所有值按比例缩放，**排序关系完全不变**，因此最佳分裂点的位置（在排序中的位置）也完全不变。

### 4.2 数学证明

假设连续特征 $A$ 的取值范围为 $[a_{\min}, a_{\max}]$，对其进行归一化：

$$
A_{\text{norm}} = \frac{A - a_{\min}}{a_{\max} - a_{\min}}
$$

对于任意候选分裂点 $t \in [a_{\min}, a_{\max}]$，归一化后的对应分裂点为：

$$
t_{\text{norm}} = \frac{t - a_{\min}}{a_{\max} - a_{\min}}
$$

由于归一化是**严格单调递增**的线性变换，**排序关系完全不变**。因此，原始分裂点 $t$ 与归一化分裂点 $t_{\text{norm}}$ 在分裂效果上**完全等价**。

更一般地，**任何严格单调变换**（如取对数、平方根等）都不会改变决策树的分裂决策。

### 4.3 为什么树模型“免疫”特征缩放？

一句话总结：**决策树由输入特征的阶跃函数组成**——每个分裂点就像在特征轴上“切一刀”，左边的归左子树，右边的归右子树。这个“切”的位置只取决于**相对顺序**，而不取决于**绝对数值**。

这与其他基于距离的算法形成鲜明对比：

| 模型类型 | 对特征缩放敏感吗？ | 原因 |
|---------|-----------------|------|
| KNN、SVM、神经网络 | **敏感** | 依赖距离计算，量纲大的特征主导 |
| 决策树、随机森林、GBDT | **不敏感** | 依赖排序关系，不依赖距离 |

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# 验证：特征缩放不影响决策树的预测结果
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 不缩放
clf_raw = DecisionTreeClassifier(random_state=42)
clf_raw.fit(X_train, y_train)
pred_raw = clf_raw.predict(X_test)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf_scaled = DecisionTreeClassifier(random_state=42)
clf_scaled.fit(X_train_scaled, y_train)
pred_scaled = clf_scaled.predict(X_test_scaled)

# 两次预测应该完全一致
print(f"预测结果是否相同: {np.all(pred_raw == pred_scaled)}")  # True
```

---

## 五、完整示例：从数据到决策树

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 训练决策树（带预剪枝）
clf = DecisionTreeClassifier(
    criterion='gini',      # CART分类树使用基尼指数
    max_depth=4,           # 预剪枝：限制深度
    min_samples_split=10,  # 预剪枝：限制分裂最小样本数
    random_state=42
)
clf.fit(X_train, y_train)

# 3. 评估
y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"树的深度: {clf.get_depth()}")
print(f"叶子节点数: {clf.get_n_leaves()}")

# 4. 可视化决策树
plt.figure(figsize=(16, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("决策树可视化（鸢尾花分类）")
plt.show()

# 5. 后剪枝：使用ccp_alpha
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 对每个alpha训练一棵树
clfs = []
for alpha in ccp_alphas:
    clf_alpha = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf_alpha.fit(X_train, y_train)
    clfs.append(clf_alpha)

# 在验证集上选择最优alpha
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# 可视化alpha的影响
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, 'b-o', label='训练集准确率')
plt.plot(ccp_alphas, test_scores, 'r-o', label='测试集准确率')
plt.xlabel('ccp_alpha')
plt.ylabel('准确率')
plt.legend()
plt.title('ccp_alpha 对模型性能的影响')
plt.show()
```

---

## 六、总结

| 概念 | 核心内容 |
|------|---------|
| **ID3** | 信息增益，多叉树，只能分类，无剪枝 |
| **C4.5** | 信息增益率，多叉树，只能分类，有剪枝 |
| **CART** | 基尼指数（分类）/ MSE（回归），二叉树，分类+回归 |
| **信息增益** | $H(D) - H(D\|A)$，偏好取值多的特征 |
| **信息增益率** | 信息增益 / 分裂信息，修正了对取值多特征的偏好 |
| **基尼指数** | $1 - \sum p_k^2$，越小越纯，CART分类用 |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y})^2$，CART回归用 |
| **预剪枝** | 生长过程中提前停止，简单但可能欠拟合 |
| **后剪枝** | 生长完成后修剪，效果好但计算量大 |
| **CCP** | $C_\alpha(T) = R(T) + \alpha\|T\|$，CART的后剪枝方法 |
| **特征尺度** | 决策树对特征缩放不敏感（依赖排序而非距离） |

### 核心要点回顾

1. **ID3、C4.5、CART** 的核心区别在于**分裂准则不同**。ID3用信息增益，C4.5用信息增益率，CART分类用基尼指数、回归用MSE。

2. **信息增益**偏好取值多的特征，**信息增益率**通过除以分裂信息来修正这一偏差，**基尼指数**与熵在数学上近似但计算更高效。

3. **预剪枝**通过 `max_depth`、`min_samples_split` 等参数在树生长过程中限制复杂度；**后剪枝**（如CART的CCP）先生成完整树再修剪，通常效果更好。

4. **决策树对特征尺度不敏感**的根本原因在于：分裂只依赖特征值的**排序关系**，而非**数值大小**。任何单调变换都不会改变分裂决策。

5. **实际使用建议**：先用预剪枝参数快速得到一个基准模型，再用 `ccp_alpha` 做精细的后剪枝调优。