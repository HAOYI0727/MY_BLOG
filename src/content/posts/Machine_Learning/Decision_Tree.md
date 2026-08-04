---
title: Decision Tree —— 决策树
published: 2025-07-14
description: 系统讲解决策树家族的三大经典算法（ID3、C4.5、CART）及其分裂准则的数学原理（信息增益、信息增益率、基尼指数、MSE），深入剖析预剪枝与后剪枝（CCP成本复杂度剪枝）的策略与权衡，并解释决策树对特征尺度不敏感的深层原因。
cover: "/assets/images/posts/decision_tree.png"
coverInContent: false
tags: [决策树, 信息增益, 基尼指数, 机器学习]
category: Machine_Learning
draft: false
---

# Decision Tree —— 决策树

## 引言

回顾**第一阶段：基础监督学习**的全部六篇文章——从线性回归的解析解出发，历经逻辑回归的概率建模、KNN 的非参数距离、朴素贝叶斯的生成式假设、SVM 的几何间隔最大化，到核技巧的非线性扩展。这些算法虽然方法论各异，但共享一个底层逻辑：**它们都在一个预先定义的假设空间中寻找“最优”的数学函数**——无论这个函数是**线性超平面、概率分布还是距离度量**。

**决策树**彻底颠覆了这套范式。它放弃了所有关于“**函数形式**”的预设，转而采用一种**与人类决策方式高度契合的机制 —— 通过一系列“是/否”的判断，将特征空间递归地划分为矩形区域，每个区域对应一个预测值**。这种“**分而治之**”的策略让决策树拥有了两个无可替代的优势：**极致的可解释性**（决策路径可以被直接转化为 **if-then 规则**）和**对特征尺度的天然免疫**（分裂仅依赖**排序关系**，无需任何缩放）。

本文将从决策树家族的三位核心成员出发 —— **ID3、C4.5 与 CART** —— 系统梳理它们在分裂准则上的根本差异：**信息增益、信息增益率、基尼指数与 MSE** 各自对应的数学原理与适用场景。随后我们深入**剪枝策略**，对比**预剪枝**的“防患于未然”与**后剪枝**（特别是 CART 的 **CCP 成本复杂度剪枝**）的“亡羊补牢”，揭示二者在**欠拟合与过拟合**之间的权衡。最后，我们将从数学上解释决策树为何**对特征缩放完全不敏感** —— 这一性质使其在处理异构特征时远比 KNN 或 SVM 从容。

> [!note]
> 
> 决策树是**第二阶段：树模型与集成学习**的基石。理解**单棵树的生长与剪枝**，是后续理解 **Bagging**（通过抽样降低方差）、**随机森林**（在 Bagging 基础上引入特征随机选择）以及 **AdaBoost、GBM、XGBoost**（通过串行拟合残差降低偏差）的**必要前提**。
> 
> 下一篇文章，我们将从“一棵树”走向“一片森林”——**Bagging 与随机森林**将展示如何通过**集成学习**让决策树从“**易过拟合的弱学习器**”蜕变为“**强泛化的组合模型**”。

---

## 一、ID3、C4.5与CART

**决策树**家族中有三大基础算法 —— **ID3、C4.5和CART**。它们之间的核心区别可以概括为一句话：**特征选择的标准不同**。

| 算法 | 提出年份 | 树类型 | 分裂准则 | 适用任务 |
|------|---------|--------|---------|---------|
| **ID3** | 1986 | 多叉树 | **信息增益** | 分类 |
| **C4.5** | 1993 | 多叉树 | **信息增益率** | 分类 |
| **CART** | 1984 | **二叉树** | **基尼指数（分类）/ MSE（回归）** | 分类 + **回归** |

### 1.1 ID3：信息增益的开创者

ID3（Iterative Dichotomiser 3）由Ross Quinlan于1986年提出，是决策树算法的开山之作。它的核心思想是**以信息增益作为特征选择的准则，选择信息增益最大的特征进行分裂**。

**ID3的局限性**：
- 只能处理**离散特征**，无法处理**连续值**，也不能处理**缺失值**
- **没有剪枝策略**，容易**过拟合**
- **偏好取值多的特征** —— 比如“编号”这样的特征，信息增益会接近1，但毫无意义

### 1.2 C4.5：ID3的全面升级

C4.5是Quinlan对ID3的改进版本，主要解决了ID3的三大痛点：

1. 用**信息增益率**代替信息增益，克服了对取值多特征的偏好
2. 引入了**连续特征离散化**：将连续特征**排序**后，取**相邻两样本值的平均数**作为候选切分点
3. 引入**悲观剪枝策略**进行**后剪枝，能够处理缺失值**

**C4.5的局限性**：需要**多次扫描和排序**数据，效率较低；只能处理**分类**任务，不能做回归。

### 1.3 CART：二叉树的全能选手

CART（Classification And Regression Tree）由Breiman等人在1984年提出，是决策树家族中最强大的成员。

**CART 与 ID3 / C4.5 的关键区别**：

- **二叉树 vs 多叉树**：CART每次只将数据分成**两份**，生成的是**二叉树**，而ID3和C4.5是多叉树
- **分类 + 回归**：CART既可以做**分类**（用**基尼指数**），也可以做**回归**（用**MSE**）
- **基尼指数代替熵**：基尼指数只涉及**平方**运算，避免了熵模型中的大量**对数**运算，**计算效率更高**

**代码实现**：

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

决策树的核心问题只有一个：**每次分裂时，应该选择哪个特征？在哪个阈值上分裂？** 不同的算法给出了不同的答案。下面我们逐一推导。

### 2.1 信息熵与信息增益（ID3）

**信息熵（Entropy）** —— 信息论中衡量**不确定性**的指标。**不确定性越大，熵越大**。

对于**样本集合** $D$，假设共有 $K$ 个**类别**，第 $k$ 类样本所占**比例**为 $p_k$，则**信息熵**定义为：

$$
H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

**熵的性质**：当所有样本都属于**同一类别**时，$H(D) = 0$（纯度**最高**）；当各类别样本**均匀分布**时，$H(D)$ 最大（纯度**最低**）

**条件熵** $H(D|A)$ —— 表示**在已知特征 $A$ 的条件下，数据集 $D$ 的不确定性**：

$$
H(D|A) = \sum_{v=1}^{V} \frac{|D^v|}{|D|} H(D^v)
$$

其中 $V$ 是特征 $A$ 的**取值个数**，$D^v$ 是特征 $A$ 取第 $v$ 个值的**样本子集**。

**信息增益** —— **分裂前后熵的减少量**：

$$
\boxed{\text{Gain}(D, A) = H(D) - H(D|A)}
$$

**ID3**选择**信息增益最大**的特征进行**分裂**。

**代码实现**：

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

信息增益有一个致命缺陷：**偏好取值数量多的特征**。比如“ID”这样的特征，每个样本取值都不同，条件熵为0，信息增益最大，但这样的**分裂毫无意义**。

**C4.5**引入了**信息增益率（Gain Ratio）** 来修正这一问题：

$$
\boxed{\text{GainRatio}(D, A) = \frac{\text{Gain}(D, A)}{\text{IV}(A)}}
$$

其中 $\text{IV}(A)$ 称为**分裂信息（Intrinsic Value）** ，衡量**特征 $A$ 自身的信息量**：

$$
\text{IV}(A) = -\sum_{v=1}^{V} \frac{|D^v|}{|D|} \log_2 \frac{|D^v|}{|D|}
$$

直观理解：**特征取值越多，$\text{IV}(A)$ 越大，信息增益率就被惩罚得越狠。**

**C4.5**选择**信息增益率最大**的特征进行分裂。

### 2.3 基尼指数（CART分类）

**CART分类树**使用**基尼指数（Gini Index）** 作为分裂准则。

**基尼指数**衡量的是**从数据集中随机抽取两个样本，其类别不一致的概率**。基尼指数**越小，纯度越高**。

对于样本集合 $D$，基尼指数定义为：

$$
\boxed{\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2}
$$

对于**二分类问题**，如果正类概率为 $p$，则：

$$
\text{Gini}(D) = 1 - p^2 - (1-p)^2 = 2p(1-p)
$$

对于特征 $A$ 的某个划分（**CART是二叉树，每次只二分**）：

$$
\text{Gini}(D, A) = \frac{|D_1|}{|D|}\text{Gini}(D_1) + \frac{|D_2|}{|D|}\text{Gini}(D_2)
$$

**CART分类树**选择**基尼指数最小**的特征和阈值进行分裂。

> [!note] CART用基尼指数代替熵的原因
> 
> 基尼指数与熵在数学上非常接近。
> 
> 对于**二分类问题**，基尼指数 $2p(1-p)$ 与熵之半 $-p\log_2 p - (1-p)\log_2(1-p)$ 的曲线**几乎重合**。
> 
> 但基尼指数**只涉及平方运算**，避免了大量的对数运算，**计算效率更高**。

**代码实现**：

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

当目标变量是**连续值**时，**CART回归树**使用**均方误差（Mean Squared Error, MSE）** 作为分裂准则。

对于节点 $m$，其预测值 $\hat{y}_m$ 为该节点**所有样本目标值的均值**，MSE为：

$$
\text{MSE}(D) = \frac{1}{|D|}\sum_{i \in D} (y_i - \hat{y}_D)^2
$$

其中 $\hat{y}_D = \frac{1}{|D|}\sum_{i \in D} y_i$。

对于某个分裂，分裂后的MSE为**左右子节点MSE的加权和**：

$$
\text{MSE}_{\text{split}} = \frac{|D_1|}{|D|}\text{MSE}(D_1) + \frac{|D_2|}{|D|}\text{MSE}(D_2)
$$

**CART回归树**选择**使分裂后MSE最小**的特征和阈值。

---

## 三、剪枝策略

决策树有一个著名的“缺点”：**如果不加限制，它可以生长到完美拟合每一个训练样本** —— 直到每个叶子节点都只包含一个样本。这样的树在训练集上准确率100%，但在测试集上表现极差 —— 这就是**过拟合**。

**剪枝（Pruning）** 就是为了解决这个问题而生的。剪枝策略分为两大类：**预剪枝**和**后剪枝**。

### 3.1 预剪枝（Pre-Pruning）

预剪枝是在**决策树生成过程中**就提前停止树的生长。

常见的**预剪枝条件**：

| 参数 | 含义 |
|------|------|
| `max_depth` | 树的**最大深度** |
| `min_samples_split` | **节点分裂**所需的最小样本数 |
| `min_samples_leaf` | **叶子节点**所需的最小样本数 |
| `max_leaf_nodes` | **最大叶子节点数** |
| `min_impurity_decrease` | 分裂所需的**最小不纯度下降** |

- **优点**：计算效率高，简单直接。
- **缺点**：基于“**贪心**”本质，过早停止可能导致**欠拟合**。比如某个节点当前分裂看起来“不划算”，但再往下分裂一层后可能会有更好的效果 —— **预剪枝无法看到这种“长远收益”**。

**代码实现**：

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

### 3.2 后剪枝（Post-Pruning）

后剪枝是**先让树充分生长**，然后**再从底部向上修剪**。

> [!note] 常见的后剪枝方法
> 
> - **成本复杂度剪枝（Cost-Complexity Pruning, CCP） —— CART**采用的方法
> - **悲观剪枝（Pessimistic Error Pruning, PEP） —— C4.5**采用的方法
> - **错误率降低剪枝（Reduced Error Pruning, REP）**


**成本复杂度剪枝（Cost-Complexity Pruning, CCP）**

CCP是**CART算法**采用的剪枝方法，其核心思想是定义一个**损失函数**，在**预测误差**和**树复杂度**之间做权衡。

对于一棵树 $T$，定义其**成本复杂度**为：

$$
\boxed{C_\alpha(T) = R(T) + \alpha \cdot |T|}
$$

其中：
- $R(T)$：树的**误差**（如**分类错误率或MSE**）
- $|T|$：树的**叶子节点数量**（衡量**复杂度**）
- $\alpha \geq 0$：**惩罚参数**，控制复杂度在损失函数中的**权重**
  - $\alpha = 0$：**只关心误差**，不关心复杂度 → 树**最大**
  - $\alpha \to \infty$：**只关心复杂度** → 树退化为**根节点**

**CCP的剪枝过程**：
1. 从**完整树** $T_0$ 开始
2. 计算每个**内部节点**被**剪枝**后的**损失函数**变化
3. **每次剪掉使损失函数增加最小的节点**
4. 得到一系列**嵌套的子树** $T_0 \supset T_1 \supset T_2 \supset ... \supset \{根节点\}$
5. 用**交叉验证**选择最优的 $\alpha$ 值，从而选择**最优子树**

**代码实现**：

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

### 3.3 预剪枝 vs 后剪枝

| 对比维度 | 预剪枝 | 后剪枝 |
|---------|--------|--------|
| **时机** | 树生长**过程**中 | 树生长**完成**后 |
| **计算开销** | **小** | **大**（需要先生成完整树） |
| **风险** | **欠拟合** | **过拟合**（如果剪枝不充分） |
| **效果** | 通常**较差** | 通常**更好** |
| **常用程度** | 常用（因为简单高效） | 更常用（因为效果更好） |

**实际建议**：先用**预剪枝参数**（如 `max_depth`、`min_samples_split`）快速得到一个“**差不多**”的模型，如果效果不理想，再考虑**使用 `ccp_alpha` 做后剪枝**。

---

## 四、决策树对特征尺度不敏感的原因

对于KNN或逻辑回归，**特征标准化（Standardization）** 或**归一化（Normalization）** 是必不可少的预处理步骤。但对于决策树，**完全不需要做特征缩放**。

### 4.1 核心原因：决策树基于“排序”而非“距离”

决策树的**分裂机制**决定了它**对特征尺度不敏感 —— 决策树寻找最佳分裂点时，仅依赖特征值的排序关系，而非原始数值大小**。

1. **特征选择**：**信息增益、信息增益率、基尼指数**都基于**概率分布**计算，与特征的**具体数值**无关。无论特征值是从0到1还是从0到1000，只要**排序关系**不变，这些指标的计算结果就不变。

2. **分裂点选择**：对于连续特征，决策树将特征值**排序**后，在**相邻值**之间尝试切分。**归一化只是把所有值按比例缩放，排序关系完全不变**，因此最佳分裂点的位置（在排序中的位置）也完全不变。

### 4.2 数学证明

假设连续特征 $A$ 的取值范围为 $[a_{\min}, a_{\max}]$，对其进行**归一化**：

$$
A_{\text{norm}} = \frac{A - a_{\min}}{a_{\max} - a_{\min}}
$$

对于任意**候选分裂点** $t \in [a_{\min}, a_{\max}]$，**归一化后**的对应分裂点为：

$$
t_{\text{norm}} = \frac{t - a_{\min}}{a_{\max} - a_{\min}}
$$

由于归一化是**严格单调递增**的线性变换，**排序关系完全不变**。因此，原始分裂点 $t$ 与归一化分裂点 $t_{\text{norm}}$ **在分裂效果上完全等价**。

更一般地，**任何严格单调变换**（如取对数、平方根等）都不会改变决策树的分裂决策。

### 4.3 树模型“免疫”特征缩放的原因

**决策树由输入特征的阶跃函数组成** —— 每个分裂点就像在特征轴上“切一刀”，左边的归左子树，右边的归右子树。这个“切”的位置**只取决于相对顺序，而不取决于绝对数值**。

这与其他基于距离的算法形成鲜明对比：

| 模型类型 | 对特征缩放是否敏感 | 原因 |
|---------|-----------------|------|
| KNN、SVM、神经网络 | **敏感** | 依赖**距离计算**，**量纲大**的特征主导 |
| 决策树、随机森林、GBDT | **不敏感** | 依赖**排序关系**，不依赖距离 |

**代码实现**：

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

## 五、完整代码实现

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

> [!note] 总结
> 
> | 概念 | 核心内容 |
> |------|---------|
> | **ID3** | **信息增益**，多叉树，只能分类，无剪枝 |
> | **C4.5** | **信息增益率**，多叉树，只能分类，**有剪枝** |
> | **CART** | **基尼指数（分类）/ MSE（回归），二叉树，分类+回归** |
> | **信息增益** | $H(D) - H(D\|A)$，偏好**取值多**的特征 |
> | **信息增益率** | 信息增益 / 分裂信息，修正了对取值多特征的偏好 |
> | **基尼指数** | $1 - \sum p_k^2$，越小越纯，**CART分类用** |
> | **MSE** | $\frac{1}{n}\sum(y_i - \hat{y})^2$，**CART回归用** |
> | **预剪枝** | 生长**过程中**提前停止，简单但可能**欠拟合** |
> | **后剪枝** | 生长**完成后**修剪，效果好但**计算量大** |
> | **CCP** | $C_\alpha(T) = R(T) + \alpha\|T\|$，CART的后剪枝方法 |
> | **特征尺度** | 决策树**对特征缩放不敏感**（依赖**排序**而非距离） |
> 
> 核心要点回顾
> 1. **ID3、C4.5、CART** 的核心区别在于**分裂准则不同**。ID3用**信息增益**，C4.5用**信息增益率**，CART**分类用基尼指数、回归用MSE**。
> 2. **信息增益**偏好取值多的特征，**信息增益率**通过除以分裂信息来修正这一偏差，**基尼指数**与熵在数学上近似但计算更高效。
> 3. **预剪枝**通过 `max_depth`、`min_samples_split` 等参数在树生长过程中限制复杂度；**后剪枝**（如CART的**CCP**）先生成完整树再修剪，**通常效果更好**。
> 4. **决策树对特征尺度不敏感**的根本原因在于：分裂只依赖特征值的**排序关系**，而非**数值大小**。任何单调变换都不会改变分裂决策。
> 5. **实际使用建议**：先用**预剪枝参数**快速得到一个**基准模型**，再用 `ccp_alpha` 做精细的**后剪枝调优**。