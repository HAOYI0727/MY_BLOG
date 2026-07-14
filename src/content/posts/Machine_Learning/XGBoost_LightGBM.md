---
title: XGBoost & LightGBM —— 梯度提升框架
published: 2025-07-20
description: 系统解析XGBoost与LightGBM两大梯度提升框架的核心工程优化技术。涵盖XGBoost的二阶泰勒展开与正则化项的数学推导、预排序与Block并行架构；LightGBM的直方图算法、GOSS梯度采样、EFB互斥特征捆绑三大加速技术，以及Level-wise与Leaf-wise树生长策略的对比。
cover: "/assets/images/posts/xgboost_lightgbm.png"
coverInContent: false
tags: [XGBoost, LightGBM, 机器学习]
category: Machine_Learning
draft: false
---

# XGBoost & LightGBM —— 梯度提升框架

## 前言

在上一篇文章中，我们从加法模型和前向分步算法出发，完整推导了梯度提升机（GBM）的理论框架。经典的GBM用一阶梯度（即负梯度）来拟合残差，虽然通用且有效，但在工程实践中面临着两个核心痛点：**训练速度慢**和**内存消耗大**。

2014年，陈天奇发布了**XGBoost（Extreme Gradient Boosting）** ，首次将**二阶泰勒展开**和**正则化项**系统性地引入梯度提升框架。三年后，微软团队推出了**LightGBM**，通过**直方图算法、GOSS采样、EFB特征捆绑**等一系列工程创新，将梯度提升的训练效率推向了一个新的高度。

XGBoost被誉为“**工程优化的奠基者**”，而LightGBM则被称为“**高效处理大数据的革新者**”。这篇文章，我们将从数学原理到工程实现，深入剖析这两个框架的核心优化技术。

---

## 一、XGBoost：二阶泰勒展开与正则化

### 1.1 从一阶到二阶：牛顿提升

在经典的GBM中，每一轮迭代通过拟合损失函数的**一阶梯度（负梯度）** 来更新模型。这本质上是在**函数空间**中做**梯度下降**——只利用了目标函数的一阶导数信息。

XGBoost的核心创新在于：**将损失函数做二阶泰勒展开，使用一阶导数和二阶导数共同决定下一步的方向**。这相当于在函数空间中做**牛顿法（Newton's Method）** 而非梯度下降法。

**数学推导**：

假设前 $t-1$ 轮已经得到模型 $\hat{y}_i^{(t-1)}$，第 $t$ 轮要学习的新树为 $f_t(x_i)$，则新的预测值为：

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$

XGBoost的目标函数为：

$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
$$

其中 $\Omega(f_t)$ 是正则化项。

对损失函数 $l$ 在 $\hat{y}_i^{(t-1)}$ 处做**二阶泰勒展开**：

$$
l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2
$$

其中：

$$
g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}
$$

$g_i$ 是一阶梯度（与GBM相同），而 $h_i$ 是**二阶梯度（Hessian）** ——这是XGBoost新增的信息。

去掉常数项后，目标函数简化为：

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)
$$

**二阶信息的价值**：一阶梯度只告诉模型“往哪个方向走”，而二阶梯度还告诉模型“每一步应该走多远”。因此，XGBoost的收敛速度比传统GBM更快。

### 1.2 正则化项：精确控制模型复杂度

XGBoost的另一个关键创新是在目标函数中**显式加入了正则化项**。

对于第 $t$ 棵树 $f_t$，其复杂度定义为：

$$
\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

其中：
- $T$ 是树的**叶子节点数量**
- $w_j$ 是第 $j$ 个叶子节点的**权重（即预测值）**
- $\gamma$ 和 $\lambda$ 是正则化参数

**两项正则化的作用**：

| 正则项 | 作用 | 效果 |
|--------|------|------|
| $\gamma T$ | 惩罚叶子节点数量 | 鼓励树结构更简单，减少分裂 |
| $\frac{1}{2}\lambda \sum w_j^2$ | L2正则化惩罚叶子权重 | 防止单个叶子权重过大，平滑预测 |

将正则化项代入目标函数，并按照叶子节点进行归组：

$$
\mathcal{L}^{(t)} = \sum_{j=1}^{T} \left[ G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 \right] + \gamma T
$$

其中 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$。

对于固定的树结构 $q(x)$，每个叶子节点的最优权重为：

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

代入后得到**结构分数（Structure Score）** ：

$$
\mathcal{L}^{(t)} = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T
$$

这个分数衡量了一棵树的质量——**值越小，树的结构越好**。

```python
import numpy as np

def xgboost_gain(G_L, H_L, G_R, H_R, G, H, lambda_=1.0, gamma=0.0):
    """
    计算XGBoost中某个分裂的增益
    G_L, H_L: 左子节点的一阶和二阶梯度之和
    G_R, H_R: 右子节点的一阶和二阶梯度之和
    G, H: 父节点的一阶和二阶梯度之和
    """
    # 分裂前的损失
    loss_before = -0.5 * (G**2 / (H + lambda_)) + gamma
    
    # 分裂后的损失
    loss_after = -0.5 * (G_L**2 / (H_L + lambda_) + G_R**2 / (H_R + lambda_)) + 2 * gamma
    
    # 增益 = 分裂前损失 - 分裂后损失
    gain = loss_before - loss_after
    return gain

# 示例
G_L, H_L = 2.0, 3.0
G_R, H_R = 1.0, 2.0
G, H = 3.0, 5.0
print(f"分裂增益: {xgboost_gain(G_L, H_L, G_R, H_R, G, H):.4f}")
```

### 1.3 预排序与Block结构

XGBoost在工程实现上的一个重要优化是**预排序（Pre-sorting）** 和 **Block结构**。

传统GBDT在寻找最佳分裂点时，每次都需要对特征值进行排序，时间复杂度高。XGBoost的做法是：

1. **训练前**，将每个特征的特征值**预先排好序**，并存储在**Block**中
2. 在寻找分裂点时，直接从Block中读取排序后的数据
3. 不同的特征Block可以**并行**处理

**注意**：XGBoost的并行是**特征维度**的并行，而不是树维度的并行——树与树之间仍然是串行训练的。

---

## 二、LightGBM：直方图算法与三大加速技术

LightGBM的名字中的“Light”意味着**轻量级**——它比XGBoost更轻、更快，特别适合大规模数据集。这一节我们深入剖析LightGBM的三大核心技术。

### 2.1 直方图算法（Histogram-based Algorithm）

XGBoost的预排序算法虽然精确，但在大数据集上内存消耗大、计算开销高。LightGBM改用**直方图算法**，将连续特征**离散化为有限个桶（bins）** 。

**算法流程**：

1. 对于每个特征，将其值域划分为 $k$ 个离散的桶（如 $k=255$）
2. 将每个样本的特征值映射到对应的桶中
3. 在训练时，基于桶的统计量（梯度之和、样本数等）来计算最佳分裂点

**直方图算法的三大优势**：

| 优势 | 说明 |
|------|------|
| **计算复杂度降低** | 预排序算法复杂度为 $O(\#data)$，直方图算法为 $O(\#bins)$，而 $\#bins \ll \#data$ |
| **内存占用减少** | 只需存储离散的桶索引（可用 `uint8_t` 存储），无需存储预排序信息 |
| **直方图做差加速** | 父节点的直方图减去兄弟节点的直方图，即可得到当前节点的直方图 |

**直方图做差（Histogram Subtraction）** 是LightGBM的一个精妙设计：在二叉树中，只要计算出左子节点的直方图，右子节点的直方图就可以通过 **“父节点直方图 - 左子节点直方图”** 快速得到，无需重新扫描数据。

```python
import numpy as np
from collections import defaultdict

class HistogramBasedSplitter:
    """直方图算法的简化实现"""
    
    def __init__(self, n_bins=255):
        self.n_bins = n_bins
    
    def build_histogram(self, feature_values, gradients, hessians):
        """构建直方图：将连续特征值分桶，统计每个桶的梯度和"""
        # 计算分桶边界
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        bin_width = (max_val - min_val) / self.n_bins
        
        hist_g = np.zeros(self.n_bins)
        hist_h = np.zeros(self.n_bins)
        hist_count = np.zeros(self.n_bins)
        
        for val, g, h in zip(feature_values, gradients, hessians):
            bin_idx = min(int((val - min_val) / bin_width), self.n_bins - 1)
            hist_g[bin_idx] += g
            hist_h[bin_idx] += h
            hist_count[bin_idx] += 1
        
        return hist_g, hist_h, hist_count
    
    def find_best_split(self, hist_g, hist_h, hist_count):
        """基于直方图寻找最佳分裂点"""
        total_g = np.sum(hist_g)
        total_h = np.sum(hist_h)
        
        best_gain = -float('inf')
        best_bin = -1
        
        left_g, left_h = 0, 0
        for i in range(self.n_bins - 1):
            left_g += hist_g[i]
            left_h += hist_h[i]
            right_g = total_g - left_g
            right_h = total_h - left_h
            
            # 计算分裂增益（XGBoost风格）
            gain = 0.5 * (left_g**2 / (left_h + 1e-6) + 
                          right_g**2 / (right_h + 1e-6) - 
                          total_g**2 / (total_h + 1e-6))
            
            if gain > best_gain:
                best_gain = gain
                best_bin = i
        
        return best_bin, best_gain
```

### 2.2 GOSS：基于梯度的单边采样

在大规模数据集上，训练样本数量巨大，如何**在不损失太多精度的前提下减少训练样本**？

**GOSS（Gradient-based One-Side Sampling，基于梯度的单边采样）** 是LightGBM的回答。

**核心思想**：

在GBDT中，**梯度大的样本意味着当前模型对其预测误差大**，需要重点学习；而**梯度小的样本已经拟合得较好**，对后续训练贡献有限。

GOSS的策略：

1. 计算所有样本在当前模型下的梯度
2. 按梯度绝对值**从大到小排序**
3. **保留所有大梯度样本**（前 $a \times 100\%$）
4. 从剩余的小梯度样本中**随机采样**（比例 $b \times 100\%$）
5. 对采样出的小梯度样本，乘以权重 $\frac{1-a}{b}$ 来补偿采样偏差

```
GOSS伪代码：
输入：数据集D，采样比例a, b，迭代次数T
1. 初始化模型
2. for t = 1 to T:
   a. 计算所有样本的梯度
   b. 按梯度绝对值排序，取前 a*100% 作为大梯度样本集 A
   c. 从剩余样本中随机采样 b*100% 作为小梯度样本集 B
   d. 对B中的样本乘以权重 (1-a)/b
   e. 用 A ∪ B 训练第t棵树
   f. 更新模型
```

**极端情况**：
- 当 $a = 0$ 时，GOSS退化为**随机采样**
- 当 $a = 1$ 时，GOSS退化为**全量训练**

GOSS的巧妙之处在于：**它保留了“难样本”（大梯度），同时用加权的方式引入了“易样本”（小梯度）的信息**，在保证精度的同时大幅减少了训练数据量。

### 2.3 EFB：互斥特征捆绑

**EFB（Exclusive Feature Bundling，互斥特征捆绑）** 是LightGBM的第三个核心技术。

**问题**：在高维稀疏数据中（如One-Hot编码后的类别特征），很多特征**几乎不会同时取非零值**——它们是**互斥的**（Exclusive）。

**EFB的核心思想**：将这些互斥的特征**捆绑（Bundle）** 成一个新的特征，从而**减少特征数量**，加速训练。

**EFB的数学化**：

将特征视为图的顶点，如果两个特征**不是互斥的**（即存在样本使两者同时非零），则在它们之间连一条边，边的权重为**冲突值**。

EFB将问题转化为**图着色问题**：用最少的颜色给顶点着色，使得相邻顶点颜色不同。每个颜色对应一个“捆绑包”。

**为什么EFB有效**：
- 稀疏数据中，互斥特征捆绑后，特征维度大幅降低
- 原本需要在 $d$ 个特征上分别寻找分裂点，现在只需在 $b$ 个捆绑特征上寻找（$b \ll d$）
- LightGBM的实验显示，整体训练速度可提升**20倍以上**

```python
# 示意：EFB的图着色思想
# 特征A和特征B互斥（不同时非零）-> 可以捆绑
# 特征A和特征C不互斥（有同时非零的样本）-> 不能捆绑

# 原始：1000个稀疏特征 -> EFB后：50个捆绑特征
# 训练速度大幅提升
```

---

## 三、Level-wise vs Leaf-wise：两种树生长策略

XGBoost和LightGBM在树生长策略上的差异，是两者最直观的区别之一。

### 3.1 XGBoost：Level-wise（按层生长）

**策略**：从根节点开始，**逐层**扩展树——先分裂当前层的所有节点，再进入下一层。

**特点**：
- 树是**平衡**的——同一层的节点深度相同
- 训练过程**稳定、可预测**
- 对参数不敏感，不容易过拟合

**缺点**：
- 可能会分裂一些**增益很小**的节点，浪费计算资源
- 在某些数据集上，不是最优的生长方式

### 3.2 LightGBM：Leaf-wise（按叶子生长）

**策略**：每次选择**增益最大**的叶子节点进行分裂，而不是按层统一分裂。

**特点**：
- 树可能**不平衡**——某些分支很深，某些很浅
- 能更快地**降低训练误差**，收敛速度更快
- 对参数更敏感（特别是 `num_leaves` 和 `min_data_in_leaf`）

**风险**：
- 在小数据集上容易**过拟合**
- 需要更精细的参数调优

### 3.3 对比总结

| 维度 | Level-wise (XGBoost) | Leaf-wise (LightGBM) |
|------|---------------------|---------------------|
| **生长方式** | 逐层分裂所有节点 | 每次选增益最大的叶子 |
| **树的平衡性** | 平衡 | 可能不平衡 |
| **收敛速度** | 较慢 | **更快** |
| **过拟合风险** | 较低 | 较高（需限制深度） |
| **参数敏感度** | 较低 | 较高 |
| **适用场景** | 小到中型数据 | **大规模数据** |

LightGBM通过 `max_depth` 参数来限制树的深度，防止leaf-wise策略导致的过拟合。

---

## 四、缺失值处理：殊途同归的智慧

XGBoost和LightGBM都**原生支持缺失值**，无需预先填充。

### 4.1 XGBoost的缺失值处理

XGBoost在训练过程中**自动学习缺失值的默认分裂方向**。

算法流程：

1. 在节点分裂时，**忽略缺失值样本**，只使用非缺失值计算最佳分裂点
2. 分别计算将缺失值**归入左子树**和**归入右子树**的增益
3. **选择增益更大的方向**作为缺失值的默认方向
4. 在预测时，缺失值样本自动走向训练时学到的默认方向

**关键点**：XGBoost的缺失值处理是**数据驱动的**——从训练数据中学习最优方向。

### 4.2 LightGBM的缺失值处理

LightGBM在直方图算法中**原生支持缺失值**：在构建直方图时，缺失值被分配到一个特殊的桶中。

在分裂时，LightGBM会**同时考虑将缺失值分配到左子树或右子树**，选择增益更大的方向。

**本质相同**：两者都是从数据中学习缺失值的最优分配方向。

```python
import xgboost as xgb
import lightgbm as lgb
import numpy as np

# XGBoost和LightGBM都默认支持缺失值
X_train = np.array([[1, 2], [np.nan, 3], [4, np.nan], [5, 6]])
y_train = np.array([0, 1, 0, 1])

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)  # 自动处理NaN

# LightGBM  
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)  # 自动处理NaN

print("两者都原生支持缺失值，无需手动填充！")
```

---

## 五、完整对比总结

| 维度 | XGBoost | LightGBM |
|------|---------|----------|
| **提出时间** | 2014年 | 2017年 |
| **分裂算法** | 预排序 + Block | **直方图算法** |
| **梯度利用** | **二阶泰勒展开**（牛顿法） | 一阶梯度（但有GOSS加速） |
| **正则化** | $\gamma T + \frac{1}{2}\lambda\sum w_j^2$ | 类似的正则化 |
| **树生长策略** | Level-wise（按层） | **Leaf-wise**（按叶子） |
| **数据采样** | 列采样（特征子采样） | **GOSS**（样本采样）+ EFB（特征捆绑） |
| **缺失值处理** | 学习默认分裂方向 | 直方图特殊桶处理 |
| **类别特征** | 需预处理（One-Hot） | **原生支持** |
| **内存占用** | 较高（预排序存储） | **较低**（直方图存储） |
| **训练速度** | 较快 | **更快**（特别是大数据） |
| **适用场景** | 小到中型数据、需要稳定性 | **大规模数据**、追求速度 |

### 核心要点回顾

1. **XGBoost的二阶泰勒展开**：将损失函数展开到二阶，利用一阶梯度 $g_i$ 和二阶梯度 $h_i$ 共同决定分裂方向，收敛速度比传统GBM更快。正则化项 $\gamma T + \frac{1}{2}\lambda\sum w_j^2$ 精确控制模型复杂度，防止过拟合。

2. **LightGBM的直方图算法**：将连续特征离散化为有限个桶，将分裂查找复杂度从 $O(\#data)$ 降至 $O(\#bins)$，内存占用大幅降低。直方图做差进一步加速了训练。

3. **GOSS采样**：保留所有大梯度样本（难样本），从小梯度样本中随机采样并加权，在保证精度的同时大幅减少训练数据量。

4. **EFB特征捆绑**：将互斥的稀疏特征捆绑成一个特征，减少特征维度，加速训练。

5. **Level-wise vs Leaf-wise**：XGBoost按层生长，树平衡稳定；LightGBM按叶子生长，每次选增益最大的叶子分裂，收敛更快但需防过拟合。

6. **缺失值处理**：两者都原生支持缺失值，从数据中学习最优的分配方向，无需手动填充。