---
title: Multimodal RAG —— 多模态RAG
published: 2026-08-12
description: 系统讲解多模态文档检索的前沿技术ColPali：从传统OCR流水线的脆弱性出发，剖析ColPali如何通过PaliGemma视觉语言模型直接编码文档页面图像，跳过OCR与布局解析环节；深入解读延迟交互机制中MaxSim评分函数的数学原理及其相比单向量余弦相似度的精度优势；分析多粒度索引策略与跨模态查询重写技术。
cover: "/assets/images/posts/multimodal_rag.png"
coverInContent: false
tags: [RAG, ColPali, 延迟交互, 多模态]
category: AI_Multimodal
draft: false
---

# Multimodal RAG —— 多模态RAG

## 引言：当RAG遇上“看不懂”的文档

检索增强生成（RAG）已成为大语言模型落地应用的标准范式——先检索相关知识，再让LLM基于这些知识生成回答。然而，当知识库中充斥着PDF报告、扫描件、财务报表这类**视觉密集型文档**时，传统的RAG系统就撞上了一堵墙。

想象一下：你问AI“2024年Q3营收最高的是哪个季度？”，系统需要从一份满是表格和折线图的年报PDF中找到答案。传统RAG的做法是：先用OCR提取文字，再用布局分析识别表格结构，最后把文本分块索引。这个过程**漫长、脆弱且容易出错**——OCR可能认错数字，布局解析可能把跨页表格拆散，图表中的视觉信息则直接被丢弃。

2024年，**ColPali**的诞生彻底改变了这一局面。它提出一个近乎“野蛮”的想法：**直接把PDF页面当图片处理，用视觉语言模型编码整页图像，端到端地完成检索**。本文将从ColPali的核心原理出发，深入延迟交互机制、多粒度索引策略和查询重写技术，带你全面理解多模态RAG的前沿进展。

---

## 一、传统文档检索的痛点：为什么OCR不够用？

### 1.1 脆弱的OCR流水线

传统文档检索系统的标准流程是：

1. **OCR识别**：将扫描件或PDF转换为可编辑文本
2. **布局解析**：识别标题、段落、表格、图片等区域
3. **文本分块**：将长文档切分为适合检索的文本块
4. **向量化索引**：将文本块通过嵌入模型转换为向量
5. **相似度检索**：计算查询向量与文档块向量的余弦相似度

这条流水线存在几个根本性问题：

- **信息丢失**：图表、布局、字体等视觉线索被完全丢弃
- **错误累积**：OCR的识别错误会逐级放大
- **流程复杂**：需要维护多个独立组件（OCR引擎、布局模型、分块策略等）
- **适应性差**：面对不同格式的文档（扫描件、原生PDF、带水印的文档），每个组件都需要单独调优

### 1.2 视觉信息的价值

文档不仅是文字的载体，更是**视觉信息的综合体**。一份PDF年报中：

- **表格**：通过行列结构和边框传达了数据的组织关系
- **图表**：用颜色、位置、大小编码了多维度信息
- **布局**：标题层级、段落缩进、分栏排版暗示了内容的逻辑结构
- **字体**：粗体、斜体、字号变化承载了语义重点

这些视觉线索在纯文本表示中全部丢失。正如ColPali论文所言：“文档是通过文本、图表、页面布局、表格甚至字体来传达信息的视觉丰富结构”。

---

## 二、ColPali：直接编码文档图像的端到端方案

### 2.1 核心思想：跳过OCR，直接处理页面图像

ColPali的核心思想极其简洁：**将文档页面视为图像，直接用视觉语言模型（VLM）编码整页，生成高质量的向量嵌入**。

这个思路跳过了整个OCR+布局解析的复杂流水线。ColPali的工作流程分为两个阶段：

**索引阶段（离线）** ：
1. 将PDF每一页渲染为图像（截图）
2. 将每页图像输入PaliGemma视觉语言模型
3. 模型输出**每个图像块（patch）的向量**，而非整页的单一向量
4. 存储所有patch向量供后续检索

**检索阶段（在线）** ：
1. 将用户查询文本输入同一个模型，输出**每个查询token的向量**
2. 通过**延迟交互**机制计算查询token与文档patch之间的匹配分数
3. 聚合所有匹配分数，得到文档与查询的相关性得分

### 2.2 架构解析：PaliGemma + ColBERT策略

ColPali建立在Google的**PaliGemma**视觉语言模型之上。PaliGemma是一个3B参数的多模态模型，包含：

- **视觉编码器**：SigLIP-SO400M，将图像分割为一系列patch进行处理
- **语言模型**：Gemma，处理文本输入并生成上下文感知的表示

ColPali在PaliGemma的基础上，添加了一个**ColBERT风格的线性投影层**。这个投影层将PaliGemma的隐藏状态映射到低维嵌入空间（通常为128维），并对每个输出进行L2归一化。

```python
# ColPali的嵌入生成流程（简化）
# 1. 图像经过SigLIP视觉编码器 → patch特征
# 2. patch特征输入Gemma语言模型 → 上下文感知的隐藏状态
# 3. 线性投影层 (hidden_size → 128) → 每个patch的128维向量
# 4. L2归一化 → 可用于相似度计算的嵌入向量
```

与传统的单向量检索模型（如CLIP/SigLIP）不同，ColPali为**每页文档生成数百甚至上千个向量**——每个图像块对应一个向量。一张PDF页面经过处理后，产生的patch数量取决于图像分辨率：

- 典型配置下，一张页面生成约 **1024个** patch向量（对应32×32的网格）
- 每个向量维度为 **128**

### 2.3 训练数据与策略

ColPali的训练数据集包含 **127,460个查询-页面配对**，由两部分组成：

- **公开学术数据集（63%）** ：来自多个学术基准的标注数据
- **合成数据集（37%）** ：从网络爬取的PDF页面，使用Claude-3 Sonnet生成伪查询

这种数据组合保证了模型既能处理标准学术文档，也能泛化到真实世界的多样化文档。

---

## 三、延迟交互（Late Interaction）：检索精度的关键

### 3.1 从“早交互”到“晚交互”

理解延迟交互，首先要理解信息检索中的两种交互范式：

**早交互（Early Interaction）** ：查询和文档在编码阶段就进行交互（如双编码器同时处理查询和文档）。这种方法计算密集，但能捕获细粒度的匹配信号。

**晚交互（Late Interaction）** ：查询和文档**独立编码**，只在最后的评分阶段才进行交互。这种方法将计算密集的编码过程离线完成，在线检索时只需计算轻量级的交互分数。

ColPali继承自ColBERT的**延迟交互范式**，其核心优势在于：**在保持高检索精度的同时，实现了高效的在线检索**。

### 3.2 MaxSim评分机制

ColPali采用**MaxSim**（Maximum Similarity）作为评分函数。

给定一个查询（包含 $M$ 个token向量 $\{q_1, q_2, ..., q_M\}$）和一个文档页面（包含 $N$ 个patch向量 $\{d_1, d_2, ..., d_N\}$），MaxSim的计算方式为：

$$\text{Score}(Q, D) = \sum_{i=1}^{M} \max_{j=1}^{N} \text{sim}(q_i, d_j)$$

其中 $\text{sim}$ 通常采用**余弦相似度**（因为向量已经过L2归一化）。

这个公式的直观理解是：**对于查询中的每个token，找到文档中与之最相似的patch，记录这个最大相似度；然后将所有token的最大相似度求和**，作为文档与查询的整体相关性得分。

```python
import torch
import torch.nn.functional as F

def maxsim_score(query_embeddings, doc_embeddings):
    """
    计算MaxSim延迟交互得分
    
    Args:
        query_embeddings: [num_query_tokens, embed_dim] 查询token向量
        doc_embeddings: [num_doc_patches, embed_dim] 文档patch向量
    
    Returns:
        float: 文档与查询的相关性得分
    """
    # 计算所有query token与所有doc patch的相似度矩阵
    # [num_query_tokens, num_doc_patches]
    sim_matrix = query_embeddings @ doc_embeddings.T
    
    # 对每个query token，取最大相似度
    max_per_token = sim_matrix.max(dim=1).values  # [num_query_tokens]
    
    # 求和得到最终得分
    return max_per_token.sum().item()
```

### 3.3 为什么延迟交互更有效？

传统双编码器（如CLIP/SigLIP）使用**单向量余弦相似度**：将整页文档压缩为一个向量，查询也压缩为一个向量，然后计算两个向量之间的余弦相似度。

这种方法的问题在于：**信息压缩损失**。一页PDF包含丰富的视觉信息（文字、表格、图表、布局），将其压缩为单一向量必然导致信息丢失。当查询关注的是文档中的某个特定区域（如“表格第三行第二列的数字”）时，单向量表示难以精准定位。

ColPali的**多向量延迟交互**完美解决了这个问题：

- **多向量表示**：每个patch独立编码，保留了空间位置信息
- **细粒度匹配**：每个查询token可以精准匹配到最相关的patch区域
- **隐式OCR**：模型能够将查询词“对齐”到图像中的具体文字或图表区域，而无需显式识别字符

实验表明，在ViDoRe基准上，ColPali的NDCG@5达到了 **81.3**，远超传统非结构化方法的67.0。

### 3.4 可解释性：延迟交互的“副产物”

延迟交互机制带来的一个意外收获是**可解释性**。由于我们知道每个查询token与哪些文档patch产生了高相似度，可以将这些匹配关系**叠加显示在原始文档图像上**，生成热力图。

这种可视化让我们可以直观地看到：模型在检索时“看”向了文档的哪些区域。例如，查询“最高发电量在哪个小时？”时，热力图会高亮显示图表中数据点最高的区域。这种透明度对于需要审计和验证的行业应用（如金融、医疗）尤为重要。

---

## 四、ColPali vs SigLIP：多向量与单向量的对决

SigLIP（我们在第一篇中详细介绍过）是单向量多模态编码器的代表，而ColPali是多向量延迟交互的代表。两者的对比清晰地展示了技术路线的分野：

| 维度 | SigLIP（单向量） | ColPali（多向量） |
|------|-----------------|-------------------|
| 每页向量数 | 1个 | ~1024个（每个patch一个） |
| 评分方式 | 余弦相似度 | MaxSim延迟交互 |
| 信息粒度 | 全局 | 局部（patch级） |
| 检索精度 | 较高 | **显著更高** |
| 存储开销 | 低 | 高（约1000倍） |
| 推理速度 | 快 | 较慢 |

ColPali的优势在于**精度**。patch级的多向量表示可以精准定位表格、图表和密集数值区域，而这些区域在全局嵌入中往往被“平滑”掉了。

SigLIP的优势在于**效率**。单向量表示存储和检索都更轻量，适合对延迟敏感的场景。

在实际应用中，两者并非互斥——可以采用**级联检索**策略：先用SigLIP做粗筛（召回候选集），再用ColPali做精排（重排序）。

---

## 五、多粒度向量索引：从页面到区块

### 5.1 页面级索引的局限

ColPali原生支持**页面级检索**——给定一个查询，判断哪一页文档最相关。这在很多场景下已经足够（如“找到包含Q3财报的那一页”）。

但当文档页面内容庞杂时（如一整页包含多个独立章节），页面级检索可能不够精细。用户可能需要的是**页面中的某个特定区块**（如某个表格、某段文字），而非整页。

### 5.2 多粒度索引策略

多粒度索引的核心思想是：**在不同粒度上建立索引，根据查询的粒度动态选择最合适的检索层级**。

常见的多粒度层级包括：

1. **页面级（Page-level）** ：整页作为一个索引单元。这是ColPali的原生粒度
2. **区块级（Region-level）** ：将页面分割为语义区块（标题、段落、表格、图片等），每个区块独立索引
3. **图块级（Tile-level）** ：将页面切分为更小的图块（如2×2网格），用于精确定位

在实践中，可以采用**三级级联检索**策略：

```python
# 三级级联检索伪代码
def multi_granularity_retrieval(query, pages):
    # 第一级：页面级粗筛
    candidate_pages = colpali_retrieve(query, pages, top_k=10)
    
    # 第二级：区块级精排
    candidate_regions = []
    for page in candidate_pages:
        regions = detect_regions(page)  # 标题、段落、表格等
        candidate_regions.extend(regions)
    top_regions = colpali_retrieve(query, candidate_regions, top_k=20)
    
    # 第三级：图块级定位（如需精确定位）
    if need_precise_localization:
        tiles = split_into_tiles(top_regions, grid=2x2)
        top_tiles = colpali_retrieve(query, tiles, top_k=5)
    
    return top_tiles or top_regions
```

这种多粒度设计让系统既能利用ColPali的视觉理解能力进行全局检索，又能通过细粒度索引实现精准定位。

---

## 六、跨模态查询重写：让查询“看得见”

### 6.1 口语化查询 vs 检索友好型查询

用户查询天然是**口语化**的——“帮我找一下那个关于营收的表格”、“去年利润最高的季度是哪张图”。而检索模型期望的输入是**与文档内容在语义空间中对齐的表示**。

两者之间存在**语义鸿沟**。用户的口语化查询可能包含指代（“那个”、“这个”）、隐含假设（“你知道的”）、模糊描述（“关于营收的”），这些在检索时都难以直接匹配。

### 6.2 查询重写的基本思路

**查询重写（Query Rewriting）** 的核心目标是将用户口语化查询转化为**适合检索的“视觉描述词”**。

这不仅仅是同义词替换，而是**模态转换**——将文本查询转换为更接近文档视觉内容的描述。例如：

- 原始查询：“去年利润最高的季度”
- 重写后：“展示2023年各季度利润的柱状图，其中Q3柱子最高”
- 重写后（更视觉化）：“利润柱状图，Q3为最高值，红色柱子”

重写后的查询包含了更多**视觉线索**（柱状图、红色、最高值），这些线索更容易与文档图像的视觉特征对齐。

### 6.3 两阶段查询重写框架

一个典型的两阶段查询重写框架如下：

**第一阶段：图像描述生成**

利用视觉语言模型（如GPT-4V、Claude-3）对文档页面生成密集描述，提取其中的视觉元素：“这是一份包含折线图和表格的财报页面，折线图显示Q3营收达到峰值$2.1B...”

**第二阶段：查询重构**

基于生成的图像描述和原始查询，将查询重写为检索友好的形式：“营收折线图，Q3峰值，$2.1B，财报页面”

研究表明，这种查询重写策略能显著提升多模态检索的召回率。

```python
def query_rewrite_for_retrieval(original_query, page_image, vlm):
    """
    将口语化查询重写为检索友好的视觉描述
    """
    # 第一阶段：生成页面视觉描述
    visual_description = vlm.generate_caption(
        page_image, 
        prompt="详细描述这张图片中的视觉元素，包括图表类型、颜色、数值等"
    )
    
    # 第二阶段：基于描述重写查询
    rewritten_query = vlm.generate(
        prompt=f"""
        原始查询：{original_query}
        页面视觉描述：{visual_description}
        请将原始查询重写为适合多模态检索的查询，突出视觉线索。
        """
    )
    
    return rewritten_query
```

---

## 七、代码实践：ColPali的端到端实现

### 7.1 环境准备

```bash
pip install colpali-engine>=0.3.0 torch transformers Pillow
```

### 7.2 加载模型与处理器

```python
import torch
from PIL import Image
from transformers import ColPaliForRetrieval, AutoProcessor

# 加载ColPali模型和处理器
model_name = "vidore/colpali-v1.3-hf"
model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# 模型配置：128维嵌入
print(f"嵌入维度: {model.config.dim}")  # 128
```

### 7.3 文档索引：生成多向量嵌入

```python
def index_document_page(page_image_path):
    """
    为单页文档生成多向量嵌入（patch-level）
    """
    # 加载页面图像
    image = Image.open(page_image_path)
    
    # 处理图像（自动resize到448x448）
    batch_images = processor.process_images([image])
    
    # 生成patch嵌入
    with torch.no_grad():
        # 输出形状: [1, num_patches, 128]
        doc_embeddings = model.forward(
            pixel_values=batch_images,
            input_ids=None,  # 无文本输入，仅编码图像
        )
    
    return doc_embeddings.squeeze(0)  # [num_patches, 128]

# 示例：索引一个PDF页面
page_embeddings = index_document_page("page_1.png")
print(f"生成了 {page_embeddings.shape[0]} 个patch向量")
# 输出: 生成了 1024 个patch向量
```

### 7.4 查询编码与检索

```python
def search_query(query_text, doc_embeddings_list, model, processor, top_k=5):
    """
    使用ColPali进行检索
    
    Args:
        query_text: 用户查询
        doc_embeddings_list: 所有文档页面的patch嵌入列表
        top_k: 返回最相关的k个文档
    """
    # 1. 编码查询：生成每个token的嵌入
    with torch.no_grad():
        # 处理查询文本
        query_inputs = processor.process_queries([query_text])
        query_embeddings = model.forward(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
        )  # [1, num_tokens, 128]
    
    query_embeddings = query_embeddings.squeeze(0)  # [num_tokens, 128]
    
    # 2. 对每个文档计算MaxSim得分
    scores = []
    for doc_id, doc_emb in enumerate(doc_embeddings_list):
        # doc_emb: [num_patches, 128]
        # 计算相似度矩阵 [num_tokens, num_patches]
        sim_matrix = query_embeddings @ doc_emb.T
        
        # MaxSim: 对每个token取最大相似度，然后求和
        max_per_token = sim_matrix.max(dim=1).values
        score = max_per_token.sum().item()
        scores.append((doc_id, score))
    
    # 3. 按得分排序返回top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# 示例检索
results = search_query(
    "Which hour of the day had the highest electricity generation in 2019?",
    doc_embeddings_list,  # 所有页面的嵌入列表
    model,
    processor,
    top_k=3
)

for doc_id, score in results:
    print(f"文档 {doc_id}: 得分 {score:.4f}")
```

### 7.5 可视化注意力热力图

```python
def visualize_attention(query_text, page_image_path, model, processor):
    """
    可视化查询token与文档patch的匹配热力图
    """
    image = Image.open(page_image_path)
    
    # 编码查询和文档
    with torch.no_grad():
        query_inputs = processor.process_queries([query_text])
        query_embeddings = model.forward(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
        ).squeeze(0)  # [num_tokens, 128]
        
        batch_images = processor.process_images([image])
        doc_embeddings = model.forward(
            pixel_values=batch_images,
            input_ids=None,
        ).squeeze(0)  # [num_patches, 128]
    
    # 计算相似度矩阵 [num_tokens, num_patches]
    sim_matrix = query_embeddings @ doc_embeddings.T
    
    # 对每个patch，取所有query token中的最大相似度
    patch_importance = sim_matrix.max(dim=0).values  # [num_patches]
    
    # 将patch重要性映射回图像网格（如32x32）
    grid_size = int(patch_importance.shape[0] ** 0.5)  # 32
    heatmap = patch_importance.reshape(grid_size, grid_size)
    
    # 上采样到原始图像大小并叠加显示
    # ... 使用matplotlib或PIL绘制热力图
    return heatmap
```

---

## 八、总结与展望

ColPali的提出标志着多模态文档检索的一次范式转变：

| 维度 | 传统OCR流水线 | ColPali |
|------|--------------|---------|
| 处理方式 | 文本提取 → 分块 → 索引 | 直接编码页面图像 |
| 信息利用 | 仅文本 | 文本+布局+图表+视觉 |
| 流程复杂度 | 高（多组件） | 低（端到端） |
| 检索精度 | 中 | **高**（NDCG@5: 81.3 vs 67.0） |
| 可解释性 | 低 | 高（热力图可视化） |

ColPali的成功验证了一个核心理念：**对于视觉密集型文档，“看懂”比“读懂”更重要**。跳过OCR的脆弱流水线，直接用视觉语言模型理解文档的视觉结构，不仅更简单，而且更准确。

然而，ColPali也面临着现实挑战：

- **存储开销**：每页1024个向量，比单向量模型多三个数量级
- **推理延迟**：MaxSim计算需要遍历所有patch，比余弦相似度慢
- **语言覆盖**：主要针对高资源语言，对其他语言的泛化能力有限

这些挑战正在被学术界和工业界积极解决。**Light-ColPali**通过动态剪枝和量化减少存储，**HPC-ColPali**采用层级化patch压缩，而Elasticsearch和Vespa等向量数据库已开始原生支持ColPali的延迟交互检索。

正如ColPali论文的作者所言：“我们发布ColPali，一个经过训练可从文档页面图像中生成高质量多向量嵌入的视觉语言模型。结合延迟交互匹配机制，ColPali在大幅超越现代文档检索流水线的同时，更加简单、快速且可端到端训练。”这或许就是多模态RAG的未来方向——让机器像人一样，“看”懂文档。