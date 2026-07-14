---
title: Retrieval-Augmented Generation —— 检索增强生成
published: 2026-03-08
description: 系统讲解检索增强生成（RAG）的核心技术原理与工程实践：从文档分块的三种策略及其适用场景，到向量检索中ANN算法的数学原理与选型指南，再到重排序阶段Bi-Encoder与Cross-Encoder的架构差异及其对MRR/NDCG的量化提升，最后介绍HyDE如何通过“假设性文档”将检索从关键词匹配升级为意图理解。
cover: "/assets/images/posts/rag.png"
coverInContent: false
tags: [AI Agent, RAG, 检索增强生成, 分块策略, 重排序]
category: AI_Agent
draft: false
---

# Retrieval-Augmented Generation —— 检索增强生成

## 引言：大模型的“开卷考试”

大语言模型虽然强大，但它的知识截止于训练数据的那一天。当用户问“昨天发生了什么新闻”时，模型要么胡编乱造（幻觉），要么承认不知道。

**检索增强生成（Retrieval-Augmented Generation, RAG）** 的解决思路极其优雅：不让模型闭卷考试，而是允许它**开卷**——先从外部知识库中检索相关文档，再把文档和问题一起交给模型生成答案。RAG通过注入外部知识来提升LLM的事实准确性。

一个标准的RAG系统包含三个核心阶段：

1. **索引（Indexing）** ：将文档分割成块 → 生成向量嵌入 → 存储至向量数据库
2. **检索（Retrieval）** ：将用户查询向量化 → 检索Top-K相关文本块
3. **生成（Generation）** ：将检索结果与查询拼接 → 输入LLM生成答案

本文将从数学原理和工程实践出发，系统讲解RAG的三个核心环节——**索引（分块策略）** 、**检索（ANN算法）** 与**重排序（Reranker）** ，并介绍HyDE这一前沿检索增强技术。

---

## 一、索引——文档分块（Chunking）策略

### 1.1 为什么分块是RAG的“第一性原理”

在将文档存入向量数据库之前，必须先将其分割成更小的片段——**分块（Chunking）** 。分块方式对检索质量的影响，超过RAG流水线中几乎任何其他决策。

核心矛盾非常直观：

- **块太小**：每个向量只代表“思想的碎片”，检索返回的片段缺乏回答问题所需的上下文，答案散落在五个块里
- **块太大**：每个向量平均了多个主题，相似度被稀释，无关内容挤占了上下文窗口

**没有普遍适用的“最佳”块大小**。正确的尺寸取决于文档类型、查询模式、嵌入模型的token限制以及LLM需要多少上下文才能生成好答案。

### 1.2 三种主流分块策略

#### 策略一：固定大小分块（Fixed-size Chunking）

最直观的方法：按预定义的字符数或token数将文本切成大小统一的块。

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  # 重叠区缓解上下文断裂
    separator="\n"
)
chunks = splitter.split_documents(docs)
```

**工作原理**：一个滑动窗口在文档上移动，产生等大小的块。通常设置10–20%的重叠，让上一个块末尾的内容在下一个块开头重复出现，减少关键句被切断的风险。

**优缺点**：
- 优点：实现简单、存储成本可预测、索引速度快、适用于任何文档类型
- 缺点：会在句子中间截断、没有语义连贯性、重叠造成内容重复

**适用场景**：快速原型验证、同质化纯文本语料、需要以最低延迟索引海量数据的场景。

#### 策略二：递归分块（Recursive Chunking）

这是一种更智能的组合式策略：**按优先级顺序尝试多种分隔符进行递归分割**。

**工作原理**：首先尝试用段落分隔符（`\n\n`）分割；如果段落仍然过大，再按句子分割；最后才按字符数强制分割。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]  # 优先级从高到低
)
```

递归分块是LangChain和LlamaIndex的**默认策略**，最适合通用文章、博客文章和混合内容。

#### 策略三：语义分块（Semantic Chunking）

不再基于字符数，而是基于**语义边界**进行分割。

**工作原理**：
1. 将文档拆分为句子
2. 为每个句子生成嵌入向量
3. 计算相邻句子嵌入的余弦相似度
4. 如果相似度高于阈值，合并到同一个块；否则开始新块

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def semantic_chunking(sentences, threshold=0.85):
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        emb1 = model.encode(current_chunk[-1])
        emb2 = model.encode(sentences[i])
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        if sim > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

语义分块保持了语言的自然流畅，保留了完整的思想单元，因此检索准确率更高。挑战在于**阈值选择**——不同文档的最佳阈值可能不同。

### 1.3 分块策略的选择指南

| 策略 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| 固定大小 | 最快 | 最低 | 快速原型、结构化数据 |
| 递归 | 快 | 中等 | 通用文章、博客（默认首选） |
| 语义 | 较慢 | 最高 | 需要高质量检索的场景 |

实践中，**递归分块是大多数RAG系统的最佳起点**。如果检索质量不满足要求，再考虑升级到语义分块。

---

## 二、检索——向量索引与ANN算法

### 2.1 从暴力搜索到近似最近邻

分块完成后，每个块通过嵌入模型转换为高维向量（通常128–4096维）。当用户提出查询时，系统需要在高维向量空间中找到与查询向量**最相似**的Top-K个向量。

最直接的方法是**暴力搜索（FLAT）** ：计算查询向量与**所有**向量的距离，排序后取Top-K：

```python
def flat_search(query_vector, vectors_db, top_k):
    distances = [euclidean_distance(query_vector, v) for v in vectors_db]
    return sorted(zip(distances, vectors_db))[:top_k]
```

暴力搜索的时间复杂度为 **O(n×d)** （n为向量数量，d为维度）。在百万级数据下，延迟达到10–100ms，十亿级则完全不可用。

**近似最近邻搜索（Approximate Nearest Neighbor, ANN）** 应运而生：**牺牲少量准确率，换取数量级的性能提升**。ANN索引能将百万级搜索延迟从秒级压缩到**毫秒级**。

### 2.2 HNSW：基于图的渐进式搜索

**HNSW（Hierarchical Navigable Small World，分层可导航小世界）** 是当前最流行的ANN算法之一。

**核心思想**：构建**多层图结构**——底层包含所有节点，上层是下层的稀疏子集。

**工作原理**：
1. **构建阶段**：将向量逐层插入图中，高层节点少、连接稀疏，低层节点多、连接稠密
2. **搜索阶段**：从最高层开始，在每层中找到与查询向量最近的节点，然后逐层向下细化

形象地说，HNSW像在**多层高速路网**中导航：先在顶层高速上快速前进（跳过大量无关节点），然后逐层下到更细的路网，最终精确定位。

**数学本质**：HNSW通过图结构将搜索空间从 **O(n)** 降低到 **O(log n)** 。某开源向量数据库的基准测试显示，采用HNSW算法的千万级向量检索比暴力搜索**快3个数量级**。

**优缺点**：
- 优点：查询延迟极低（毫秒级）、适合高维向量
- 缺点：**内存占用高**，呈线性增长

**适用场景**：数据规模适中（百万到千万级）、内存充裕、对查询延迟要求极高的场景。

### 2.3 IVF-PQ：量化压缩的存储优化

当数据规模达到**亿级甚至十亿级**时，HNSW的内存占用可能成为瓶颈。**IVF-PQ（倒排文件 + 乘积量化）** 通过**量化压缩**将内存占用降低**10倍以上**。

**IVF（倒排文件）** ：通过K-means聚类将数据划分为多个簇（通常100–1000个），检索时先定位候选簇，再在簇内搜索。

**PQ（乘积量化）** ：将高维向量分割为多个子向量，对每个子空间进行聚类，用聚类中心索引替代原始向量存储。

**压缩效果惊人**：
- 原始向量：4字节 × 128维 = **512字节**
- PQ编码后：2字节 × 16子空间 = **32字节**
- **压缩比 16:1**

**优缺点**：
- 优点：内存占用极低、适合超大规模数据集
- 缺点：查询延迟略高于HNSW、存在量化精度损失

### 2.4 索引算法选型指南

| 算法 | 内存占用 | 查询延迟 | 准确率 | 适用数据规模 |
|------|---------|---------|--------|-------------|
| FLAT（暴力） | 极高 | 极慢 | 100% | <10万 |
| HNSW | 高 | 极快（毫秒级） | 95–99% | 百万–千万级 |
| IVF-PQ | **极低** | 较快 | 90–95% | 亿级+ |

**选型建议**：
- **小规模（<10万）** ：FLAT足够，100%召回率
- **中等规模（百万–千万）** ：**HNSW是首选**，查询延迟最低
- **超大规模（亿级+）** ：**IVF-PQ**，内存占用是关键约束

---

## 三、重排序（Reranker）——从“召回”到“精排”

### 3.1 为什么需要重排序？

第一阶段的ANN检索优化的是**召回（Recall）** ——确保相关文档不被漏掉。但召回的Top-K中，**最相关的文档不一定排在第一位**。

这个问题的代价有多大？**模型只读取提示词中的内容。如果正确的块排在第7位，它和第700位没什么区别**。

**重排序（Reranking）** 就是来解决这个问题的：用**一个更精确但更慢的模型**，对初排返回的少量候选（50–100个）进行**重新打分和排序**，把最相关的推到最前面。

### 3.2 Bi-Encoder vs Cross-Encoder：两种架构的本质差异

理解重排序，首先需要理解两种编码器架构的根本区别：

**Bi-Encoder（双编码器）** ：查询和文档**分别独立编码**成向量，然后计算向量相似度。

```
Query → Encoder → q_vector ─┐
                              ├→ 相似度计算
Doc   → Encoder → d_vector ─┘
```

Bi-Encoder的优势是**速度快**——文档向量可以预先计算并存储，查询时只需要编码查询向量一次。缺点是**交互不足**——查询和文档在编码过程中互不知晓对方的存在。

**Cross-Encoder（交叉编码器）** ：查询和文档**拼接在一起**，作为一个整体输入模型。

```
[Query + Doc] → Encoder → 相关性分数
```

Cross-Encoder的优势是**精度高**——模型可以在编码过程中让查询和文档的token充分交互，深层理解语义关系。缺点是**速度慢**——每对（查询，文档）都需要独立计算，无法预计算。

**重排序的核心逻辑**：用Bi-Encoder做**快速召回**（50–100个候选），用Cross-Encoder做**精准排序**（选出Top-3～5）。

### 3.3 Reranker的量化效果

一份真实的RAG系统评测数据显示：

| 阶段 | Recall@5 | NDCG@5 | MRR |
|------|----------|--------|-----|
| 纯向量检索 | 0.88 | 0.75 | 0.74 |
| 混合检索+RRF | 0.90 | 0.79 | 0.78 |
| **+ Reranker精排** | **0.90** | **0.85** | **0.88** |

Reranker不改变候选集（Recall不变），但**NDCG提升了0.06，MRR提升了0.09**。这意味着**最相关的文档被推到了更靠前的位置**——这正是Reranker的价值所在。

**MRR（Mean Reciprocal Rank）** 衡量的是**第一个正确答案的排名倒数**。MRR从0.78提升到0.88，意味着第一个正确答案的平均排名从约1.28提升到了约1.14——**首条命中率显著提高**。

在生产环境中，一个设计良好的重排序层可以将 **precision@3提升40–60%** ，增加延迟不到200ms。

### 3.4 主流Reranker模型

当前生产环境中最主流的三个Reranker：

1. **BGE-Reranker-v2-m3**（BAAI开源）：约5.68亿参数，多语言，CPU上对50个候选批处理约80ms。**开源方案的首选**。

2. **Cohere Rerank**（托管API）：商业方案，精度高但需要付费。

3. **Voyage rerank-2**：新兴方案。

### 3.5 Reranker的代码实现

```python
from sentence_transformers import CrossEncoder
from typing import List, Tuple

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化Cross-Encoder重排序器
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        对候选文档进行重排序
        
        数学上，这等价于：
        score_i = CrossEncoder(query, doc_i)  for each candidate i
        sorted_candidates = sort_by_score(descending)
        
        Cross-Encoder将query和doc拼接后一起编码，
        让两者充分交互，产生更精确的相关性分数
        """
        # 构建(query, document)对
        pairs = [(query, doc) for doc in candidates]
        
        # Cross-Encoder计算相关性分数
        scores = self.model.predict(pairs)
        
        # 按分数降序排列
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]

# 使用示例
reranker = Reranker()
query = "什么是检索增强生成？"
candidates = [...]  # 从向量数据库检索到的50-100个候选
top_results = reranker.rerank(query, candidates, top_k=5)
```

**工程实践要点**：
- 第一阶段召回**50–100个候选**，而不是10个——Recall在这里是廉价的
- 生产环境需要**降级方案**：本地Cross-Encoder为主，托管API为备选
- 设置**延迟和成本预算**，超时则优雅降级

---

## 四、HyDE——让检索从“匹配关键词”升级到“理解意图”

### 4.1 传统检索的盲区

传统RAG的检索流程是：Query → Embedding → Vector Search → Retrieved Chunks。

问题在于：**向量数据库基于语义相似度检索，但相似≠相关**。

举例来说，用户问：“LangSmith如何帮助监控LLM应用？”如果存储的文档里从未出现过“monitor”、“tracking”或“observability”这些词，哪怕文档里其实写了答案，检索质量也会大打折扣。

本质原因是：**短查询的语义信息量太少**，无法充分表达用户的真实意图。

### 4.2 HyDE的核心思想

**HyDE（Hypothetical Document Embeddings，假设性文档嵌入）** 由Luyu Gao等人提出。核心思路极其简洁而巧妙：

> **不直接嵌入用户的查询，而是先让LLM生成一份“假设性答案文档”，再嵌入这份假设文档去检索。**

```
传统RAG:  Query → Embed → Search
HyDE:     Query → LLM生成假设文档 → Embed假设文档 → Search
```

生成的假设文档**比原始短查询承载了更丰富的语义**。检索系统搜索的是“上下文意图”而非“关键词匹配”。

### 4.3 数学原理

传统密集检索中，查询和文档被编码为单向量表示，检索基于向量相似度。

HyDE的核心创新在于**将查询从“问题空间”映射到“答案空间”** ：

设查询为 $q$，传统方法直接检索：

$$\text{retrieved} = \text{ANN}(\text{Embed}(q))$$

HyDE引入一个生成步骤：

$$\text{hypothesis} = \text{LLM}(q)$$
$$\text{retrieved} = \text{ANN}(\text{Embed}(\text{hypothesis}))$$

由于假设文档 $h$ 在语义上更接近真实文档的分布（都是“答案”而非“问题”），嵌入空间中的距离更小，检索准确率更高。

### 4.4 HyDE的代码实现

```python
def hyde_retrieve(query: str, llm, embed_model, vector_db, top_k: int = 10) -> List[str]:
    """
    HyDE检索流程
    """
    # 1. 生成假设性答案文档
    prompt = f"""
    请为以下问题生成一个详细的答案文档。即使你不确定答案，也请生成一个合理的假设性答案。
    
    问题: {query}
    
    假设性答案:
    """
    hypothetical_doc = llm.generate(prompt)
    
    # 2. 嵌入假设文档（而非原始查询）
    query_embedding = embed_model.encode(hypothetical_doc)
    
    # 3. 用假设文档的嵌入进行向量检索
    results = vector_db.search(query_embedding, top_k=top_k)
    
    return results
```

### 4.5 HyDE的实践要点

**优点**：
- **投入产出比极高**：代码改动小，通常能快速提升召回率
- **零样本**：无需特定任务训练数据
- **多语言支持**：在各种任务中表现良好

**注意事项**：
- 生成的假设文档可能**不准确**——但这没关系，它只是检索的“桥”，不是最终答案
- HyDE最好与**重排序（Reranker）** 配合使用
- 实践中可以生成**多个假设文档**（如5个）并融合检索结果

---

## 五、总结：RAG检索链路的三级火箭

RAG的检索质量，取决于**索引、检索、重排序**三个环节的协同优化：

**第一级：索引（分块策略）** ——决定了“知识库”的质量。分块太粗，检索精度下降；分块太细，上下文割裂。**递归分块是最稳妥的起点**。

**第二级：检索（ANN算法）** ——决定了“找得到”的速度和广度。HNSW提供极致的查询速度，IVF-PQ提供极致的存储效率。**根据数据规模选择**——百万级用HNSW，亿级用IVF-PQ。

**第三级：重排序（Reranker）** ——决定了“排得准”的精度。Cross-Encoder虽然慢，但只对少量候选操作，**用200ms的延迟换取40–60%的精度提升**，是RAG流水线中性价比最高的环节。

而**HyDE**则是在检索之前增加了一道“理解意图”的工序——**让LLM先“想”后“搜”** ，从“匹配关键词”升级到“理解意图”。

正如一位工程师所说：**“你的检索器可以在Top-10上统计优秀，却仍然把错误的Top-3交给LLM。”**RAG工程的核心，就是确保**最相关的文档出现在LLM看到的前几个位置**。