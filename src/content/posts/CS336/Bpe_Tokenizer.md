---
title: BPE(Byte-Pair Encoding) Tokenizer —— 从零构建BPE分词器
published: 2026-03-02
description: 系统讲解字节级BPE（Byte-Pair Encoding）分词器的完整实现原理与代码。从词级分词与字符级分词的困境出发，剖析GPT-2风格字节级BPE的优势；详细拆解预分词阶段GPT-2正则表达式含义；深入推导BPE训练的核心数据结构与贪心合并循环的频率更新逻辑；详解编码时按合并优先级应用规则的确定性流程；最后在TinyStories数据集上训练出18,017词汇量的分词器并生成.dat文件供模型训练使用。
cover: "/assets/images/posts/bpe_tokenizer.png"
coverInContent: false
tags: [BPE, Tokenizer, 字节级BPE, 预分词, 子词分词, LLM基础]
category: CS336
draft: false
---

# BPE(Byte-Pair Encoding) Tokenizer —— 从零构建BPE分词器

## 引言

在大型语言模型（LLM）的整个技术栈中，**分词器**（Tokenizer）常常是被最多人忽视、却又最容易引发各种“诡异问题”的环节。

Andrej Karpathy在一次著名的讲座中指出：**LLM中大量的奇怪行为和问题都可以追溯到分词器**——比如模型为何无法正确处理简单的字符串反转任务，为何在非英语语言上性能显著下降，为何在看到特定字符串时会“宕机” —— 这些问题的根源往往不在模型架构，而在于文本进入模型之前的“第一公里” —— **如何将原始文本转化为模型可以理解的数字序列**。

本文将从零开始逐行实现一个完整的 **字节级BPE（Byte-Pair Encoding）分词器**。深入理解BPE算法的每一个设计决策，剖析GPT-2风格预分词策略的微妙之处，并最终在TinyStories数据集上训练出一个可用的分词器。

本文配套的代码实现参考了Stanford CS336课程作业的设计思路，所有核心逻辑均从零编写，不依赖任何第三方分词库。

---

## 一、从词级分词到子词分词

在探讨BPE的具体实现之前，我们有必要先理解一个问题：**为什么大语言模型不直接用空格分词？**

### 1.1 词级分词（Word-level Tokenization）的困境

最直观的分词方式是按**空格和标点**将文本切分为单词。对于英语这样的语言，这种方法看似自然，但存在两个致命缺陷：

- **词表爆炸（Vocabulary Explosion）** —— 自然语言中的词汇量极其庞大。英语约有50万个常用单词，加上各种专有名词、技术术语、复合词，词表大小可以轻易**突破百万级别**。而现代LLM的词表通常被限制在**10万以内**——词表过大会导致**嵌入层（Embedding Layer）的参数爆炸**，同时使得**softmax输出层的计算**变得不可承受。
- **未登录词（OOV, Out-of-Vocabulary）问题** —— 无论词表多大，总会有训练语料中未出现过的词。对于这些词，词级分词器只能将其**映射为一个统一的`<UNK>`（Unknown）标记**，这意味着模型**彻底丢失了该词的任何信息**。

### 1.2 字符级分词（Character-level Tokenization）的局限

一个极端是直接**用单个字符作为token**。这彻底解决了OOV问题，但带来了新的麻烦：**序列长度急剧膨胀**。一个英文单词平均5-6个字符，这意味着模型需要处理5-6倍长度的序列，而**Transformer的注意力机制复杂度是$O(n^2)$的**——序列每增长一倍，计算量增长四倍。

### 1.3 子词级分词（Byte-Pair Encoding）的折中

**BPE（Byte-Pair Encoding）** 提供了一条中间道路。它**不把单词当作不可分割的最小单元，也不退回到单个字符**，而是学习出一套**子词（subword）** 的词汇表。

BPE的核心思想极其简洁：**从最细的粒度（单个字节或字符）出发，反复合并数据中出现频率最高的相邻单元对，直到词汇表达到预定大小**。

这种做法的妙处在于：
- **常见词**（如"the"、"and"）会作为**完整token**出现在词表中，**编码效率高**；
- **罕见词**会被拆分为**若干已知子词的组合**（如"tokenization" → "token" + "ization"），不会沦为`<UNK>`；
- **词表大小**可以精确控制，在编码效率和模型复杂度之间取得平衡。

BPE最初是1994年提出的一种数据压缩算法，2015年被Sennrich等人引入神经机器翻译，随后被GPT-2采用并推广为LLM分词的事实标准。今天，GPT、Llama、Mistral等所有主流LLM都在使用BPE或其变体来训练分词器。

---

## 二、字节级BPE

### 2.1 “字节级”的含义

传统BPE在**字符**级别操作——**词汇表初始化为训练语料中出现的所有字符**。但这种方法有一个隐患：如果测试时遇到训练语料中从未出现过的Unicode字符（比如一个罕见的表情符号），它依然会被映射为`<UNK>`。

GPT-2引入了一个巧妙的改进：**不在字符级别操作，而在UTF-8字节级别操作**。UTF-8编码中，任何字符（包括所有Unicode字符）都可以表示为**一个或多个字节的序列**。字节只有**256**种可能 —— 这恰好构成了BPE的**初始词汇表**：

```python
vocab = {i: bytes([i]) for i in range(256)}  # 256个基础字节
```

这意味着**任何文本**——无论是英语、中文、阿拉伯语，还是emoji、特殊符号——都可以**被无损地表示为这256个基础字节的组合**。**OOV问题**在字节级BPE中从根本上**不存在**了。

当然，直接使用字节序列会有代价：一个中文字符在UTF-8中占3个字节，如果完全不做合并，序列长度会膨胀3倍。而**BPE的合并**过程正是要解决这个问题——通过**迭代合并高频字节对**，逐步将**常见的字节序列**（如某个中文字符的3个字节、常见英文词缀等）压缩为**单个token**。

### 2.2 基础词汇表初始化

在`train_bpe`函数中，基础词汇表的初始化正是从256个字节开始的：

```python
vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
cur_id: int = 256  # 新token的id从256开始分配
```

**`vocab`是一个从token ID到字节序列的映射**。ID 0-255 分别对应**单个字节`b'\x00'`到`b'\xff'`**。所有后续通过**合并**产生的新token，其ID将从256开始依次**递增**。

---

## 三、特殊token的处理

在初始化基础词汇表之后，BPE训练需要做的第一件事是将**特殊token**（special tokens）加入词汇表。

### 3.1 特殊token的含义

特殊token是**永远不会被拆分成多个子词、始终被视为一个整体**的标记。它们通常用于标记文本中的特殊结构：

- `<|endoftext|>`：GPT系列使用的**文档结束标记**
- `[CLS]` / `[SEP]`：BERT使用的**分类标记和分隔标记**
- `<s>` / `</s>`：某些模型使用的句子**开始/结束标记**

这些token在训练和推理中具有特殊的语义含义，必须保证它们**在分词过程中保持完整**，不能被BPE的合并规则拆散。

### 3.2 特殊token的正确处理

```python
for spe_t in special_tokens:
    if len(vocab) >= vocab_size:
        break
    spe_b = spe_t.encode("utf-8")
    if spe_b not in cur_bytes:
        vocab[cur_id] = spe_b
        cur_bytes.add(spe_b)
        cur_id += 1
```

这段逻辑的关键点在于：
1. 将特殊token字符串编码为**UTF-8字节序列**；
2. 检查该字节序列是否已经存在于词汇表中（避免重复）；
3. 如果不存在，**将其作为一个整体加入词汇表，占用一个独立的token ID**。

特殊token的引入是**在BPE训练开始之前**完成的 —— 它们**不参与**后续的合并过程，也不会被拆散。

---

## 四、预分词（Pre-tokenization）

### 4.1 预分词的作用

BPE的核心操作是“**合并相邻的token对**”。但这里有一个微妙的问题：**我们是否允许跨单词边界进行合并？**

如果允许跨单词合并，那么"hello world"中的"lo"（来自"hello"的末尾）和" wo"（来自" world"的开头）可能会被合并，产生**跨越单词边界的token** —— 这显然不符合我们对语言结构的直觉。

**预分词（Pre-tokenization）** 的作用正是在BPE合并之前，**将文本切分成若干边界明确的片段**，确保BPE的合并操作**不会跨越这些边界**。

GPT-2引入的预分词策略已经成为后续模型的事实标准。其核心是一个精心设计的**正则表达式**。

### 4.2 GPT-2预分词正则表达式解析

GPT-2的预分词正则表达式如下：

```python
PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

这个表达式由多个**子模式**通过`|`（或）连接而成，匹配时按**从左到右的顺序**尝试。下面逐一拆解每个子模式的设计意图：

**子模式1：`'(?:[sdmt]|ll|ve|re)`**
- 匹配英语中常见的缩略形式：`'s`、`'d`、`'m`、`'t`、`'ll`、`'ve`、`'re`。例如："they're" → 被拆分为 `"they"` 和 `"'re"`；"don't" → 被拆分为 `"don"` 和 `"'t"`
- 将缩略形式单独截出来的原因：如果让它们和前面的词连在一起，BPE可能会把它们**合并成一个token**（如"they're"整体），但这会**降低模型的泛化能力** —— 模型如果能分别学到"they"和"'re"的独立表示，就能更好地理解"we're"、"you're"等结构。

**子模式2：`?\p{L}+`**
- `\p{L}`是Unicode属性，匹配**任何语言的字母字符**（拉丁字母、汉字、阿拉伯字母等）。前面的`?`表示**可选的一个空格**。
- 这意味着：`"hello"` → 匹配为 `"hello"`；`" hello"` → 匹配为 `" hello"`（**保留前导空格**）；`"世界"` → 匹配为 `"世界"`（中文汉字同样被`\p{L}`匹配）

**子模式3：`?\p{N}+`**
- 类似地，`\p{N}`匹配**任何数字字符**（不限于0-9，还包括全角数字等）。

**子模式4：`?[^\s\p{L}\p{N}]+`**
- 这是一个**补集匹配**：匹配既不是空白符、又不是字母、又不是数字的字符。
- 这包括：标点符号（`.`、`,`、`!`、`?`等）；emoji表情（`😊`、`🎉`等）；特殊符号（`@`、`#`、`$`等）

**子模式5：`\s+(?!\S)`**
- 匹配**行尾的空白符**（空格、换行等）。`(?!\S)`是一个否定前瞻断言，确保后面不跟非空白字符——即只匹配**末尾**的空白。

**子模式6：`\s+`**
- 匹配**普通的空白符序列**，作为最后的兜底。

> [!note] 空格要保留在token前面的原因
> 
> `?\p{L}+`等模式中的**空格是放在字母前面的**（`?`在`\p{L}+`之前）。这意味着匹配结果中，空格是**作为token的前缀**被保留的，如`" hello"`而非`"hello "`。
> 
> 这个设计有深刻的用意：**在GPT-2的BPE词表中，空格是有意义的**——"apple"和" apple"会被视为不同的token（后者在实现中通常表示为`Ġapple`，`Ġ`代表空格）。
> 
> 这种设计使得模型能够**区分"词首"和"词中"的同一单词**，有助于学习**词边界信息**。

### 4.3 预分词的代码实现

在`train_bpe`中，预分词分为两步：

**第一步：按特殊token分割文本**

```python
text_chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
```

这**确保特殊token被单独分离出来**，不会与普通文本混在一起参与BPE训练。

**第二步：对每个chunk应用GPT-2预分词正则**

```python
token_freqs = defaultdict(int)
for chunk in text_chunks:
    for word in regex.findall(PRETOKENIZER_PATTERN, chunk):
        word_bytes = word.encode("utf-8")
        bytes_list = [bytes([b]) for b in word_bytes]
        token_freqs[tuple(bytes_list)] += 1
```

每个匹配到的“词”（即预分词片段）被转换为**单字节的元组**，并**记录其频率**。`token_freqs`的**键是`tuple[bytes]`**（如`(b'h', b'e', b'l', b'l', b'o')`），**值是这个词在语料中出现的总次数**。

**关键设计决策**：`token_freqs`存储的是**每个预分词片段内部**的字节序列，不同片段之间的字节**不会被统计为相邻对**。这确保了BPE的合并永远不会跨越预分词边界。

---

## 五、BPE训练核心

### 5.1 训练算法概览

BPE训练的本质是一个**贪心压缩过程**：每次选择当前语料中**出现频率最高的相邻token对进行合并**，将合并后的**新token加入词汇表**，然后**重复**这一过程，直到**词汇表达到预定大小或没有更多可合并的对**。

其高层伪代码如下：
1. 初始化词汇表为256个**基础字节**
2. 对语料进行**预分词**，统计每个预分词片段的**频率**
3. 统计**所有相邻token对的频率**
4. while 词汇表大小 < 目标大小:
    a. 选择**频率最高的相邻token对**
    b. 将这对**合并**为一个新token，加入词汇表
    c. **更新语料中所有出现该对的位置**
    d. **更新相邻token对频率表**

### 5.2 数据结构设计

在深入训练循环之前，先理解代码中使用的关键数据结构：

- **`token_freqs`** : `Dict[Tuple[bytes, ...], int]` —— 键是**预分词片段的字节序列**（元组形式），值是**该片段**在语料中出现的**总次数**。
- **`pair_freqs`** : `Dict[Tuple[bytes, bytes], int]` —— 键是**相邻的token对**（两个字节序列），值是**该对**在所有片段中出现的**总次数**。
- **`token_seq_to_pairs`** : `Dict[Tuple[bytes, ...], Set[Tuple[bytes, bytes]]]` —— 缓存每个片段中包含的所有**相邻token对**，用于加速“**哪些片段受本次合并影响**”的查找。

`token_seq_to_pairs`是一个关键的**性能优化**。如果没有这个缓存，**每次合并后都需要遍历所有片段并检查其中是否包含`best_pair`** —— 复杂度为$O(\text{词表大小} \times \text{平均片段长度})$。有了这个缓存，便可以**在$O(1)$时间内判断一个片段是否受某次合并影响**。

### 5.3 初始化频率表

在开始迭代合并之前，我们需要**统计所有相邻token对的初始频率**：

```python
pair_freqs = defaultdict(int)
for token_seq, freq in token_freqs.items():
    for i in range(len(token_seq) - 1):
        pair = (token_seq[i], token_seq[i+1])
        pair_freqs[pair] += freq
```

对于每个**预分词片段**（如`(b'h', b'e', b'l', b'l', b'o')`），我们遍历其中**所有相邻的token对**`(token_seq[i], token_seq[i+1])`，**将该对的频率加上该片段的出现频率**。

### 5.4 主训练循环

```python
while len(vocab) < vocab_size:
    if not pair_freqs:
        break
    
    # 1. 选择频率最高的合并对
    max_freq = max(pair_freqs.values())
    most_freq_pairs = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
    best_pair = max(most_freq_pairs)  # 频率相同时选择字典序最大的
    
    # 2. 创建新token并更新词汇表
    new_token = best_pair[0] + best_pair[1]
    vocab[cur_id] = new_token
    merges.append(best_pair)
    cur_id += 1
    
    # 3. 找出受本次合并影响的片段
    affected_tokens_freqs = [
        (token_seq, freq)
        for token_seq, freq in token_freqs.items() 
        if best_pair in token_seq_to_pairs[token_seq]
    ]
    
    # 4. 更新受影响的片段
    for token_seq, freq in affected_tokens_freqs:
        # 4a. 从pair_freqs中移除旧token对的贡献
        for i in range(len(token_seq) - 1):
            old_pair = (token_seq[i], token_seq[i+1])
            pair_freqs[old_pair] -= freq
            if pair_freqs[old_pair] <= 0:
                del pair_freqs[old_pair]
        
        # 4b. 构建新的token序列（将所有best_pair替换为new_token）
        new_token_seq = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
                new_token_seq.append(new_token)
                i += 2
            else:
                new_token_seq.append(token_seq[i])
                i += 1
        new_token_seq = tuple(new_token_seq)
        
        # 4c. 将新token序列产生的对频率添加到pair_freqs
        for i in range(len(new_token_seq) - 1):
            new_pair = (new_token_seq[i], new_token_seq[i+1])
            pair_freqs[new_pair] += freq
        
        # 4d. 更新token_freqs和token_seq_to_pairs
        del token_freqs[token_seq]
        token_freqs[new_token_seq] = freq
        del token_seq_to_pairs[token_seq]
        new_pairs_set = set()
        for i in range(len(new_token_seq) - 1):
            new_pairs_set.add((new_token_seq[i], new_token_seq[i+1]))
        token_seq_to_pairs[new_token_seq] = new_pairs_set
```

**步骤1：选择最高频的相邻对**

当多个token对具有**相同的最高频率**时，代码选择**字典序最大**的那个。这是一种**确定性的tie-breaking策略**。不同的tie-breaking策略（如字典序最小、随机选择）会产生略微不同的合并结果，但在大规模语料上**差异通常很小**。

**步骤2：创建新token**

新token就是两个被合并token的**字节序列拼接**。例如，如果`best_pair = (b'h', b'e')`，则`new_token = b'he'`

**步骤3-4：更新频率表（核心难点）**

这是BPE训练中**最微妙的部分**。将`best_pair`合并为`new_token`后，所有包含`best_pair`的片段都发生了变化，进而影响到**哪些相邻token对仍然存在**（旧的被移除，新的被加入）；**这些相邻token对的频率**。

采用**先减后加**的策略：
1. **先移除旧对的贡献**：遍历**受影响片段**中的所有**相邻对**，从`pair_freqs`中**减去**该片段的频率；
2. **再添加新对的贡献**：构建新序列后，遍历**新序列**中的所有**相邻对**，将频率**加到**`pair_freqs`中。

这种做法的正确性依赖于一个关键事实：**每次合并只影响包含`best_pair`的片段**。不包含`best_pair`的片段，其内部的所有相邻对都不变。

### 5.5 训练终止条件

训练在以下两种情况下终止：
1. **词汇表达到目标大小**（`len(vocab) >= vocab_size`）；
2. **没有更多可合并的对**（`pair_freqs`为空）。

在真实训练中，第二种情况很少发生——只要语料足够大，总会有频率至少为1的相邻对存在。

---

## 六、分词器的编码（Encoding）

训练完成后，得到了两个产物：
- **`vocab`** : **从token ID到字节序列的映射**
- **`merges`** : **按训练顺序排列的合并规则列表**

现在需要用它们来**编码新的文本 —— 将任意输入字符串转换为token ID序列。**

### 6.1 编码的核心挑战

编码的核心挑战是**如何确定性地应用合并规则？**

给定一个字节序列，可能存在**多种应用合并规则的方式**。例如，假设合并规则包含`(b'a', b'b') -> b'ab'`和`(b'b', b'c') -> b'bc'`，对于字节序列`b'abc'`，是先合并`ab`还是先合并`bc`？

答案是：**按照合并规则被学习的顺序（优先级）来应用**。**最早**被学习的合并规则具有**最高优先级**。在编码时，每一步都选择**当前序列中存在的、优先级最高的合并规则**来执行。

### 6.2 Tokenizer类的初始化

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merge_priorities: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.byte_to_id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}
```

`merge_priorities`是一个**从token对到优先级索引的映射 —— 索引越小，优先级越高**（越早被学习）。这使得编码时可以**在$O(1)$时间内查询任意token对的合并优先级**。

### 6.3 BPE合并的核心算法

```python
def _apply_bpe_merge(self, bytes_seq: bytes) -> List[bytes]:
    cur_bytes: List[bytes] = [bytes([b]) for b in bytes_seq]
    
    while len(cur_bytes) > 1:
        # 1. 找出当前序列中所有可合并的token对
        exist_pairs: Set[Tuple[bytes, bytes]] = set()
        for i in range(len(cur_bytes) - 1):
            pair = (cur_bytes[i], cur_bytes[i+1])
            if pair in self.merge_priorities:
                exist_pairs.add(pair)
        if not exist_pairs:
            break
        
        # 2. 选择优先级最高的token对
        best_pair = min(exist_pairs, key=lambda pair: self.merge_priorities[pair])
        
        # 3. 执行合并
        new_bytes: List[bytes] = []
        i = 0
        while i < len(cur_bytes):
            if i < len(cur_bytes) - 1 and (cur_bytes[i], cur_bytes[i+1]) == best_pair:
                new_bytes.append(cur_bytes[i] + cur_bytes[i+1])
                i += 2
            else:
                new_bytes.append(cur_bytes[i])
                i += 1
        
        cur_bytes = new_bytes
    
    return cur_bytes
```

这个算法的核心逻辑是**贪心应用最高优先级的合并**：
1. 扫描当前字节序列，找出所有**同时存在于合并规则中的相邻token对**；
2. 从中选出**优先级最高**（`merge_priorities`值最小）的一对；
3. 在序列中**将所有出现**的该对**合并**为一个新token；
4. **重复**上述过程，直到没有更多可合并的对。

**关键点：每次只合并一个token对（优先级最高的那个），而不是一次性合并所有可合并的对**。这是因为合并一个对之后，可能会产生**新的可合并对**（如合并`ab`后产生`abc`），这些新对的优先级可能**高于**原本第二优先级的对。

### 6.4 完整编码流程

```python
def encode(self, text: str) -> List[int]:
    if not text:
        return []
    
    # 1. 构建特殊token的正则模式
    sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
    special_token_pattern = '|'.join(map(regex.escape, sorted_special_tokens))
    
    # 2. 按特殊token分割文本
    if self.special_tokens:
        chunks = regex.split(f'({special_token_pattern})', text)
    else:
        chunks = [text]
    
    # 3. 处理每个chunk
    token_ids: List[int] = []
    for chunk in chunks:
        if not chunk:
            continue
        
        if chunk in self.special_tokens:
            # 特殊token直接编码
            special_token_bytes = chunk.encode("utf-8")
            special_token_id = self.byte_to_id[special_token_bytes]
            token_ids.append(special_token_id)
        else:
            # 普通文本：预分词 -> BPE合并 -> 查表
            for token in regex.findall(PAT, chunk):
                bpe_token_bytes = self._apply_bpe_merge(token.encode("utf-8"))
                for token_bytes in bpe_token_bytes:
                    token_id = self.byte_to_id[token_bytes]
                    token_ids.append(token_id)
    
    return token_ids
```

编码流程可以概括为四个层次：

1. **特殊token层**：**特殊token被整体识别**，直接映射到对应的token ID，不经过任何拆分或合并；
2. **预分词层**：非特殊文本被GPT-2**正则表达式切分为若干“词”片段**；
3. **BPE合并层**：每个片段内部的**字节序列**通过`_apply_bpe_merge`进行**合并**；
4. **查表层**：合并后的每个token字节序列通过`byte_to_id`**查找对应的token ID**。

--- 

## 七、分词器的解码（Decoding）

解码是编码的逆过程，但简单得多：

```python
def decode(self, token_ids: List[int]) -> str:
    bytes_seq = b''.join(self.vocab[id] for id in token_ids)
    return bytes_seq.decode("utf-8", errors="replace")
```

解码不需要BPE合并规则 —— **只需要将每个token ID映射回对应的字节序列，将所有字节序列拼接，然后用UTF-8解码为字符串**。

`errors="replace"`参数确保即使遇到无效的UTF-8字节序列（鲁棒性考虑），也不会抛出异常，而是用`�`（U+FFFD）**替换无效字节**。

---

## 八、在TinyStories上训练BPE

### 8.1 训练脚本

```python
if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-valid.txt", 
        20000, 
        special_tokens
    )
    print(f"训练完成，最终词汇表大小: {len(vocab)}")  # 18017
    print(f"生成了 {len(merges)} 次合并。")  # 17760
```

在TinyStories验证集（约4.5MB文本）上训练，目标词汇表大小为20,000：

- **初始词汇表**：256个基础字节 + 1个特殊token = 257
- **实际合并次数**：17,760次
- **最终词汇表大小**：18,017

在训练过程中，`pair_freqs`在某些迭代中可能为空（没有更多可合并的对），或者某些特殊token的加入占用了词汇表空间但并未增加合并次数。

### 8.2 数据预处理：生成`.dat`文件

训练好的分词器可以用于**将文本数据集转换为token ID序列**，以便后续的模型训练：

```python
def text_to_dat(text_file, dat_file, tokenizer):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    token_array = np.array(tokens, dtype=np.uint16)
    
    fp = np.memmap(dat_file, dtype=np.uint16, mode='w+', shape=token_array.shape)
    fp[:] = token_array[:]
    fp.flush()
```

这里使用`np.memmap`将token序列直接写入磁盘，便于后续高效加载——**模型训练时可以直接内存映射这个文件，无需一次性将全部数据加载到内存中**。

### 8.3 词表大小的影响

不同`vocab_size`的选择对分词效果有显著影响：

| 词汇表大小 | 优点 | 缺点 |
|-----------|------|------|
| 较小（如1K-5K） | **嵌入层参数少，模型更紧凑** | **序列较长，罕见词被拆分为过多子词** |
| 较大（如20K-100K） | **序列更短，常见词保留完整** | **嵌入层参数多，训练和推理开销大** |

在实际应用中，词表大小需要在**编码效率**（序列长度）和**模型复杂度**（词表大小）之间权衡。GPT-2使用50,257的词表，GPT-4进一步扩展，Llama系列使用32K，而DeepSeek-V3则扩展到了128K词汇表以优化多语言压缩效率。

> [!note] 总结
> 本文从零实现了完整的字节级BPE分词器，涵盖了从训练到编码解码的全流程。深入理解了以下几个关键设计决策：
> 
> 1. **字节级BPE**：以256个基础字节为起点，从根本上消除了OOV问题；
> 2. **预分词**：GPT-2风格的正则表达式确保合并不会跨越词边界，同时保留空格信息；
> 3. **贪心合并**：每次选择频率最高的相邻token对进行合并，这是一种简单而有效的压缩策略；
> 4. **优先级编码**：编码时按照合并被学习的顺序应用规则，保证确定性。
> 
> BPE分词器虽然只是LLM技术栈中很小的一环，但它决定了模型“看到”的输入是什么样的——**分词器的质量直接影响模型的泛化能力和鲁棒性**。理解分词器的工作原理，是深入理解LLM行为的基础。