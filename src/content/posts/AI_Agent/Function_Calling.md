---
title: Function Calling —— 工具调用
published: 2026-03-06
description: 系统讲解AI Agent工具调用的核心技术原理与工程实践：从JSON Schema的结构化约束机制出发，剖析Structured Outputs如何动态屏蔽非法token以保证输出匹配Schema；分析并行工具调用的延迟优势；深入解读tools与tool_choice的设计哲学；揭示工具描述撰写的核心技巧。
cover: "/assets/images/posts/function_calling.png"
coverInContent: false
tags: [AI Agent, 工具调用, Function Calling, 并行调用, MCP]
category: AI_Agent
draft: false
---

# Function Calling —— 工具调用

## 引言：让LLM从“说话”到“做事”

在前两篇文章中，我们分别拆解了AI Agent的四大核心模块和三种主流范式。无论架构如何设计、范式如何选择，有一个组件始终处于Agent能力扩展的最前沿——**工具调用（Tool Calling / Function Calling）** 。

如果说LLM是Agent的“大脑”，那么工具调用就是让大脑指挥“四肢”的**神经系统**。没有工具调用，Agent只能是一个高级的对话机器人——它能说会道，却无法真正改变世界。有了工具调用，Agent可以查询数据库、发送邮件、调用API、执行代码——**从“会说话”进化到“会做事”** 。

2023年6月，OpenAI在GPT-4和GPT-3.5 Turbo中首次引入了函数调用（Function Calling）功能。此后，这一能力迅速成为所有主流LLM厂商的标配。然而，**工具调用远非“把函数名和参数传给模型”那么简单**。从JSON Schema的结构化约束，到并行调用的延迟优化，再到工具描述的文字艺术——每一个细节都深刻影响着Agent系统的可靠性、效率和成本。

本文将系统讲解工具调用的核心技术原理与工程实践，涵盖：结构化输出的JSON Schema约束机制、并行工具调用的延迟分析、OpenAI `tools`与`tool_choice`参数的设计哲学，以及最容易被低估却最关键的一环——**工具描述撰写技巧**。

---

## 一、结构化输出——JSON Schema的约束力量

### 1.1 从“写JSON”到“返回结构化调用”

在函数调用出现之前，让LLM输出结构化数据的唯一方式是**提示工程**——在prompt中要求模型“以JSON格式返回”。这种方式的问题显而易见：模型可能会忘记加引号、漏掉字段、输出格式不统一。本质上，这是**把格式约束的责任完全推给了模型的“自觉”**。

函数调用的革命性之处在于：**它不再要求模型“生成一段JSON文本”，而是让模型直接返回一个结构化的函数调用对象**。

你通过JSON Schema描述函数，模型会返回一个带有函数名和JSON参数的结构化调用。你的应用解析这个调用，执行对应的函数，然后（可选）将结果发回给模型生成最终回答。

```python
# 定义工具（使用JSON Schema描述参数）
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定位置的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，如'北京'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位"
                }
            },
            "required": ["location"]
        }
    }
}]
```

### 1.2 JSON Schema的数学本质

从数学角度看，JSON Schema是对模型输出空间的一个**约束**。在没有约束的情况下，模型的输出空间是整个token序列空间 $\mathcal{V}^*$（其中 $\mathcal{V}$ 是词表）。JSON Schema将这一空间限制为一个**结构化的子集**：

$$\mathcal{S}_{\text{schema}} = \{ x \in \mathcal{V}^* \mid \text{validate}(x, \text{schema}) = \text{true} \}$$

模型在生成时，需要在满足格式约束的前提下最大化条件概率：

$$x^* = \arg\max_{x \in \mathcal{S}_{\text{schema}}} P(x \mid \text{prompt})$$

这就是为什么函数调用比纯提示工程更可靠——**模型不是在“努力”输出JSON，而是在一个被约束的输出空间中进行概率采样**。

### 1.3 Structured Outputs：从“有效JSON”到“匹配Schema”

需要特别注意的是，**普通的JSON模式（JSON mode）只保证输出是有效的JSON，并不保证输出匹配你指定的具体Schema**。

OpenAI在2024年推出了**Structured Outputs**（结构化输出）功能。当开启Structured Outputs时，模型为函数调用生成的参数**保证匹配你提供的JSON Schema**。

```python
# 开启strict模式，确保参数严格匹配Schema
{
    "parameters": {
        "type": "object",
        "properties": {
            "attendee": {"type": "string"},
            "date": {"type": "string"},
            "time": {"type": "string"}
        },
        "required": ["attendee", "date", "time"],
        "additionalProperties": false  # 禁止额外字段
    },
    "strict": true  # 开启Strict模式
}
```

Structured Outputs的核心机制在数学上可以理解为：在解码过程中**动态地屏蔽**那些会导致输出不符合Schema的token。这相当于在每个解码步骤 $t$，将可行token集合从整个词表 $\mathcal{V}$ 缩小为：

$$\mathcal{V}_t^{(\text{valid})} = \{ v \in \mathcal{V} \mid \text{prefix}_{t-1} \oplus v \text{ 能扩展为某个符合Schema的完整输出} \}$$

然后模型只在 $\mathcal{V}_t^{(\text{valid})}$ 上进行概率采样：

$$x_t \sim P(\cdot \mid x_{<t}, \text{prompt}), \quad \text{且 } x_t \in \mathcal{V}_t^{(\text{valid})}$$

### 1.4 实践建议：用Pydantic做双重保障

即使使用了Structured Outputs，**在生产环境中仍然建议使用验证库（如Pydantic）对参数进行二次校验**。原因有二：

1. **模型可能出错**：虽然概率极低，但Structured Outputs并非100%可靠
2. **防御性编程**：即使模型输出符合Schema，参数值本身可能不合法（如日期格式错误）

```python
from pydantic import BaseModel, ValidationError

class WeatherParams(BaseModel):
    location: str
    unit: str  # 可以在Pydantic中进一步校验

# 解析并验证模型返回的参数
try:
    params = WeatherParams(**json.loads(tool_call.function.arguments))
except ValidationError as e:
    # 处理参数验证失败
    pass
```

---

## 二、并行工具调用——延迟的艺术与权衡

### 2.1 核心概念：一次请求，多个调用

**并行工具调用（Parallel Tool Calling）** 是OpenAI等厂商提供的一项关键能力，它允许模型在**单次响应中返回多个工具调用**。

这意味着什么？假设一个用户问：“苹果、谷歌、微软今天的股价分别是多少？”在没有并行调用的情况下，模型可能需要三次独立的API往返——每次调用一个股票查询工具，串行等待结果。而有了并行调用，模型可以**一次性返回三个工具调用**，你的应用可以并发执行它们，然后一次性将结果返回给模型。

### 2.2 延迟对比：数学分析

设单个工具调用的平均执行延迟为 $L$（包括网络往返、API处理等）。对于 $n$ 个独立的工具调用：

**串行执行的总延迟**为：

$$T_{\text{serial}} = n \cdot L$$

**并行执行的总延迟**为：

$$T_{\text{parallel}} = \max_{i} L_i + \text{overhead}$$

如果所有工具延迟相近（$L_i \approx L$），则：

$$T_{\text{parallel}} \approx L + \text{overhead}$$

延迟降低的倍数为：

$$\text{speedup} = \frac{T_{\text{serial}}}{T_{\text{parallel}}} \approx n$$

**实际数据更具说服力**：

- 如果每个工具调用耗时500ms，5个串行调用需要2.5秒，而并行调用只需要约500ms
- 有实测数据显示，串行执行（4-12个工具调用）每轮需要12-24秒，而并行执行仅需1-2秒，**延迟降低约4倍**
- 行业基准测试表明，并行工具调用**可将延迟降低60%到70%**

### 2.3 并行调用的现实挑战

然而，并行工具调用并非银弹。实践中存在几个重要挑战：

**模型行为不一致**：不同模型对并行调用的支持程度差异显著。有开发者报告，GPT-4.1在无需额外提示的情况下，约80%的场景会自动进行并行工具调用；而GPT-5即便加了额外提示，在相同场景下只有约25%的概率会并行调用。

**流式输出的延迟感知**：当模型只选择一个工具时，工具调用事件几乎立即到达；但当模型选择多个工具时，会出现**数秒的停顿**，然后所有工具调用事件同时到达。这并非真正的“流式”，而是批处理式的爆发。

**特定API的限制**：在Responses API中，Web Search和File Search等内置工具可能会阻止自定义函数被并行调用。

### 2.4 何时启用并行调用

**适合并行调用的场景**：
- 工具调用之间**相互独立**，无数据依赖
- 需要从多个数据源**并行获取信息**
- 用户查询涉及**多个独立实体**（如多个公司的股价）

**不适合并行调用的场景**：
- 工具调用之间存在**顺序依赖**（B需要A的结果作为输入）
- 需要对工具调用结果进行**严格排序**的场景

在实践中，可以通过设置 `parallel_tool_calls` 参数来控制是否启用此功能。

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    parallel_tool_calls=True  # 启用并行调用
)
```

---

## 三、`tools`与`tool_choice`——设计哲学与精细控制

### 3.1 从`functions`到`tools`：一次重要的API演进

在OpenAI的早期实现中，函数调用通过 `functions` 和 `function_call` 参数实现。这些参数如今已被**弃用**，所有新代码都应改用 `tools` 和 `tool_choice` 的新风格。

这一演进背后的设计哲学是什么？

**`tools` 是一个更通用的抽象**。它不仅支持自定义函数（function），还支持内置工具（如 `file_search`、`code_interpreter`、`web_search` 等）。将函数调用纳入 `tools` 的统一框架下，为未来的扩展预留了空间。

正如OpenAI官方文档所述，工具调用的完整流程包含五个高层步骤：
1. 向模型发起请求，附带它可以调用的工具列表
2. 模型返回一个工具调用
3. 你的应用执行该工具
4. 将执行结果返回给模型
5. 模型生成最终回答

### 3.2 `tool_choice`的三种模式

`tool_choice` 参数是控制模型工具调用行为的**核心开关**。它提供了三种模式：

**`"auto"`（默认）** ：模型自行判断是否调用工具、调用哪个工具。这是最灵活的模式，适用于通用对话场景。

**`"none"`** ：强制模型不调用任何工具，即使工具定义可用，模型也仅生成纯文本回复。适用于知识问答、创意内容生成等场景。

**`"required"`** ：强制模型必须调用至少一个工具。适用于那些确定需要工具参与的流程化任务。

此外，还可以**指定特定的工具名称**，强制模型调用某个具体的工具：

```python
tool_choice={
    "type": "function",
    "function": {"name": "get_weather"}
}
```

### 3.3 设计哲学：三种模式的数学含义

从决策论的角度看，这三种模式对应着不同的**决策策略**：

- **`"auto"`** ：模型在行动空间 $\mathcal{A} = \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{tools}}$ 上自由选择，最大化 $P(a \mid \text{context})$
- **`"none"`** ：将行动空间限制为 $\mathcal{A} = \mathcal{A}_{\text{text}}$，即只允许文本回复
- **`"required"`** ：将行动空间限制为 $\mathcal{A} = \mathcal{A}_{\text{tools}}$，且至少选择一个工具

`"required"` 模式的意义在于**将“是否使用工具”这个决策从模型中剥离出来**，交给开发者控制。这在某些场景下至关重要——例如，当一个工作流确定需要调用某个API时，你不希望模型“自作主张”地跳过它。

> **💡 实践建议**：除非有特殊需求，否则保持 `tool_choice: "auto"` 是最安全的选择。`"required"` 适用于需要确保工具被调用的场景，`"none"` 适用于需要纯文本回复的场景。

### 3.4 多工具的“发现-选择”机制

当向模型提供多个工具时，模型需要完成两个任务：**发现**（从工具列表中找到合适的工具）和**选择**（决定使用哪个工具）。这个过程的数学本质是：

$$\text{tool}^* = \arg\max_{t \in \mathcal{T}} P(t \mid \text{context}, \text{descriptions}(\mathcal{T}))$$

其中 $\mathcal{T}$ 是所有可用工具的集合，`descriptions` 是每个工具的描述文本。

这意味着：**工具选择的准确率直接取决于描述文本的质量**。这引出了我们下一个——也是最容易被低估的——话题。

---

## 四、工具描述撰写——最被低估的工程能力

### 4.1 一个真实的数据点

先看一组数据：**同样的GPT-4.1模型，同样的工具代码，模糊描述下工具调用准确率约60%，精心优化的描述下准确率超过90%**。

这30个百分点的差距来自哪里？**完全来自 `description` 字段的写法**。

**LLM选择工具时看不到你的函数实现——它只能看到你写的description。这段文字就是LLM理解工具的唯一窗口**。

工具描述写得好不好，**直接决定了Agent的“智商”**。再强的模型，遇到模糊的工具描述也会选错工具。写得好，工具调用准确率能从60%提升到95%以上。

### 4.2 常见的反面案例

```python
# ❌ 反面案例：三个工具的描述几乎一样
tools = [
    {
        "name": "get_order",
        "description": "获取订单信息",  # 太模糊了！
        "parameters": {"order_id": {"type": "string"}}
    },
    {
        "name": "search_orders",
        "description": "搜索订单",  # 和上面有什么区别？
        "parameters": {"query": {"type": "string"}}
    },
    {
        "name": "get_user_orders",
        "description": "获取用户订单",  # 和get_order有什么区别？
        "parameters": {"user_id": {"type": "string"}}
    }
]
```

当用户说“帮我查一下我的订单”时，LLM完全不知道该选哪个工具。

### 4.3 好工具描述的六要素

根据多位工程师的实战经验，一份高质量的工具描述应包含以下要素：

**① 一句话功能说明（最重要）**

LLM扫描工具列表时，首先看到的就是第一句话。公式是：**动词 + 操作对象 + 核心能力边界**。

```python
# ❌ 太模糊
"description": "处理文本"

# ❌ 太技术化（与调用决策无关）
"description": "基于Transformer的NLP管道"

# ✅ 清晰明确
"description": "总结用户提供的长文本，输出3条核心观点"
```

**② 适用场景（何时用）**

写清楚“什么情况下应该调用这个工具”。

```python
"description": """
获取当前股价和涨跌幅。
〖适用场景〗用户询问具体股票的实时价格，如'苹果现在多少钱'、'特斯拉今天涨了吗'
"""
```

**③ 不适用场景（何时不用）——最容易被忽略**

明确告诉模型**什么时候不应该调用这个工具**，可以大幅减少误用。

```python
"description": """
通过精确的订单号查询单个订单的详细信息。
〖适用场景〗用户提供了具体的订单号（如'ORDER-20240315-001'）
〖不适用场景〗用户只说'我的订单'但没有提供订单号；需要查询多个订单时
"""
```

**④ 参数格式和示例（不要让LLM猜）**

每个参数都应该有清晰的格式说明和示例值。

```python
"order_id": {
    "type": "string",
    "description": "订单号，格式为'ORDER-YYYYMMDD-XXX'，例如'ORDER-20240315-001'。如果用户没有提供订单号，不要猜测，应该向用户询问。",
    "pattern": "^ORDER-\\d{8}-\\d{3}$"
}
```

**⑤ 区分相似工具**

当有多个相似工具时，**必须明确区分它们的适用边界**。

**⑥ 为LLM写作，不是为人类写作**

工具描述首先服务于模型的“选择判断”，不是服务于人类炫技。避免使用过于专业或晦涩的术语。

### 4.4 一个完整的正面案例

```python
# ✅ 正面案例：清晰、精确、有边界
tools = [
    {
        "name": "get_order_by_id",
        "description": """
通过精确的订单号查询单个订单的详细信息，包括订单状态、商品列表、支付信息和物流追踪号。
〖适用场景〗用户提供了具体的订单号（如'ORDER-20240315-001'）
〖不适用场景〗用户只说'我的订单'但没有提供订单号；需要查询多个订单时
""",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，格式为'ORDER-YYYYMMDD-XXX'，例如'ORDER-20240315-001'。如果用户没有提供订单号，不要猜测，应该向用户询问。",
                    "pattern": "^ORDER-\\d{8}-\\d{3}$"
                }
            },
            "required": ["order_id"],
            "additionalProperties": false
        },
        "strict": True
    },
    {
        "name": "get_user_all_orders",
        "description": """
获取指定用户的所有订单列表，按时间倒序排列，最多返回50条。
〖适用场景〗用户想查看自己的全部订单历史，或说'我的所有订单'/'最近的订单'但没有提供订单号
〖不适用场景〗用户已经提供了具体订单号；需要搜索特定条件的订单时
""",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户ID，从当前会话的用户上下文中获取，不要向用户询问"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回订单数量，默认10，最大50",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": ["user_id"],
            "additionalProperties": false
        },
        "strict": True
    }
]
```

### 4.5 为什么描述质量如此重要——信息论视角

从信息论的角度看，工具描述的质量决定了**LLM在工具选择时的信息熵**。

设工具集合为 $\mathcal{T}$，描述文本为 $D$。LLM选择工具 $t$ 的决策基于：

$$P(t \mid \text{query}, D)$$

**好的描述** $D_{\text{good}}$ 使得后验概率高度集中——LLM能明确知道该选哪个工具：

$$H(T \mid \text{query}, D_{\text{good}}) \approx 0$$

**差的描述** $D_{\text{bad}}$ 使得后验概率分散——LLM在多个工具之间犹豫不决：

$$H(T \mid \text{query}, D_{\text{bad}}) \gg 0$$

这就是为什么精心撰写的描述能将准确率从60%提升到90%以上。

### 4.6 工具数量的最优实践

关于工具数量，**建议不超过20个**。当工具数量较多时，可以考虑：

- 将相似意图合并为**一个工具，用参数区分**（如用一个 `query_database` 工具，通过 `query_type` 参数区分不同查询）
- 如果必须分开，**确保每个工具的描述足够清晰地区分彼此**

---

## 五、工具调用的未来——MCP与Responses API

### 5.1 MCP：工具调用的标准化协议

**模型上下文协议（Model Context Protocol, MCP）** 正在成为AI Agent工具调用的标准化方案。

MCP的核心价值在于：**它标准化了工具发现、工具调用和结果返回的方式**，使得Agent可以动态地发现和调用来自不同提供者的工具，而无需为每个工具编写定制的集成代码。

截至2025年7月，已有超过4000个MCP服务器、覆盖40多个类别被部署。MCP正在从“可选协议”变成“事实标准”。

### 5.2 Responses API：新一代Agentic API

OpenAI在2025年3月推出了**Responses API**，这是新一代的Agentic API。与传统的Chat Completions API相比，Responses API的核心理念是 **“Agentic by default”** ——它原生支持多工具调用、多轮对话和不同类型的数据处理。

Responses API的一个重要特性是**服务端托管的工具**（如Code Interpreter、Web Search、File Search等），这些工具在OpenAI的服务端执行，无需在每次调用时都通过你的后端进行往返。

---

## 总结：从“能调用”到“会调用”

工具调用表面上看是一个简单的API功能——定义几个函数，传给模型，执行返回的调用。但在工程实践中，**从“能调用”到“会调用”之间，隔着一条巨大的鸿沟**。

这条鸿沟由四个层次的能力跨越：

1. **结构化约束**（JSON Schema + Structured Outputs）：确保模型输出的参数可靠、可解析
2. **性能优化**（并行调用）：将多个独立工具调用的延迟从 $n \cdot L$ 降低到 $\max(L_i)$
3. **行为控制**（`tool_choice`）：精确控制模型“何时调用”、“调用什么”
4. **描述工程**（高质量的工具描述）：将工具调用准确率从60%提升到95%以上

这四个层次中，**描述工程是最容易被低估、却投入产出比最高的一环**。再强的模型、再完善的Schema，遇到模糊的工具描述也会选错工具。正如一位工程师所说：**“LLM怎么用你的工具，80%取决于Schema写得好不好”** 。

在AI Agent从“玩具”走向“工具”的今天，掌握工具调用的工程化实践，是每一个Agent开发者必须跨越的门槛。