---
title: Agent Architecture —— LLM & Planning & Tool & Memory
published: 2026-03-02
description: 系统讲解AI Agent的核心架构——LLM（推理引擎）、规划（Planning）、工具（Tools）与记忆（Memory）四大模块的设计原理与协同机制。从感知-规划-行动闭环出发，剖析LLM作为“中央推理引擎”的多重角色、规划的数学建模与两种范式、工具调用的标准化接口与MCP协议演进，以及短期记忆与长期记忆的二分法设计。
cover: "/assets/images/posts/agent_architecture.png"
coverInContent: false
tags: [AI Agent, LLM, 规划, 工具调用, Function Calling, MCP, 记忆, RAG]
category: AI_Agent
draft: false
---

# Agent_Architecture —— LLM & Planning & Tool & Memory

## 引言：从“大脑”到“完整个体”

2025年被普遍认为是AI Agent的元年。从Auto-GPT到Devin，从MCP协议到多智能体协作，AI Agent已然成为当前最炽热的技术风口。然而，热度之下也伴随着混乱——很多初创项目把一个加了“工具调用”的prompt当作Agent系统，不少企业部署的所谓Agent最终被证明只是“自动填表机器人+LLM问答助手”的拼装体。

AI Agent不是prompt拼接游戏，也不是LLM的UI封装。它是一种**系统工程**。

如果说传统的大语言模型只是一个“大脑”——它接收输入、产生输出，那么一个真正的AI Agent则是 **“拥有大脑和四肢的完整个体”** 。它不仅会思考，还会为了目标去调用工具、执行代码、操作软件。

本文将系统讲解AI Agent的核心架构——**LLM（推理引擎）+ 规划（Planning）+ 工具（Tools）+ 记忆（Memory）** ——这一被deepresearch、Manus、Claude Code等现象级产品共同遵循的基础框架。

---

## 一、什么是AI Agent——感知-决策-行动的闭环

### 1.1 定义

Agent（智能体）的经典定义来自Russell & Norvig的《人工智能：一种现代方法》：**能够通过传感器感知环境并通过执行器对环境采取行动的任何事物**。

在LLM时代，这个定义被具体化为：**AI Agent是将基础模型与一个执行循环相结合的系统，能够观察环境、规划、调用工具、更新记忆并验证结果**。换句话说，Agent不再只是一个文本生成器，而是一个**将自然语言意图转化为现实世界操作的控制器**。

### 1.2 核心特征

一个真正的AI Agent具备以下核心特征：

- **自主性**：无需人类持续干预，可独立运作
- **感知能力**：通过传感器、API或文本输入等方式从环境中获取信息
- **推理与决策能力**：基于感知信息和内部知识进行逻辑分析和规划
- **执行能力**：通过执行器、API调用或文本输出等方式影响环境
- **目标导向性**：所有行动都围绕实现一个或多个特定目标展开

### 1.3 感知-规划-行动闭环

AI Agent的运作遵循一个经典的 **“感知-思考-行动”循环**。这个闭环在技术上具体表现为：

**第一步：感知**——Agent从用户或环境中接收输入（用户指令、传感器数据、数据库查询结果等），并将其转化为系统可以理解和处理的内部表示。

**第二步：规划与推理**——这是Agent的“思考”环节，也是其智能的核心。LLM在此扮演“中央处理器”的角色：
- 目标分解：将用户的宏观指令分解为一系列可执行的子任务
- 工具决策：根据当前上下文和子任务，决定是否需要调用外部工具
- 策略制定：规划完成任务的最佳步骤和顺序

**第三步：行动**——Agent执行规划好的行动，可能包括内部计算、工具调用、动作输出等。

**第四步：观察与反馈**——Agent观察行动产生的结果，将这些新信息作为下一轮循环的输入，评估是否更接近目标。如果未达成目标或遇到错误，它会重新规划或尝试替代方案。

这个循环持续进行，直到目标达成或任务终止。

```python
class AgentLoop:
    """
    Agent的感知-规划-行动闭环的抽象表示
    """
    def __init__(self, llm, memory, tools, planner):
        self.llm = llm          # 推理引擎
        self.memory = memory    # 记忆系统
        self.tools = tools      # 工具集
        self.planner = planner  # 规划模块
    
    def run(self, user_goal: str) -> str:
        """执行Agent闭环"""
        # 1. 感知：接收用户目标
        current_state = {"goal": user_goal, "history": []}
        
        while not self._is_goal_achieved(current_state):
            # 2. 规划：生成行动计划
            plan = self.planner.plan(current_state)
            
            # 3. 行动：执行计划中的每一步
            for action in plan:
                result = self._execute_action(action)
                # 4. 观察与反馈：更新状态
                current_state["history"].append({
                    "action": action,
                    "result": result
                })
                self.memory.update(current_state)
        
        return self._generate_response(current_state)
```

---

## 二、LLM——Agent的“中央推理引擎”

### 2.1 为什么LLM是核心

在AI Agent的架构中，LLM不仅仅是一个复杂的文本生成器；它充当着**基础推理引擎**的角色，是赋予Agent决策能力、制定计划并与环境交互能力的中央“心智”。

LLM之所以能承担这一角色，是因为它具备三大核心能力：

1. **海量的世界知识**：在预训练阶段学习了广泛的人类知识
2. **强大的上下文理解能力**：能够理解复杂、模糊的自然语言指令
3. **涌现的推理能力**：在足够大的规模下，展现出思维链等复杂推理能力

### 2.2 LLM作为推理引擎的数学本质

从数学角度看，LLM作为推理引擎，其核心是一个**条件概率模型**。在Agent的每一步决策中，LLM需要根据当前上下文生成下一步的行动：

$$a_t \sim P(a_t \mid \text{context}_t)$$

其中 $\text{context}_t$ 包含了系统提示、历史对话、工具调用结果、当前状态等所有相关信息。LLM的“推理”本质上是在这个条件概率分布上进行采样。

然而，仅仅依靠LLM的原始概率分布是不够的。Agent系统通过**提示工程**（prompt engineering）来引导LLM的推理方向。例如，思维链（Chain-of-Thought）技术通过让LLM“逐步思考”来引导其展现复杂的推理过程。

在数学上，思维链可以看作是在条件概率中引入了中间推理步骤 $r_1, r_2, ..., r_k$：

$$P(a \mid \text{context}) = \sum_{r_1, ..., r_k} P(a \mid r_k, \text{context}) \prod_{i=1}^{k} P(r_i \mid r_{i-1}, \text{context})$$

其中 $r_0 = \text{context}$。通过显式地生成这些中间推理步骤，LLM的决策过程变得更加可解释和可靠。

### 2.3 LLM在Agent中的多重角色

在Agent系统中，LLM承担着多重角色：

- **意图理解者**：解析用户的自然语言指令，提取关键信息
- **任务分解者**：将复杂目标拆解为可执行的子任务
- **决策制定者**：根据当前状态决定下一步该做什么
- **工具选择者**：判断需要调用哪些工具以及如何调用
- **结果解释者**：将工具返回的原始数据转化为用户可理解的回答

```python
class LLMReasoningEngine:
    """
    LLM作为中央推理引擎的抽象
    """
    def __init__(self, model, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
    
    def reason(self, 
               user_query: str, 
               memory_context: str, 
               tool_descriptions: str) -> dict:
        """
        基于当前上下文进行推理，决定下一步行动
        
        Returns:
            {
                "thought": "推理过程",
                "action": "要执行的动作",
                "action_input": "动作的参数"
            }
        """
        prompt = self._build_prompt(
            system=self.system_prompt,
            memory=memory_context,
            tools=tool_descriptions,
            query=user_query
        )
        
        response = self.model.generate(prompt)
        return self._parse_decision(response)
    
    def _build_prompt(self, system, memory, tools, query):
        """构建包含系统指令、记忆、工具描述和用户查询的完整提示"""
        return f"""
        {system}
        
        记忆:
        {memory}
        
        可用工具:
        {tools}
        
        用户请求: {query}
        
        请逐步推理并决定下一步行动。
        """
```

---

## 三、规划（Planning）——从目标到行动的路径

### 3.1 规划的本质

规划是AI Agent中极其重要的关键步骤，**任务规划的质量直接影响最终回答的效果**。好的任务规划甚至能让小模型的回答效果超越大模型。

规划的本质是**将高层目标分解为一系列可执行的子任务，并确定这些子任务的最佳执行顺序**。

### 3.2 规划的数学建模

从数学角度看，规划问题可以形式化为一个**规划域**（planning domain）上的搜索问题。

设：
- $\mathcal{G}$：目标空间
- $\mathcal{S}$：状态空间
- $\mathcal{A}$：行动空间
- $T: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$：状态转移函数

一个规划 $\pi = (a_1, a_2, ..., a_n)$ 是一个行动序列，使得：

$$T(T(...T(s_0, a_1), a_2), ..., a_n) \in \mathcal{G}$$

即，从初始状态 $s_0$ 开始，依次执行行动序列中的每个行动，最终达到目标状态。

在LLM-based Agent中，规划模块利用LLM的推理能力来生成这个行动序列。LLM通过“思维链”式的推理，将复杂目标逐步分解：

$$g \rightarrow (g_1, g_2, ..., g_n)$$

其中 $g$ 是原始目标，$g_i$ 是分解后的子目标，且 $\bigcup_i g_i = g$。

### 3.3 规划的两种范式

在实际应用中，任务规划存在两种主要范式：

**自主规划（LLM-driven）** ：完全交由大模型自主完成任务的拆解和规划。适用于开放域、非结构化的场景。

**人工规划（Human-driven）** ：由人工预先定义任务流程（SOP），Agent只负责执行和分析。适用于高频、垂直、需要高稳定性的场景。

企业级应用往往采用**混合架构**——高频场景用人工规划的确定性流程，开放场景用LLM的自主规划能力。

### 3.4 规划模块的代码抽象

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SubTask:
    """子任务的数据结构"""
    id: str
    description: str
    required_tools: List[str]
    dependencies: List[str]  # 依赖的其他子任务ID
    status: str  # "pending", "running", "completed", "failed"

class Planner:
    """
    Agent的规划模块
    """
    def __init__(self, llm):
        self.llm = llm
        
    def plan(self, goal: str, available_tools: List[str]) -> List[SubTask]:
        """
        将高层目标分解为子任务序列
        
        数学上，这等价于求解：
        plan = argmax_{p} P(p | goal, tools)
        其中p是任务分解方案
        """
        # 1. 让LLM进行任务分解
        decomposition = self._decompose_goal(goal, available_tools)
        
        # 2. 构建子任务依赖图
        tasks = self._build_dependency_graph(decomposition)
        
        # 3. 拓扑排序确定执行顺序
        ordered_tasks = self._topological_sort(tasks)
        
        return ordered_tasks
    
    def _decompose_goal(self, goal: str, tools: List[str]) -> List[Dict]:
        """调用LLM进行任务分解"""
        prompt = f"""
        目标: {goal}
        可用工具: {tools}
        
        请将上述目标分解为3-5个可执行的子任务。
        每个子任务需要说明: 任务描述、所需工具、依赖的前置任务。
        """
        response = self.llm.generate(prompt)
        return self._parse_decomposition(response)
    
    def _topological_sort(self, tasks: List[SubTask]) -> List[SubTask]:
        """
        基于依赖关系进行拓扑排序
        
        这确保了: 如果任务B依赖任务A，则A在B之前执行
        """
        # 实现标准的拓扑排序算法
        # 时间复杂度: O(V + E)，其中V是任务数，E是依赖边数
        ...
```

---

## 四、工具（Tools）——Agent的“四肢”

### 4.1 工具的本质

如果说LLM是Agent的“大脑”，那么**工具就是Agent的“四肢”** ——它们让Agent能够真正地影响世界。

工具使LLM能够与外部环境和应用程序交互，包括**数据获取**（如搜索、数据库查询）和**行动执行**（如发送邮件、调用API）两类。

工具调用的核心在于**建立语言模型输出与可执行函数之间的映射关系**。它赋予了模型从“说”（say）到“做”（do）的能力。

### 4.2 Function Calling：工具调用的标准化接口

目前主流的工具调用机制是**Function Calling**（函数调用）。其核心思想是：通过对模型进行特定的微调，使其能够识别出用户意图中需要借助外部工具才能完成的部分，并输出结构化的函数调用指令。

在数学上，Function Calling可以看作是在标准的条件概率生成中引入了一个**结构化的行动空间**：

$$
P(a_t \mid \text{context}) = \begin{cases}
P_{\text{text}}(y_t \mid \text{context}) & \text{若 } a_t \text{ 是文本回复} \\
P_{\text{function}}(f, \theta \mid \text{context}) & \text{若 } a_t \text{ 是函数调用}
\end{cases}
$$

其中 $f$ 是要调用的函数名，$\theta$ 是函数参数。

模型的输出通过一个**结构化接口**（通常是JSON格式）来表达工具调用：

```json
{
    "name": "search",
    "arguments": {
        "query": "2024年诺贝尔物理学奖得主"
    }
}
```

### 4.3 工具调用的工作流程

一个完整的工具调用流程包含以下环节：

1. **工具注册**：将可用的工具以统一的接口规范（JSON Schema）注册到系统中
2. **意图识别**：LLM分析用户请求，判断是否需要调用工具
3. **函数选择**：LLM从可用工具中选择最合适的那个
4. **参数提取**：LLM从上下文中提取调用所需的具体参数
5. **执行与返回**：系统执行函数调用，将结果返回给LLM
6. **结果整合**：LLM将工具返回的结果融入最终的回复中

### 4.4 工具的抽象接口设计

```python
from typing import Any, Dict, Callable
from dataclasses import dataclass
import json

@dataclass
class Tool:
    """
    工具的标准化抽象接口
    """
    name: str                          # 工具名称
    description: str                   # 功能描述
    parameters: Dict[str, Any]         # 参数的JSON Schema
    func: Callable                     # 实际执行的函数
    
    def to_openai_schema(self) -> Dict:
        """转换为OpenAI Function Calling格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """执行工具调用"""
        return self.func(**kwargs)


class ToolRegistry:
    """
    工具注册中心：管理所有可用工具
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """注册一个工具"""
        self._tools[tool.name] = tool
    
    def get_schemas(self) -> List[Dict]:
        """获取所有工具的OpenAI schema"""
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    def execute(self, name: str, arguments: Dict) -> Any:
        """根据名称和参数执行工具"""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].execute(**arguments)


# 使用示例：注册一个搜索工具
def search_function(query: str) -> str:
    """模拟搜索功能"""
    return f"搜索结果: 关于'{query}'的信息..."

search_tool = Tool(
    name="search",
    description="搜索互联网获取信息",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            }
        },
        "required": ["query"]
    },
    func=search_function
)

registry = ToolRegistry()
registry.register(search_tool)
```

### 4.5 工具调用的演进：MCP协议

随着Agent系统越来越复杂，工具调用的标准化需求日益迫切。**模型上下文协议**（Model Context Protocol, MCP）应运而生，它标准化了对各种服务的API访问。

MCP包含三个核心组件：
- **MCP主机**：LLM应用程序
- **MCP客户端**：维护连接
- **MCP服务器**：提供上下文和能力

MCP的核心价值在于：**如何发现工具、如何调用工具、返回结果怎么表达**都被标准化为统一接口。这极大地降低了Agent接入新工具的复杂度。

---

## 五、记忆（Memory）——Agent的“意识流”与“经验库”

### 5.1 为什么Agent需要记忆

LLM本身不具备记忆能力——每次对话都是独立的，模型无法“记住”之前说过什么。然而，一个真正的Agent需要在多轮交互中保持连贯性，需要从历史经验中学习，需要记住用户的偏好。

记忆机制是实现持续、连贯和个性化交互的**核心基石**。它让Agent具备了知识与经验积累和优化的能力。

### 5.2 记忆的二分法：短期记忆与长期记忆

Agent的记忆系统模拟了人类的认知结构，通常划分为**短期记忆**和**长期记忆**。

#### 短期记忆（Short-term Memory）

短期记忆是Agent用于处理当前任务或单次会话的**临时信息存储区**。它本质上是Agent的 **“工作台”或“意识流”** ，包含了理解当前请求并生成回应所必需的全部上下文信息。

在技术实现上，短期记忆直接对应于LLM的**上下文窗口**（context window）。这个窗口是一个固定长度的token序列，包含了：
- 系统指令：定义Agent的角色、能力和行为准则
- 对话历史：当前会话中所有交替出现的用户查询和Agent回复
- 工具返回结果：工具调用的返回数据
- 当前查询：用户最新提出的问题

短期记忆的有效性依赖于Transformer架构的**注意力机制**（Attention Mechanism）。该机制允许模型在处理当前词句时，动态地关注上下文窗口中所有相关的历史信息。

短期记忆的核心限制是**有限容量**——上下文窗口的大小是硬性限制。当对话内容超出窗口长度时，会发生“上下文截断”，最早的信息被丢弃。

#### 长期记忆（Long-term Memory）

长期记忆是Agent在多次交互和会话之间**持久化存储信息**的系统。它相当于Agent的“个人日记”或“知识库”，用于保留需要跨会话记忆的关键信息。

长期记忆在技术实现上通常通过**外部向量数据库**来实现。采用**检索增强生成**（RAG）技术，将信息嵌入为向量表示，建立可检索的记忆库。

长期记忆可以进一步细分为：
- **语义记忆**（Semantic Memory）：存储概念性知识、事实信息
- **情景记忆**（Episodic Memory）：存储特定事件、经历的记录

```python
from typing import List, Dict, Any
import numpy as np

class ShortTermMemory:
    """
    短期记忆：基于上下文窗口
    """
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = []
        self.current_token_count = 0
    
    def add(self, role: str, content: str):
        """添加一条消息到短期记忆"""
        # 估算token数（简化版）
        token_count = len(content) // 4
        self.messages.append({"role": role, "content": content})
        self.current_token_count += token_count
        
        # 如果超出窗口限制，移除最早的消息
        while self.current_token_count > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            self.current_token_count -= len(removed["content"]) // 4
    
    def get_context(self) -> List[Dict[str, str]]:
        """获取当前上下文"""
        return self.messages


class LongTermMemory:
    """
    长期记忆：基于向量数据库的语义检索
    """
    def __init__(self, vector_db):
        self.vector_db = vector_db  # 如 Chroma, Pinecone, FAISS
    
    def store(self, content: str, metadata: Dict[str, Any]):
        """
        存储信息到长期记忆
        
        数学上，这等价于将文本嵌入到向量空间：
        v = Embed(content) ∈ R^d
        然后将v存储在向量数据库中
        """
        embedding = self._embed(content)
        self.vector_db.insert(embedding, content, metadata)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关记忆
        
        数学上，这等价于在向量空间中寻找最近邻：
        q = Embed(query) ∈ R^d
        results = argmin_{v ∈ V} ||q - v||₂
        其中V是存储的所有向量
        """
        query_embedding = self._embed(query)
        results = self.vector_db.search(query_embedding, top_k)
        return results
    
    def _embed(self, text: str) -> np.ndarray:
        """将文本转换为向量嵌入"""
        # 使用嵌入模型（如 text-embedding-ada-002）
        pass
```

### 5.3 记忆的协同工作

短期记忆与长期记忆在Agent系统中协同工作：

1. **写入**：当前会话的关键信息可以从短期记忆**沉淀**到长期记忆
2. **读取**：在开始新会话或处理新任务时，从长期记忆中**检索**相关信息加载到短期记忆
3. **更新**：长期记忆中的信息可以根据新的交互进行**更新和修正**

这种协同机制使得Agent既能保持当前对话的连贯性（短期记忆），又能实现跨会话的知识积累（长期记忆）。

---

## 六、总结：四大模块的协同

AI Agent的核心架构可以用一个简洁的公式来概括：

$$\text{Agent} = \text{LLM} + \text{Planning} + \text{Memory} + \text{Tools}$$

这四个模块各司其职，又紧密协同：

| 模块 | 角色 | 核心功能 |
|------|------|----------|
| **LLM** | 大脑/推理引擎 | 理解意图、生成推理、制定决策 |
| **Planning** | 规划师 | 分解目标、制定步骤、确定顺序 |
| **Tools** | 四肢/执行器 | 获取信息、执行操作、影响环境 |
| **Memory** | 记忆系统 | 维持上下文、存储经验、实现个性化 |

它们共同构成了一个完整的**感知-规划-行动**闭环。这个闭环让AI从一个被动的“信息处理器”跃迁为一个主动的 **“目标达成者”** 。

正如李飞飞等人在其Agent AI综述中所阐述的：**从感知、认知、行动，到学习与记忆，构成了一个动态迭代的智能体体系**。这不仅是技术的整合，更是对未来AGI路径的系统性构想。

在接下来的博客中，我们将深入探讨在这个核心架构之上构建的各种具体范式——从ReAct的推理-行动交织，到Reflexion的自我反思，再到LATS的树搜索——看它们如何一步步扩展Agent的能力边界。