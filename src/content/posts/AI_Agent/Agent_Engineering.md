---
title: Agent Engineering —— Prompt & Context & Harness & Loop
published: 2026-04-12
description: 系统剖析AI Agent工程从Prompt Engineering到Context Engineering再到Harness Engineering的三层演进路径：以数学公式形式化Prompt如何塑造条件概率分布、Context如何通过装配方程治理信息环境、Harness如何通过七层架构构建执行控制框架，并探讨Loop Engineering作为第四层延伸的嵌套关系。
cover: "/assets/images/posts/agent_engineering.png"
coverInContent: false
tags: [Prompt Engineering, Context Engineering, Harness Engineering, Loop Engineering, AI Agent, 沙箱]
category: AI_Agent
draft: false
---

# Agent Engineering —— Prompt & Context & Harness & Loop

## 引言：为什么你的Agent在Demo里惊艳，在生产中拉胯？

2026年，几乎每家企业都在谈AI Agent。团队花了三周搭了一个Agent原型，接入了内部知识库，CEO看了Demo点头说“不错”——然后呢？上线两周后，Agent把客户的订单信息张冠李戴，把合同条款搞混，在凌晨三点自动发了一封莫名其妙的邮件。

这个故事正在无数企业里反复上演。问题不在模型不够聪明——**问题在于：你在用Demo阶段的工程方法论，去解决生产阶段的系统问题**。

过去四年，AI工程领域经历了一场静悄悄的范式迁移。从2022年的**Prompt Engineering**，到2025年的**Context Engineering**，到2026年的**Harness Engineering**。Andrej Karpathy在2025年说“prompt engineering应该让位给context engineering”；Mitchell Hashimoto（HashiCorp联合创始人）在2026年2月提出“engineer the **harness**”；Boris Cherny（Anthropic Claude Code负责人）在2026年6月说“我不再prompt Claude了，我设计**Loop**来prompt Claude”。

这不是大佬们在造词。他们各自独立地撞上了同一堵墙：**单靠改善prompt无法让Agent在生产环境中可靠运行；单靠优化context无法防止Agent犯同样的错**。

本文将系统拆解Agent工程的四个递进层级，给出每一层的数学原理与代码实现。

---

## 一、Prompt Engineering——“怎么问”

### 1.1 定义与核心问题

**Prompt Engineering**是最基础的工程层，关注的是**如何用自然语言精确地表达任务**，从而引导模型输出符合预期的结果。

在大模型刚兴起时，最直观的现象是：同一个模型，仅仅改变提问方式，输出结果就可能发生巨大变化。因此，当时的核心共识是：**模型不是不会，而是你没有把问题表达清楚**。

Prompt Engineering的核心手段包括：角色设定、思维链（CoT）引导、少样本示例、输出格式约束等。

### 1.2 数学本质：塑造概率分布

从数学上看，Prompt Engineering的本质是**通过输入文本调整模型的条件概率分布**。设词表为 $\mathcal{V}$，给定提示词 $p$，LLM生成输出 $y$ 的概率为：

$$P(y \mid p) = \prod_{t=1}^{|y|} P(y_t \mid y_{<t}, p)$$

Prompt Engineering的目标是找到一个最优的提示词 $p^*$，使得目标输出 $y^*$ 的概率最大化：

$$p^* = \arg\max_{p \in \mathcal{P}} P(y^* \mid p)$$

其中 $\mathcal{P}$ 是所有可能的提示词空间。

更精确地说，提示词通过**塑造模型的局部概率分布**来影响输出。一个精心设计的prompt将目标输出从“低概率事件”变为“高概率事件”。temperature参数控制着这个概率分布的平滑程度：

$$p_i' = \text{softmax}\left(\frac{\log(p_i)}{\text{temperature}}\right)$$

### 1.3 局限：天花板是硬性的

Prompt Engineering的局限性同样明显：

- **无状态**：每次调用独立，没有记忆
- **无法注入外部知识**：无法告知模型代码库里的最新变更
- **无法处理跨步骤协作**：一旦任务需要调用工具或追踪状态，单靠Prompt撑不住整个系统

> **Prompt Engineering问的是“怎么表达任务”。**

### 1.4 代码示例：从朴素到工程化

```python
# ❌ 朴素Prompt：信息太少
prompt_naive = "Fix the bug in my code"

# ✅ 工程化Prompt：结构化、分步骤、带约束
prompt_engineered = """
你是一位资深Python工程师，正在审查一个生产环境Bug。

背景信息：
- Bug导致orders.py第47行抛出KeyError
- 仅在周末批处理时触发
- 系统使用PostgreSQL + 读副本

请按以下步骤执行：
1. 首先定位根本原因，不修改任何代码
2. 描述什么数据条件触发了该错误
3. 提出一个保持向后兼容的修复方案
4. 列出需要新增的测试用例

在确认诊断之前，不要修改任何文件。
"""
```

正如HumanLayer工程团队的观察：**“这不是模型问题，是配置问题。”** 更聪明的模型只是被分配更难的任务，同样的失败模式照样会出现。

---

## 二、Context Engineering——“给模型看什么”

### 2.1 定义与核心问题

当Prompt Engineering碰到天花板后，行业意识到：**问题不在于“怎么说”，而在于“给模型看什么”**。

**Context Engineering**（上下文工程）关注的是**模型在执行任务时处于什么信息环境中**。它不再只关注单次prompt，而是关注模型每次推理时看到的**全部信息**：系统提示、对话历史、检索到的文档（RAG）、工具调用结果、项目文件内容等。

Context Engineering的本质是**治理信息环境**——从“写一条好prompt”升级到“设计模型每次决策时的完整信息环境”。

### 2.2 数学建模：上下文装配方程

Context Engineering可以用一个**上下文装配方程**来形式化：

$$C = A(c_1, c_2, c_3, c_4, c_5, c_6)$$

其中 $C$ 是装配完成的上下文（最终输入给LLM的内容），$A$ 是装配函数，$c_i$ 是各个上下文组件：

- $c_1$：系统指令（System Prompt）
- $c_2$：用户输入与对话历史
- $c_3$：检索结果（RAG）
- $c_4$：工具调用返回结果
- $c_5$：当前任务状态与中间结果
- $c_6$：系统规则与安全约束

Context Engineering的优化目标可以形式化为：

$$F^* = \arg\max_{F} \mathbb{E}_{\tau \sim T} \left[ \text{Reward}\left(P_\theta(Y \mid C_F(\tau)), Y_\tau^*\right) \right]$$

其中 $F$ 是一组上下文生成函数（如检索、选择、格式化、装配等），$C_F(\tau)$ 是任务 $\tau$ 的上下文，$Y_\tau^*$ 是期望输出。

### 2.3 核心能力

Context Engineering需要解决三个核心问题：

**① 检索（Retrieval）** ：从知识库中动态注入相关文档。这是RAG的核心。**选对文档而不是全量灌入**，是Context Engineering的第一原则。

**② 压缩（Compression）** ：在长对话或长文档中保留关键信息、丢弃噪音。**“正确的300个token胜过10万个嘈杂的token”**。

**③ 时序管理（Temporal Management）** ：决定历史信息何时保留、何时摘要、何时丢弃。典型实践是“渐进式披露”——不是一次性提供全部信息，而是在需要时逐步注入。

### 2.4 Prompt vs Context：包含关系

需要特别强调的是：**Prompt只是Context的一个子集**。Context Engineering包含Prompt Engineering，并在其上增加了信息环境的治理维度。

> **Context Engineering问的是“模型工作时应该处于什么信息环境里”。**

### 2.5 代码示例：上下文装配器

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ContextComponent:
    """上下文组件"""
    name: str
    content: str
    priority: int  # 优先级，决定装配顺序
    max_tokens: int  # 该组件允许的最大token数

class ContextAssembler:
    """
    上下文装配器：将多个组件装配为最终上下文
    
    数学上，这实现了装配函数 C = A(c₁, c₂, ..., cₙ)
    """
    
    def __init__(self, max_total_tokens: int = 128000):
        self.max_total_tokens = max_total_tokens
        self.components: List[ContextComponent] = []
    
    def add_component(self, component: ContextComponent):
        """添加上下文组件"""
        self.components.append(component)
    
    def assemble(self, query: str) -> str:
        """
        装配最终上下文
        
        策略：按优先级排序，在token预算内渐进式注入
        """
        # 1. 按优先级排序
        sorted_components = sorted(self.components, key=lambda c: c.priority, reverse=True)
        
        # 2. 渐进式装配
        assembled = []
        current_tokens = 0
        
        # 先注入系统指令（最高优先级）
        for comp in sorted_components:
            # 估算token数（简化版）
            estimated_tokens = len(comp.content) // 4
            
            if current_tokens + estimated_tokens <= self.max_total_tokens:
                assembled.append(f"--- {comp.name} ---\n{comp.content}")
                current_tokens += estimated_tokens
            else:
                # token预算不足，尝试压缩
                compressed = self._compress(comp.content, self.max_total_tokens - current_tokens)
                if compressed:
                    assembled.append(f"--- {comp.name} (压缩) ---\n{comp}")
                    break
        
        # 3. 添加当前查询
        assembled.append(f"--- 用户查询 ---\n{query}")
        
        return "\n\n".join(assembled)
    
    def _compress(self, content: str, budget: int) -> str:
        """压缩内容以适应token预算"""
        # 实际实现可使用LLM生成摘要
        # 或用滑动窗口截取
        if len(content) // 4 <= budget:
            return content
        # 简化：取前budget个token
        return content[:budget * 4]
```

---

## 三、Harness Engineering——“在哪里安全运行”

### 3.1 定义与核心问题

即便模型理解正确、信息充分，也未必能够稳定完成任务。常见问题包括：执行过程中逐渐偏离目标、错误使用工具、长链路任务中状态混乱、无法发现自身错误。

这类问题本质上已经超出了**输入侧优化**的范畴。Prompt和Context解决的是“输入问题”，而这里需要解决的是 **“执行过程问题”** 。

**Harness Engineering**（驾驭工程）应运而生。Harness的原意是“马具”——套在马身上的缰绳、嚼子和鞍具。马提供动力，但马具控制方向、速度和安全。在AI语境中：**模型是马，Harness是缰绳。模型提供智能，Harness提供控制**。

行业的核心公式是：

$$\text{Agent} = \text{Model} + \text{Harness}$$

Harness是AI Agent中除模型本身之外的所有代码、配置和执行逻辑。原始模型不是Agent——只有当Harness赋予它状态、工具执行、反馈循环和可强制执行的约束时，它才成为Agent。

### 3.2 Harness的七层架构

卡内基梅隆大学等研究团队提出了Harness工程的**ETCLOVG七层分类**：

| 层级 | 名称 | 职责 |
|------|------|------|
| **E** | 执行环境与沙箱 | Agent代码在哪里运行、受到什么约束 |
| **T** | 工具接口与协议 | 外部能力如何被描述、发现和调用 |
| **C** | 上下文管理 | 模型在短期、会话级和持久化层面能看到什么 |
| **L** | 生命周期与编排 | 组织状态读写控制流，覆盖单Agent循环到多Agent协作 |
| **O** | 可观测性 | 捕获轨迹、成本、失败和可靠性信号 |
| **V** | 验证 | 将任务和轨迹转化为评估、失败归因和回归反馈 |
| **G** | 治理 | 权限、身份、策略、安全加固、审计和人工监督 |

前四层构成Harness的**结构核心**，后三层对应围绕核心的**控制平面**。

### 3.3 Harness的量化效果

Harness Engineering的价值可以通过数据量化：

- 仅仅是给同一个大语言模型换上一套更精巧的Harness架构，它在**Terminal Bench 2.0**（衡量AI编程能力的权威榜单）上的通过率，直接从**52.8%拉升到了66.5%**。底层模型的权重一个字节都没改。
- 在不改模型权重的情况下，**仅调整harness层本身**，就可能显著改变Agent在coding和terminal benchmark上的表现。
- 研究显示，围绕固定语言模型的wrapper——每一步存储、检索和呈现什么——可以改变**端到端性能高达可观的程度**。

### 3.4 Harness的核心构件

一个完整的Harness包含以下核心构件：

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import time

@dataclass
class ToolSpec:
    """工具规范"""
    name: str
    description: str
    parameters: Dict[str, Any]
    permissions: List[str]  # 所需权限
    is_dangerous: bool = False

@dataclass
class ExecutionContext:
    """执行上下文"""
    task_id: str
    user_id: str
    session_id: str
    permissions: List[str]
    start_time: float = field(default_factory=time.time)
    state: Dict[str, Any] = field(default_factory=dict)

class Harness:
    """
    Agent Harness：模型的"缰绳、马鞍与路"
    
    Harness = 执行环境 + 工具接口 + 上下文管理 + 编排 + 可观测性 + 验证 + 治理
    """
    
    def __init__(self, model, tools: List[ToolSpec], max_steps: int = 20):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.logger = logging.getLogger("harness")
        self.audit_trail: List[Dict] = []  # 审计日志
    
    def run(self, task: str, context: ExecutionContext) -> str:
        """
        在Harness约束下执行任务
        
        Harness的四个核心控制机制：
        1. 权限校验（治理层）
        2. 工具调用边界（工具接口层）
        3. 可观测性（观测层）
        4. 审计就绪（治理层）
        """
        self.logger.info(f"开始执行任务: {task}, 用户: {context.user_id}")
        
        # 1. 权限校验（Governance）
        if not self._check_permissions(context, task):
            self._audit("permission_denied", context, {"task": task})
            raise PermissionError(f"用户 {context.user_id} 无权执行此任务")
        
        state = {"task": task, "step": 0, "history": []}
        
        while state["step"] < self.max_steps:
            # 2. 构建上下文（Context Management）
            prompt = self._build_prompt(state, context)
            
            # 3. 模型推理
            response = self.model.generate(prompt)
            
            # 4. 解析行动
            action = self._parse_action(response)
            
            # 5. 工具调用边界检查（Tool Interface）
            if action["type"] == "tool_call":
                tool_name = action["tool"]
                if tool_name not in self.tools:
                    self._audit("unknown_tool", context, {"tool": tool_name})
                    return f"错误：未知工具 {tool_name}"
                
                tool = self.tools[tool_name]
                
                # 6. 工具级权限校验
                if not self._check_tool_permissions(context, tool):
                    self._audit("tool_permission_denied", context, {"tool": tool_name})
                    return f"错误：无权调用工具 {tool_name}"
                
                # 7. 执行工具（在沙箱中）
                try:
                    result = self._execute_in_sandbox(tool, action["args"])
                    observation = f"Observation: {result}"
                except Exception as e:
                    observation = f"Observation: 错误 - {str(e)}"
                    self.logger.error(f"工具执行失败: {e}")
                
                # 8. 审计记录（Observability + Governance）
                self._audit("tool_call", context, {
                    "tool": tool_name,
                    "args": action["args"],
                    "result": result if "result" in locals() else None,
                    "error": str(e) if "e" in locals() else None
                })
                
                state["history"].append({"action": action, "observation": observation})
            
            elif action["type"] == "finish":
                # 9. 验证输出（Verification）
                if self._verify_output(action["answer"]):
                    self._audit("task_completed", context, {"answer": action["answer"]})
                    return action["answer"]
                else:
                    self._audit("verification_failed", context, {"answer": action["answer"]})
                    # 继续循环，让模型修正
            
            state["step"] += 1
        
        self._audit("max_steps_reached", context, {"steps": state["step"]})
        return "达到最大步数，任务未完成"
    
    def _check_permissions(self, context: ExecutionContext, task: str) -> bool:
        """治理层：权限校验"""
        # 实现权限检查逻辑
        return True
    
    def _check_tool_permissions(self, context: ExecutionContext, tool: ToolSpec) -> bool:
        """治理层：工具权限校验"""
        return all(p in context.permissions for p in tool.permissions)
    
    def _execute_in_sandbox(self, tool: ToolSpec, args: Dict) -> Any:
        """执行环境与沙箱：工具在受控环境中执行"""
        # 实际实现中，这里会使用容器化或隔离执行环境
        pass
    
    def _verify_output(self, output: str) -> bool:
        """验证层：输出验证"""
        # 实现输出验证逻辑
        return True
    
    def _audit(self, event_type: str, context: ExecutionContext, data: Dict):
        """治理+观测层：审计日志"""
        self.audit_trail.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": context.user_id,
            "task_id": context.task_id,
            "data": data
        })
        self.logger.info(f"审计: {event_type} - {data}")
```

### 3.5 Harness vs Context vs Prompt：三层包含关系

三者不是竞争关系，而是**逐层扩展的包含关系**：

> **Context Engineering包含Prompt Engineering；Harness Engineering包含Context Engineering**。

用一句话概括：
- **Prompt Engineering解决表达问题**
- **Context Engineering解决信息问题**
- **Harness Engineering解决执行问题**

**外层不取消内层，而是在内层的基础上增加新的工程维度**。如果你的prompt写得模糊、context配置混乱，再好的harness也救不了你——因为模型每次推理时收到的指令和信息本身就是低质量的。

---

## 四、Loop Engineering——第四维度的延伸

### 4.1 从三层到四层

2026年中，行业开始讨论**Loop Engineering**（循环工程）作为第四层。其核心思想是：**设计一个让Agent自主循环的系统，而不是手动逐轮prompt模型**。

> **“你不应该继续给编码Agent写提示词了，你应该设计循环，让循环去提示你的Agent。”**

Loop Engineering的五个组成部分是：自动化、工作树、skill、插件和连接器、子Agent，外加一个跨会话的记忆层。

### 4.2 四层的嵌套关系

四层工程是**嵌套关系**，不是替代关系：

```
Loop Engineering（循环工程）
    └── Harness Engineering（执行工程）
            └── Context Engineering（信息工程）
                    └── Prompt Engineering（表达工程）
```

**Loop Engineering包含以上所有层**。

### 4.3 各层级落地侧重点

| 层级 | 核心问题 | 落地侧重点 | 典型技术 |
|------|---------|-----------|---------|
| **Prompt** | 怎么问 | 指令表达、输出格式、角色设定 | 思维链、少样本、结构化输出 |
| **Context** | 给模型看什么 | RAG检索、记忆召回、上下文压缩 | 向量数据库、语义分块、渐进式披露 |
| **Harness** | 在哪里安全运行 | 工具接入边界、权限校验、审计就绪 | 沙箱、权限系统、可观测性、治理 |
| **Loop** | 怎么让Agent自己跑 | 自动化心跳、工作树隔离、子Agent制衡 | 循环编排、状态持久化、自主决策 |

---

## 五、总结：从控制模型到控制智能体

AI工程的演进从来不是替代，而是**层层叠加的抽象升级**。

- **Prompt Engineering**（2022-2024）：把模型当作一个文本框，优化输入
- **Context Engineering**（2025）：把模型当作一个信息处理器，优化信息环境
- **Harness Engineering**（2026）：把模型当作一个执行者，构建控制框架
- **Loop Engineering**（2026-）：把模型当作一个自主工作者，设计让它自己跑的循环

每一次演进，都是工程关注点的实质性跃迁。正如研究团队所总结的：**“Harness——提示词、工具选择、内存管理、编排逻辑、安全检查——让智能变得可操作”**。而Loop Engineering则更进一步：**把人从“循环里的操作员”挪到“循环外的设计者”**。

**模型是马，Harness是缰绳，Loop是让它自己跑起来的自动驾驶系统**。三者的组合，才是Agent从Demo走向生产的完整路径。