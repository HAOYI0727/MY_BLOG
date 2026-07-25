---
title: Agent Paradigms —— ReAct & Plan-and-Solve & Reflection
published: 2026-04-04
description: 系统剖析AI Agent的三大核心范式：ReAct（推理与行动交替的Thought-Action-Observation三拍循环）、Plan-and-Solve（先分解规划后逐步执行的两阶段架构）与Reflection（生成-批评-修正的自我优化机制）。从数学建模角度揭示ReAct如何通过外部观察修正条件概率分布、Plan-and-Solve如何将任务分解为子目标序列、Reflexion如何以“语义梯度”替代参数梯度实现零样本学习，并通过实验数据对比三种范式的适用场景与成本特征。
cover: "/assets/images/posts/agent_paradigms.png"
coverInContent: false
tags: [AI Agent, ReAct, Plan-and-Solve, Reflection, CoT]
category: AI_Agent
draft: false
---

# Agent Paradigms —— ReAct & Plan-and-Solve & Reflection

## 引言：从“单次推理”到“闭环智能”

在上一篇文章中，我们拆解了AI Agent的四大核心模块——LLM、规划、工具与记忆。如果把这四个模块比作Agent的“器官”，那么**范式（Paradigm）** 就是让这些器官协同工作的“神经系统”——它决定了Agent如何思考、如何行动、如何从错误中学习。

早期的大语言模型应用遵循一种朴素的模式：用户输入问题 → 模型生成答案。这种“一次性推理”在面对复杂任务时往往力不从心——模型可能在推理中途出错却无法修正，可能缺乏关键信息却无法主动获取，可能生成错误答案却浑然不觉。

为了解决这些问题，研究者们先后提出了三种具有里程碑意义的Agent范式：

- **ReAct（2022）** ：让Agent在“推理”与“行动”之间来回切换，边想边做
- **Plan-and-Solve（2023）** ：让Agent先规划再执行，避免在复杂任务中“走一步看一步”导致的迷失
- **Reflection / Reflexion（2023）** ：让Agent能够批评自己的输出并迭代改进，实现“自我优化”

这三种范式并非相互替代，而是**层层递进、互为补充**。理解它们的核心机制，是构建真正可用Agent系统的必修课。

---

## 一、ReAct——推理与行动的协同

### 1.1 从CoT到ReAct：打破“信息真空”

在ReAct提出之前，**思维链（Chain-of-Thought, CoT）** 已经证明了大语言模型通过“逐步思考”可以解决复杂的推理问题。然而，CoT存在一个根本性的缺陷：**模型的推理过程完全在“信息真空”中进行，完全依赖其内部知识**。这意味着：
- 模型无法获取外部信息来填补知识空白
- 模型无法验证自己推理步骤的正确性
- 模型容易产生“幻觉”（hallucination）

2022年，Shunyu Yao等人提出了**ReAct（Reasoning + Acting）** 范式。ReAct的核心思想极其简洁却极具革命性：**让LLM交替生成推理轨迹（reasoning traces）和任务特定的行动（task-specific actions）**。推理轨迹帮助模型诱导、跟踪和更新行动计划并处理异常，而行动则允许模型与外部知识库或环境交互以收集额外信息。

用更通俗的话说：**CoT让模型“自己想”，ReAct让模型“边想边做”** 。

### 1.2 Thought-Action-Observation：三拍循环

ReAct的运作遵循一个标准化的三拍循环：

**Thought（思考）** ：Agent根据当前状态生成一段推理文本，分析“我现在知道什么”“我还需要什么”“下一步该做什么”。这段推理不会影响外部环境，但会更新模型的内部状态。

**Action（行动）** ：Agent根据思考结果执行一个具体行动。这个行动可以是调用一个工具（如搜索、计算、API调用），也可以是输出最终答案。

**Observation（观察）** ：Agent接收行动执行后从外部环境返回的反馈。这个观察结果成为下一轮循环的输入。

这三个步骤形成一个闭环，不断迭代直到任务完成。一个典型的ReAct轨迹如下：

```
Thought 1: 我需要查一下2024年诺贝尔物理学奖得主是谁。
Action 1: 搜索“2024 诺贝尔物理学奖”
Observation 1: 搜索结果：John J. Hopfield和Geoffrey E. Hinton因在人工神经网络方面的基础性发现和发明获奖。
Thought 2: 好的，我已经知道了获奖者。现在我需要确认他们的国籍。
Action 2: 搜索“John J. Hopfield 国籍”
Observation 2: John J. Hopfield是美国物理学家。
Thought 3: 我已经收集了足够的信息，可以给出最终答案了。
Action 3: 完成
```

### 1.3 数学建模：ReAct的条件概率框架

从数学角度看，ReAct可以形式化为一个条件概率生成过程。

设 $c_t = (o_1, a_1, o_2, a_2, ..., o_t)$ 为时刻 $t$ 的上下文。在每一步，模型根据当前上下文生成下一个token：

$$P(a_t \mid c_t) = \prod_{i=1}^{|a_t|} P(x_i \mid x_{<i}, c_t)$$

其中 $a_t$ 可以是推理文本（Thought）、行动指令（Action）或最终答案。

ReAct的整个轨迹是一个概率链：

$$P(\text{trajectory}) = \prod_{t=1}^{T} P(a_t \mid c_t)$$

其中状态更新是推理文本和观察结果的拼接：

$$c_{t+1} = c_t \oplus \text{Thought}_t \oplus \text{Observation}_t$$

这个数学框架揭示了ReAct的本质：**它是一个在“推理空间”和“行动空间”之间交替采样的过程**，通过外部观察来不断修正条件概率分布。

### 1.4 CoT与ReAct：互补而非替代

需要特别强调的是，**ReAct并非要取代CoT，而是与之互补**。

CoT擅长的是**纯粹的推理任务**——那些不需要外部信息、完全依赖内部知识就能解决的问题。而ReAct擅长的是**需要与外部世界交互的任务**——问答、事实核查、决策制定等。

实验数据清楚地展示了这种互补性：在事实核查任务（FEVER）上，ReAct显著优于CoT，因为它可以主动验证信息。而在某些纯推理任务上，CoT可能更加高效。

从工程角度看，两者的关系可以这样理解：**CoT是ReAct的“推理引擎”** ——ReAct中的Thought步骤本质上就是在执行CoT式的推理。ReAct在CoT的基础上增加了“行动”和“观察”两个维度，使推理不再局限于模型的内部知识。

### 1.5 ReAct的动态纠错：在歧义环境中导航

ReAct最令人印象深刻的能力之一，是它在面对歧义或错误时能够**动态纠错**。

传统的CoT推理是一条直线——模型一旦在某一步推理出错，整个答案就可能完全错误，而且没有任何机制可以纠正。ReAct则不同：当模型发现自己的推理与观察结果不一致时，它可以**重新思考、调整策略**。

这种纠错能力在数学上可以理解为：观察结果 $o_t$ 提供了新的证据，更新了模型对世界的信念：

$$P(\text{goal} \mid c_{t+1}) \propto P(o_t \mid \text{goal}, c_t) \cdot P(\text{goal} \mid c_t)$$

当观察结果与当前假设矛盾时，后验概率 $P(\text{goal} \mid c_{t+1})$ 会重新分配，促使模型修正其推理方向。

LangChain团队通过分析200多个生产环境案例发现，**87%的成功Agent实现最终收敛于ReAct模式**。这一数据充分说明了ReAct在实际应用中的价值。

### 1.6 代码实现：从零构建ReAct Agent

以下是一个精简但完整的ReAct Agent实现：

```python
from typing import List, Dict, Any, Optional
import json
import re

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    
    核心循环: Thought → Action → Observation
    """
    
    def __init__(self, llm, tools: Dict[str, callable], max_steps: int = 10):
        """
        初始化ReAct Agent
        
        Args:
            llm: 大语言模型接口（具有 generate(prompt) 方法）
            tools: 工具字典 {工具名: 可调用函数}
            max_steps: 最大迭代步数，防止无限循环
        """
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.trajectory = []  # 存储完整的 Thought-Action-Observation 轨迹
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词，定义ReAct的格式规范"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.__doc__ or '无描述'}"
            for name, tool in self.tools.items()
        ])
        
        return f"""
        你是一个ReAct智能体。你可以通过“思考→行动→观察”的循环来解决问题。
        
        可用工具:
        {tool_descriptions}
        
        请严格按照以下格式输出你的每一步：
        Thought: 你的推理过程（自然语言）
        Action: 要调用的工具名称，或 "Finish" 表示结束
        Action Input: 工具的输入参数（JSON格式），或最终答案
        
        注意：Action Input 必须是有效的JSON字符串。
        """
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        解析LLM的输出，提取 Thought、Action 和 Action Input
        
        数学上，这是将非结构化的文本映射到结构化行动空间：
        parse: Text → {Thought, Action, Action Input}
        """
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", response, re.DOTALL)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?=\n\s*(?:Thought|Action)|\Z)", response, re.DOTALL)
        
        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "action_input": action_input_match.group(1).strip() if action_input_match else ""
        }
    
    def run(self, question: str) -> str:
        """
        运行ReAct主循环
        
        数学上，这等价于迭代执行：
        for t = 1 to T:
            Thought_t, Action_t = LLM(context_t)
            Observation_t = Environment(Action_t)
            context_{t+1} = context_t + Thought_t + Observation_t
        """
        context = f"问题: {question}\n"
        
        for step in range(self.max_steps):
            # 1. Thought + Action: 调用LLM进行推理和决策
            prompt = self._build_system_prompt() + "\n" + context
            response = self.llm.generate(prompt)
            
            # 2. 解析输出
            parsed = self._parse_response(response)
            self.trajectory.append(parsed)
            
            print(f"\n--- 第 {step + 1} 步 ---")
            print(f"Thought: {parsed['thought']}")
            print(f"Action: {parsed['action']}")
            
            # 3. 检查是否完成
            if parsed["action"] == "Finish":
                print(f"✅ 完成！答案: {parsed['action_input']}")
                return parsed["action_input"]
            
            # 4. Observation: 执行工具调用
            if parsed["action"] in self.tools:
                try:
                    # 解析JSON参数
                    args = json.loads(parsed["action_input"]) if parsed["action_input"] else {}
                    result = self.tools[parsed["action"]](**args)
                    observation = f"Observation: {result}"
                except Exception as e:
                    observation = f"Observation: 错误 - {str(e)}"
            else:
                observation = f"Observation: 未知工具 '{parsed['action']}'"
            
            print(f"{observation}")
            
            # 5. 更新上下文，进入下一轮
            context += f"Thought: {parsed['thought']}\n"
            context += f"Action: {parsed['action']}\n"
            context += f"Action Input: {parsed['action_input']}\n"
            context += f"{observation}\n"
        
        return "达到最大步数，任务未完成"


# 使用示例
def search(query: str) -> str:
    """搜索互联网获取信息"""
    # 实际实现中调用真实的搜索API
    return f"关于 '{query}' 的搜索结果: ..."

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

agent = ReActAgent(
    llm=your_llm,
    tools={"search": search, "calculate": calculate},
    max_steps=5
)

result = agent.run("2024年诺贝尔物理学奖得主是谁？")
```

---

## 二、Plan-and-Solve——先规划，后执行

### 2.1 核心思想：解决CoT的“漏步”问题

2023年，Lei Wang等人提出了**Plan-and-Solve（PS）Prompting**。其核心动机是解决思维链在处理多步骤复杂问题时容易出现的 **“漏步错误”（missing-step errors）** 。

CoT和ReAct虽然强大，但它们都遵循一种 **“边走边看”** 的线性模式——每一步决策都基于上一步的结果。这种模式在简单任务上表现良好，但在复杂任务中容易导致“只见树木不见森林”——模型可能在一个细枝末节上浪费大量时间，或者遗漏了某个关键步骤。

Plan-and-Solve的解决思路非常直观：**先将整个任务分解为一系列子任务（规划阶段），然后按计划逐个执行（求解阶段）** 。

### 2.2 两阶段架构

Plan-and-Solve将任务处理明确地分为两个解耦的阶段：

**第一阶段：规划（Planning）**

在这一阶段，Agent不对具体问题进行求解，而是制定一个高层次的行动计划。这个计划将整个任务分解为若干逻辑连贯的子目标（subgoals）。

规划阶段的Prompt模板通常是：

> “Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.”

**第二阶段：求解（Solving）**

在这一阶段，Agent严格按照第一阶段制定的计划逐步执行。每一步执行都聚焦于一个特定的子目标，避免了在无关方向上的探索。

为了进一步提高求解质量，研究者还提出了**PS+** 变体，加入了更详细的指令：

> “Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then...”

### 2.3 数学建模：任务分解为子目标

从数学角度看，Plan-and-Solve的核心是将一个复杂的目标函数分解为若干子目标的组合。

设原始任务为 $G$，规划阶段将其分解为子目标序列 $(g_1, g_2, ..., g_n)$，使得：

$$G = g_1 \circ g_2 \circ ... \circ g_n$$

其中 $\circ$ 表示任务的组合（顺序执行）。

每个子目标 $g_i$ 的求解可以形式化为：

$$g_i = \arg\max_{a} P(a \mid \text{plan}, \text{context}_{i-1})$$

其中 $\text{context}_{i-1}$ 包含了之前所有子任务的执行结果。

这种分解的关键优势在于**降低了每一步的决策复杂度**——Agent不需要在每一步都重新考虑全局目标，只需要专注于当前子任务。

### 2.4 成本控制：无细粒度反馈的高效策略

Plan-and-Solve的一个重要特点是**成本效率**。与ReAct每步都需要与环境交互不同，Plan-and-Solve的规划阶段完全不涉及外部交互，只在执行阶段才逐步求解。

实验数据显示，Plan-and-Solve在纯数值任务上提供了**最佳的“准确率-成本”比**——在FinQA数据集上比ReAct提高了2.8个百分点，而token消耗仅为思维树的七分之一左右。

这意味着**Plan-and-Solve特别适合那些“步骤清晰、可预测”的任务**——比如数学计算、结构化数据处理、标准操作流程等。对于这类任务，预先规划可以避免ReAct式的“试错”带来的额外token开销。

### 2.5 代码实现：Plan-and-Solve的核心逻辑

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SubTask:
    """子任务的数据结构"""
    id: int
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Any = None

class PlanAndSolveAgent:
    """
    Plan-and-Solve Agent
    
    两阶段架构：
    1. 规划阶段：将复杂任务分解为子任务序列
    2. 求解阶段：按计划逐步执行每个子任务
    """
    
    def __init__(self, llm, tools: Dict[str, callable]):
        self.llm = llm
        self.tools = tools
        self.plan: List[SubTask] = []
    
    def _plan_phase(self, question: str) -> List[SubTask]:
        """
        规划阶段：将任务分解为子目标
        
        数学上，这等价于求解：
        plan = argmax_{p} P(p | question)
        其中p是任务分解方案
        """
        prompt = f"""
        请将以下复杂任务分解为3-5个逻辑连贯的子任务。
        
        任务: {question}
        
        请按以下格式输出计划：
        步骤1: [子任务描述]
        步骤2: [子任务描述]
        ...
        
        注意：每个子任务应该是独立、可执行的，且步骤之间应有明确的依赖关系。
        """
        
        response = self.llm.generate(prompt)
        return self._parse_plan(response)
    
    def _parse_plan(self, response: str) -> List[SubTask]:
        """解析LLM生成的计划"""
        tasks = []
        lines = response.strip().split('\n')
        for line in lines:
            if line.strip().startswith(('步骤', 'Step')):
                # 提取步骤编号和描述
                parts = line.split(':', 1) if ':' in line else line.split('）', 1)
                if len(parts) > 1:
                    desc = parts[1].strip()
                    tasks.append(SubTask(
                        id=len(tasks) + 1,
                        description=desc,
                        status="pending"
                    ))
        return tasks
    
    def _solve_phase(self, question: str) -> str:
        """
        求解阶段：按计划逐步执行
        
        数学上，这等价于：
        for each subgoal g_i in plan:
            result_i = Execute(g_i, context_{i-1})
            context_i = context_{i-1} + result_i
        """
        context = f"原始任务: {question}\n"
        context += "执行计划:\n"
        for task in self.plan:
            context += f"- {task.description}\n"
        context += "\n现在开始逐步执行:\n"
        
        for i, task in enumerate(self.plan):
            task.status = "in_progress"
            
            # 执行当前子任务
            prompt = f"""
            {context}
            
            当前正在执行步骤 {task.id}: {task.description}
            请完成这个子任务，并输出结果。
            """
            
            response = self.llm.generate(prompt)
            task.result = response
            task.status = "completed"
            
            context += f"\n步骤 {task.id} 结果: {response}\n"
        
        # 综合所有子任务的结果，生成最终答案
        final_prompt = f"""
        基于以下各步骤的执行结果，请给出最终答案。
        
        {context}
        """
        return self.llm.generate(final_prompt)
    
    def run(self, question: str) -> str:
        """运行Plan-and-Solve主流程"""
        # 阶段1: 规划
        print("📋 规划阶段: 分解任务...")
        self.plan = self._plan_phase(question)
        for task in self.plan:
            print(f"  {task.id}. {task.description}")
        
        # 阶段2: 求解
        print("\n⚙️ 求解阶段: 逐步执行...")
        return self._solve_phase(question)
```

---

## 三、Reflection / Reflexion——通过自我批评实现迭代优化

### 3.1 从“试错”到“反思”：Agent的学习机制

ReAct和Plan-and-Solve虽然强大，但它们都有一个共同的局限：**每次任务都是“从零开始”，无法从过去的错误中学习**。如果一个Agent在某个任务上失败了，下一次遇到类似任务时，它仍然会犯同样的错误。

人类解决复杂问题的方式则完全不同——我们通过**试错和反思**来学习。当我们失败时，我们会思考“我哪里做错了”“下次应该如何改进”，并将这些反思应用于未来的尝试。

2023年，Noah Shinn等人提出了**Reflexion**框架。其核心思想极具洞察力：**不是通过更新模型权重来强化Agent，而是通过语言反馈（linguistic feedback）** 。

> Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials.

用更通俗的话说：**Reflexion让Agent把自己的失败经验写成“反思日记”，在下一次尝试前先读一遍**。

### 3.2 批评-修正机制（Critic-Revise）

Reflection/Reflexion的核心机制可以概括为 **“生成器-批评器-修正器”** 的三元循环：

**第一步：生成（Generate）** ——生成器（Generator）根据用户请求生成初始输出。

**第二步：批评（Critique）** ——批评器（Critic）评估生成器的输出，指出其中的错误、遗漏或可改进之处。批评可以是：
- **自我批评**：同一个模型既当生成器又当批评器
- **外部批评**：使用另一个模型或规则系统进行评审

**第三步：修正（Revise）** ——生成器根据批评器的反馈，修订并生成改进后的输出。

这三个步骤可以反复迭代，直到输出质量达到满意标准。

### 3.3 数学建模：Reflexion作为语义梯度

Reflexion最精妙之处在于其对“学习”的重新定义。传统的强化学习通过**策略梯度**来更新模型参数：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right]$$

这需要大量的训练样本和昂贵的模型微调。

Reflexion则完全不同。它将“梯度”从**参数空间**转移到了**语义空间**。设：

- $\tau$ 为失败的轨迹
- $f: \tau \rightarrow r$ 为反思生成函数（由LLM实现）
- $r$ 为反思文本

Reflexion的“语义梯度”可以形式化为：

$$\Delta \pi \approx \text{LLM}(\text{prompt} \oplus r)$$

即：反思文本 $r$ 被追加到提示中，从而在不改变模型权重的情况下“调整”了策略。

从信息论角度看，Reflexion相当于在**提示空间**中执行优化：

$$\text{prompt}_{t+1} = \text{prompt}_t \oplus \text{Reflection}(\text{trajectory}_t)$$

这种方法的美妙之处在于：**它利用了LLM本身的语义理解能力来实现“学习”，而不需要任何参数更新**。

### 3.4 实验数据：Reflexion的强大效果

Reflexion的实验结果令人印象深刻：

- 在**HumanEval代码生成基准**上，Reflexion达到了**91%的pass@1准确率**，超越了此前GPT-4的80%
- 在**AlfWorld决策任务**上，成功率达到了**97%**
- 在**HotPotQA问答任务**上，成功率为**51%**

这些数据证明了“语言反馈”这一范式的有效性。

### 3.5 迭代次数控制：何时停止反思

反思虽然强大，但**无限迭代是不现实的**——每次反思都会消耗token和时间。因此，迭代次数控制是一个关键的工程问题。

实践中常用的停止策略包括：

**固定迭代次数**：设置一个最大迭代次数（通常为3-5轮），达到后自动停止。这是最简单也最常用的策略。

**置信度阈值**：让Agent对自己的输出进行置信度评分（如1-10分），达到阈值（如8分）后停止。

**外部验证**：使用一个独立的验证Agent来评估输出质量，通过时停止。

**边际收益递减检测**：当连续两轮迭代的改进幅度低于某个阈值时停止。

在实际生产中，**建议设置3-5次反思循环**，以平衡质量提升与成本开销。

### 3.6 代码实现：Reflection Agent

```python
from typing import List, Dict, Any, Optional

class ReflectionAgent:
    """
    Reflection Agent: 通过自我批评和迭代修正来优化输出
    
    核心循环: Generate → Critique → Revise (重复直到满意)
    """
    
    def __init__(self, llm, max_iterations: int = 3, confidence_threshold: float = 8.0):
        """
        初始化Reflection Agent
        
        Args:
            llm: 大语言模型接口
            max_iterations: 最大迭代次数
            confidence_threshold: 置信度阈值（1-10），达到后停止
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.reflection_history: List[str] = []
    
    def _generate(self, task: str, context: str = "") -> str:
        """生成阶段：产生初始输出"""
        prompt = f"""
        任务: {task}
        
        {context}
        
        请生成你的回答。
        """
        return self.llm.generate(prompt)
    
    def _critique(self, task: str, output: str) -> str:
        """
        批评阶段：评估输出质量，指出问题
        
        数学上，这等价于：
        critique = argmax_c P(c | task, output)
        其中c是批评文本
        """
        prompt = f"""
        任务: {task}
        
        生成的回答:
        {output}
        
        请以批评者的身份评估这个回答。指出：
        1. 回答中存在的错误或不准确之处
        2. 遗漏的重要信息
        3. 可以改进的地方
        
        批评:
        """
        return self.llm.generate(prompt)
    
    def _revise(self, task: str, output: str, critique: str) -> str:
        """
        修正阶段：根据批评改进输出
        
        数学上，这等价于：
        revised = argmax_r P(r | task, output, critique)
        """
        prompt = f"""
        任务: {task}
        
        原始回答:
        {output}
        
        批评意见:
        {critique}
        
        请根据上述批评意见，修订并生成改进后的回答。
        """
        return self.llm.generate(prompt)
    
    def _evaluate_confidence(self, task: str, output: str) -> float:
        """评估对当前输出的置信度"""
        prompt = f"""
        任务: {task}
        
        回答: {output}
        
        请对这个回答的质量进行评分（1-10分），只返回数字。
        """
        response = self.llm.generate(prompt)
        try:
            return float(response.strip())
        except:
            return 5.0
    
    def run(self, task: str) -> str:
        """
        运行Reflection主循环
        
        迭代过程:
        for i in range(max_iterations):
            output = Generate(task, context)
            critique = Critique(task, output)
            if quality_sufficient(output):
                break
            output = Revise(task, output, critique)
        """
        print(f"📝 任务: {task}\n")
        
        # 初始生成
        current_output = self._generate(task)
        print(f"--- 初始生成 ---\n{current_output}\n")
        
        for iteration in range(self.max_iterations):
            # 1. 批评
            critique = self._critique(task, current_output)
            print(f"--- 第 {iteration + 1} 轮批评 ---\n{critique}\n")
            self.reflection_history.append(critique)
            
            # 2. 评估置信度
            confidence = self._evaluate_confidence(task, current_output)
            print(f"置信度评分: {confidence}/10")
            
            # 3. 检查是否应该停止
            if confidence >= self.confidence_threshold:
                print("✅ 置信度达标，停止迭代")
                break
            
            # 4. 修正
            current_output = self._revise(task, current_output, critique)
            print(f"--- 第 {iteration + 1} 轮修正后的输出 ---\n{current_output}\n")
        
        return current_output
```

---

## 四、三种范式的对比与选择

### 4.1 核心差异

| 维度 | ReAct | Plan-and-Solve | Reflection |
|------|-------|----------------|------------|
| **核心机制** | 推理与行动交替 | 先规划后执行 | 生成-批评-修正循环 |
| **决策模式** | 在线、自适应 | 离线、结构化 | 迭代、自优化 |
| **外部交互** | 每步都可能 | 主要在执行阶段 | 可选（可纯内部） |
| **错误处理** | 动态纠错 | 规划避免 | 迭代修正 |
| **成本特征** | 中等 | 低（无细粒度反馈） | 较高（多轮） |
| **适用场景** | 开放域、不确定环境 | 结构化、可预测任务 | 内容生成、质量敏感任务 |

### 4.2 如何选择

**选择ReAct**：当任务环境**不可预测**、需要动态调整策略时。例如：网络搜索、多步问答、客户服务对话。

**选择Plan-and-Solve**：当任务**步骤清晰、可预先规划**时。例如：数学计算、数据处理、标准操作流程。

**选择Reflection**：当**输出质量是关键**、可以接受多轮迭代时。例如：代码生成、内容创作、报告撰写。

### 4.3 混合架构：取长补短

在实际生产系统中，这三种范式往往不是互斥的，而是可以**组合使用**。例如：

- **Plan-and-Solve + ReAct**：先用Plan-and-Solve制定整体计划，然后在每个子任务的执行中使用ReAct进行动态调整
- **ReAct + Reflection**：ReAct负责与环境交互收集信息，Reflection负责对收集到的信息进行批判性分析和整合
- **三者结合**：规划 → 执行（ReAct式） → 反思优化

---

## 五、总结：从线性到循环，从单次到迭代

回顾这三种范式的演化，我们可以清晰地看到一条**能力递增**的路径：

**ReAct**让Agent从“一次性推理”进化到“闭环推理”——它可以在推理过程中获取外部信息并动态调整。

**Plan-and-Solve**让Agent从“边走边看”进化到“先看路再走”——它通过预先规划避免了在复杂任务中迷失方向。

**Reflection**让Agent从“每次从零开始”进化到“从错误中学习”——它通过自我批评和迭代修正实现了持续的自我优化。

这三种范式共同构成了当前AI Agent能力的基础框架。理解它们，就是理解了如何让AI从“会说话”进化到“会思考、会行动、会学习”。

正如ReAct论文中所说：“推理轨迹帮助模型诱导、跟踪和更新行动计划并处理异常，而行动允许它与外部来源交互以收集额外信息。”这三者——推理、行动、反思——正是智能的本质要素。