---
name: 第四阶段 - Agent 框架
description: 用框架搭建多步骤 Agent，理解手写 Agent loop 与框架 Agent 的差异
type: project
---

# 第四阶段：Agent 框架

> 预计耗时：2-3周 | 开始日期：____ | 完成日期：____
> 前置：第一至第三阶段全部完成
> 关联画像：ai_learning_profile.md

---

## 为什么用框架

第一阶段你可能已经"手写"了一个简单的 agent loop（调 API → 判断是否要调工具 → 执行工具 → 回传 → 回复）。框架的价值在于：
- 封装了通用的 agent 模式（ReAct、Plan-and-Execute 等）
- 提供工具注册、记忆管理、错误处理的标准化方式
- 方便组合多个 Agent 实现复杂流程

但框架也有代价：复杂度高、调试困难、过度封装。**本阶段的目标是理解框架做了什么，而不是成为某个框架的专家。**

---

## Step 1：LangChain 基础

**学习目标**：了解 LangChain 的核心概念，跑通第一个 LangChain 应用

**操作建议**：
1. 安装：`pip install langchain langchain-openai langgraph`
2. 核心概念理解：
   - **PromptTemplate**：带变量的 prompt 模板
   - **ChatModel**：统一封装不同 LLM 的调用
   - **Runnable**：LangChain 的执行抽象（类似 Unix pipe）
   - **Memory**：对话历史管理
3. 实现：用 LangChain 重写第一阶段的"CLI 智能助手"（不用 LangChain 的 Agent，只是用它的组件）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手，回答不超过50字。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "Python 中列表和元组有什么区别？"})
print(result)
```

**Milestone 1**：
- [x] 用 LangChain 的 chain 完成一次对话
- [x] 理解 `|` 操作符的作用（类似 Unix pipe 的链式调用）
- [x] 能用 LangChain 管理对话历史

---

## Step 2：LangChain 的 Tool 与 Agent

**学习目标**：用 LangChain 的工具系统和 Agent 模式实现多工具调度

**操作建议**：
1. 学习 LangChain 的 Tool 定义方式：
   ```python
   from langchain.tools import tool

   @tool
   def get_weather(city: str) -> str:
       """获取指定城市的天气"""
       # 模拟返回
       return f"{city}：晴天，25°C"
   ```
2. 用 LangGraph 创建 Agent：
   ```python
   from langgraph.prebuilt import create_react_agent

   tools = [get_weather, calculate]
   agent = create_react_agent(llm, tools)
   result = agent.invoke({"messages": [("user", "北京明天天气怎样？")]})
   ```
3. 对比：手写 agent loop（第一阶段）vs LangGraph Agent
   - 代码量差异
   - 错误处理差异
   - 调试难度差异

**Milestone 2**：
- [x] 用 LangGraph 实现包含 3+ 个工具的 Agent
- [x] Agent 能正确选择和路由工具
- [x] 能画出 LangGraph 的执行流程图
- [x] 列出 3 个 LangGraph 比你手写更方便的地方

---

## Step 3：Agent 的记忆与状态管理

**学习目标**：理解 Agent 如何在多轮对话中保持状态

**操作建议**：
1. 了解 LangGraph 的状态机模型：
   - `State`：Agent 的当前状态（包含消息历史、中间变量等）
   - `Node`：状态转换的函数（LLM 调用、工具执行等）
   - `Edge`：状态之间的转换路径
2. 实现一个有状态的 Agent：
   - 记住用户偏好（"我喜欢用 Markdown 格式"）
   - 在多轮对话中使用这些偏好
3. 对比不同的记忆方式：
   - 消息历史（传所有对话）
   - 摘要记忆（只保留摘要）
   - 向量存储记忆（从向量库检索相关历史）

**Milestone 3**：
- [x] Agent 能在第 5 轮对话中记住第 1 轮的用户偏好
- [x] 能解释 State、Node、Edge 三个概念
- [x] 对比 3 种记忆方式的适用场景

---

## Step 4：多 Agent 协作

**学习目标**：实现多 Agent 协作场景，超越第一阶段的"线性理解"

**操作建议**：
1. 了解多 Agent 的几种模式：
   - **Supervisor（主管模式）**：一个主 Agent 决定分配任务给谁
   - **Debate（辩论模式）**：多个 Agent 各自回答，再由裁判 Agent 选出最好的
   - **Sequential（链式模式）**：A 的输出作为 B 的输入（你之前理解的这种）
   - **Handoff（交接模式）**：Agent A 处理不了，转给 Agent B
2. 用 LangGraph 实现 Supervisor 模式：
   - Supervisor Agent 接收用户请求
   - 根据意图分配给 Writer、Researcher、CodeReviewer 等子 Agent
   - 子 Agent 执行后结果返回给 Supervisor
3. 对比单 Agent 和多 Agent 的效果差异

```
用户: "帮我写一个 Python 函数，计算斐波那契数列，然后审查这段代码的质量"

Supervisor:
  ├─ 分配给 Writer: 写斐波那契函数
  ├─ 分配给 CodeReviewer: 审查代码
  └─ 汇总两个结果，回复用户
```

**Milestone 4**：
- [x] 实现至少 2 种多 Agent 模式（推荐 Supervisor + Sequential）
- [x] 对比单 Agent 和多 Agent 在同一个任务上的表现
- [x] 能说出多 Agent 相比单 Agent 的真正优势（不仅仅是"先后调用"）
- [x] 能说出多 Agent 的缺点（成本、延迟、复杂度、Agent 之间可能互相误导）

---

## Step 5：手写 Agent vs 框架 Agent 对比

**学习目标**：深入理解框架封装了什么，不依赖框架也能写出 Agent

**操作建议**：
1. 不依赖 LangChain，手写一个 ReAct 风格的 Agent：
   - Thought → Action → Observation 循环
   - 支持多个工具
   - 有最大迭代次数限制
2. 和 LangGraph 的实现对比：
   - 代码量
   - 可读性
   - 可调试性
   - 扩展性
3. 理解 LangGraph 底层做了什么：本质上也是 ReAct loop + 状态管理

```python
# 手写 ReAct Agent 伪代码
def react_agent(question, tools, max_steps=5):
    messages = [("system", "你是一个助手，可以用工具解决问题。每一步先 Thought，再 Action。")]
    messages.append(("user", question))

    for step in range(max_steps):
        # LLM 决定下一步
        response = llm(messages, tools=tools)
        
        if response.has_tool_calls:
            # 执行工具
            result = execute_tool(response.tool_calls)
            messages.append(("tool", result))
        else:
            # 直接回复
            return response.content
    
    return "达到最大迭代次数，无法完成。"
```

**Milestone 5**：
- [ ] 手写 ReAct Agent 能正确完成 3+ 个工具的调度
- [ ] 画出手写 Agent 和 LangGraph Agent 的架构对比图
- [ ] 能解释框架的哪些封装是真正有价值的，哪些是过度设计

---

## Step 6：Agent 的错误处理与健壮性

**学习目标**：让 Agent 在面对异常情况时优雅降级

**操作建议**：
1. 测试以下异常情况：
   - 工具调用失败（网络超时、返回错误）
   - LLM 返回了不存在的工具名
   - LLM 在应该调工具时直接回复
   - LLM 陷入循环（反复调同一个工具）
2. 实现对应的处理策略：
   - 工具失败后重试 / 降级回复
   - 工具名不匹配时给出可用工具列表
   - 循环检测（检查是否重复调用同一工具同一参数）
3. 了解 LangGraph 的 built-in 错误处理机制

**Milestone 6**：
- [ ] 对 5 种异常场景都有处理策略
- [ ] Agent 在异常时不会 crash，而是给用户有用的提示
- [ ] 有异常处理日志记录

---

## 阶段验收

### 综合项目：多工具多步骤 Agent

实现一个能完成以下任务的 Agent：

1. **代码助手场景**：
   - 用户给一段代码 → Agent 自动审查（bug、性能、安全）
   - 如果用户问"怎么修" → Agent 给出修改建议
   - 如果用户说"帮我改" → Agent 输出修改后的代码
2. 至少 3 个工具：code_review、suggest_fix、rewrite_code
3. 支持多轮对话
4. 有工具调用失败时的降级处理
5. 能画出完整的 Agent 执行流程图

**验收标准**：
- [ ] Agent 能完成"审查 → 建议 → 修改"的完整流程
- [ ] 工具选择正确率 >= 80%
- [ ] 有异常处理和日志
- [ ] 能解释 Agent 的每一步决策
- [ ] 代码结构清晰，手写版和框架版都有（至少各一个）

---

## 常见陷阱

- **框架不是银弹**：LangGraph 很好用，但理解不了底层原理就调不好
- **过度设计**：一开始就搞多 Agent，其实单 Agent 可能就够了
- **忽视调试**：Agent 的行为不确定性强，必须有好的日志和 trace
- **工具描述不清**：Agent 选错工具，很多时候是 tool description 写得太模糊
- **循环问题**：Agent 可能陷入"调工具 → 不满意 → 再调同一个工具"的死循环，必须有最大步数限制
