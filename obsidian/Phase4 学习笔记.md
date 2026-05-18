# Phase 4: Agent 框架 学习笔记

> 用框架搭建多步骤 Agent，理解手写 Agent loop 与框架 Agent 的差异

---

## Step 1: LangChain 基础

### LangChain 核心概念

#### LCEL（LangChain Expression Language）

LCEL 是 LangChain 的链式调用语法，用 `|` 操作符串联组件，类似 Unix pipe：

```python
chain = prompt | llm | parser
```

数据流：输入 dict → prompt 渲染 → LLM 调用 → parser 提取 → 最终输出。

#### PromptTemplate

带变量的 prompt 模板，`ChatPromptTemplate.from_messages()` 接收消息列表：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "系统提示词"),
    ("placeholder", "{chat_history}"),  # 动态插入历史消息
    ("human", "{question}"),            # 用户输入变量
])
```

- `"placeholder"` 是一个特殊占位符，用于动态插入消息列表（如对话历史）
- `"human"` 和 `"ai"` 对应角色，`"system"` 是系统提示

#### ChatModel

LangChain 统一封装了不同 LLM 的调用。通过 `langchain_openai.ChatOpenAI` 可以对接任何 OpenAI 兼容的 API：

```python
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
```

#### OutputParser

提取 LLM 的回复内容。最常用的是 `StrOutputParser()`，把 `AIMessage` 对象转为纯字符串。

#### RunnableParallel

并行执行多个 branch，结果合并为 dict：

```python
parallel = RunnableParallel(formal=formal_chain, casual=casual_chain)
result = parallel.invoke({"question": "什么是 API？"})
# result["formal"] 和 result["casual"] 同时执行
```

### 对话历史管理

LangChain 管理对话历史的核心机制：

1. 在 prompt 中用 `"placeholder"` 占位 `{chat_history}`
2. 调用时传入历史消息列表：`chain.invoke({"chat_history": history, "question": q})`
3. 每轮对话后用 `HumanMessage` 和 `AIMessage` 追加到历史列表

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []
chat_history.append(HumanMessage(content="用户说的话"))
chat_history.append(AIMessage(content="AI 的回复"))
```

### 遇到的问题

**Windows 控制台编码问题**：LLM 返回包含 emoji 的内容时，Windows GBK 编码会报错 `UnicodeEncodeError`。修复方法是在脚本开头重定向 stdout 为 UTF-8：

```python
import io, sys
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

### 与 Phase 1 的对比

| 维度 | Phase 1 手写 | LangChain |
|------|-------------|-----------|
| LLM 调用 | 直接构造 HTTP 请求或用 openai 库 | `ChatOpenAI` 统一封装 |
| Prompt 管理 | 字符串拼接 | `ChatPromptTemplate` + 变量 |
| Chain 串联 | 手动按顺序调用 | `|` 操作符一键串联 |
| 历史管理 | 自己维护 list | `HumanMessage`/`AIMessage` + placeholder |

LangChain 的价值在于把常用模式标准化了，减少了"拼接字符串 → 调 API → 解析 JSON"的样板代码。

### LangChain 是"模板语言"——不是底层 API 规则

Step 1 跑完后产生了一个重要认知：**LangChain 的所有语法约定都是它自己定义的一层模板语言，不是底层 API 的规则。**

- `"placeholder"` 是 `ChatPromptTemplate` 独有的，OpenAI/Dashscope 的 API 里没有这个概念——底层 API 只认一个消息列表，直接传进去就行
- `"human"` / `"ai"` 是 LangChain 起的别名，内部会自动转成 `"user"` / `"assistant"` 再发给 API
- `"placeholder"` 和普通 `"human"` 的关键区别：placeholder 接收**消息列表**（`List[BaseMessage]`），会原样展开插入多条消息；`"human"` 只接收字符串，只创建一条 HumanMessage

**本质**：LangChain 在底层 API 之上包了一层模板语言，学 LangChain 就是学这套规则。理解这一点有助于避免把 LangChain 的约定误认为是 LLM API 本身的机制。

#### RunnableLambda：在 chain 中间插入自定义逻辑

用 `RunnableLambda` 把普通函数包装成 `Runnable`，就能串入 chain：

```python
from langchain_core.runnables import RunnableLambda

upper_chain = (
    RunnableLambda(lambda x: {"question": x["question"].upper()})
    | prompt
    | llm
    | StrOutputParser()
)
```

数据流：输入 → 转大写 → prompt 渲染 → LLM → 解析输出。这是扩展 chain 行为的标准方式，比如可以在中间加缓存、日志、输入校验等。

---

## Step 2: LangChain 的 Tool 与 Agent

### Tool 定义方式

用 `@tool` 装饰器定义工具，框架会自动从函数签名和 docstring 生成 schema：

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。参数 city 为城市名称。"""
    return "晴天，25°C"
```

Agent 能看到的工具信息：
- **函数名**：`get_weather`
- **参数 schema**：从 `city: str` 推断
- **描述**：从 docstring 提取

不需要手写 JSON schema 或手动描述参数，这是相比手写 Agent 的第一个方便之处。

### ReAct Agent

ReAct（Reasoning + Acting）是一种 Agent 模式，核心循环：
**Thought → Action → Observation → Thought → ... → Final Answer**

LangGraph 提供 `create_agent`（原 `create_react_agent`）一键创建：

```python
from langchain.agents import create_agent  # 注意：新版从 langchain.agents 导入

tools = [get_weather, calculate, search_knowledge]
agent = create_agent(llm, tools)
result = agent.invoke({"messages": [("user", "北京天气怎么样？")]})
```

### 实验观察

**多工具组合能力**：
- 输入 "上海和深圳的天气哪个更好？我想算一下 (25 + 26) / 2 的平均温度"
- Agent **自动并行调用**了 `get_weather`（两次，上海和深圳）和 `calculate`
- 最终回复综合了三个工具的结果

**执行过程可视化**（Demo 2 输出）：
```
[User] 杭州天气如何？然后算一下 30 + 25 是多少
[AI → Tool] 调用: ['get_weather', 'calculate']
[Tool Result] 晴天，30°C，湿度 55%，西南风 1 级
[Tool Result] 30 + 25 = 55
[AI] 杭州当前天气为晴天...而 30 + 25 的计算结果是 55。
```

Agent 先思考需要哪些工具 → 同时调用 → 拿到结果后综合回复。

### LangGraph 执行流程图

通过 `agent.get_graph().draw_ascii()` 可以查看 Agent 的执行图结构。

**Step 2 — `create_agent` 自动生成的 ReAct Agent**：

```
           __start__
               │
               ▼
            ┌──────┐
            │model │  ← LLM 调用节点
            └──────┘
            ╱      ╲
          ╱          ╲
    ┌──────┐     +-------+
    │__end__│     │ tools │  ← 工具执行
    └──────┘     +-------+
                    │
                    └────→ 回到 model (隐式循环)
```

流程：`__start__ → model → (需要工具? → tools → model) → __end__`

**Step 3 — 手写 `StateGraph` 有状态 Agent**：

```
           __start__
               │
               ▼
            ┌──────┐
            │ llm  │  ← 自定义 LLM 节点（注入用户偏好）
            └──────┘
            ╱      ╲
          ╱          ╲
    ┌──────┐     +-------+
    │__end__│     │ tools │  ← 工具执行
    └──────┘     +-------+
                    │
                    └────→ 回到 llm
```

流程：`START → llm → should_continue 判断 → (tools → llm) / END`

两个图结构一样，区别在于：
- `create_agent`：边和路由都是框架内置的，黑盒开箱即用
- `StateGraph`：手动定义每个节点和边，完全可控（如注入用户偏好、条件路由等）

### LangGraph vs 手写 Agent 对比

| 维度 | 手写 Agent Loop | LangGraph Agent |
|------|-----------------|-----------------|
| 工具描述 | 手动构造 JSON schema + 描述文字 | `@tool` 自动生成 |
| ReAct 循环 | 手写 for 循环 + 判断 | `create_agent` 内置 |
| 工具调用解析 | 自己解析 LLM 返回的 JSON | 框架自动解析和执行 |
| 消息格式 | 手动构造 `{"role": "tool", ...}` | `ToolMessage` 对象 |
| 代码量 | ~100-150 行 | ~10 行（不含工具定义）|

**LangGraph 真正有价值的 3 个点**：
1. **Tool schema 自动生成**——函数签名和 docstring 直接转为 LLM 能理解的格式，不需要手写描述
2. **ReAct loop 内置**——"调用→执行→回传"的循环逻辑不需要自己写，避免边界情况 bug
3. **消息格式统一**——不需要自己维护 `role: "tool"`, `tool_call_id` 等字段

### 遇到的问题

**LangGraph API 迁移**：`create_react_agent` 已从 `langgraph.prebuilt` 迁移到 `langchain.agents.create_agent`，旧 API 会报 `LangGraphDeprecatedSinceV10` 警告。代码已更新使用新 API。

---

## Step 3: Agent 的记忆与状态管理

### LangGraph 状态机模型：State、Node、Edge

LangGraph 用**状态图（State Graph）**的方式管理 Agent 的执行流程，三个核心概念：

- **State（状态）**：一个包含所有 Agent 数据的字典，核心是 `messages` 字段（对话历史），也支持自定义字段（如 `user_preferences`）。`add_messages` 是一个 reducer，负责安全地合并消息列表（按 ID 去重追加）。
- **Node（节点）**：状态转换的函数，比如 `call_llm`（调用 LLM）、`ToolNode`（执行工具）。每个 node 接收当前 state，返回需要更新的字段。
- **Edge（边）**：定义 state 如何从一个 node 流向另一个 node。`add_edge` 是固定路径，`add_conditional_edges` 是根据条件动态选择下一节点（如判断是否需要调工具）。

```python
graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", ToolNode([get_weather, calculate]))
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "llm")
graph = graph_builder.compile()
```

执行流程：`START → llm → (需要工具? → tools → llm) → END`

### 三种记忆方式

| 维度 | 完整消息历史 | 摘要记忆 | 向量检索记忆 |
|------|-------------|---------|-------------|
| 实现难度 | 最简单 | 中等 | 最复杂 |
| Token 消耗 | 线性增长 | 固定（摘要） | 固定（top-k） |
| 上下文完整性 | 100% 保留 | 有信息损失 | 仅保留相关内容 |
| 长期记忆能力 | 差（超限即丢） | 中（摘要会漂移） | 好（持久化存储） |
| 适用场景 | 短对话 | 中等长度对话 | 长对话/知识库 |

**摘要记忆实现**：用 LLM 对早期对话生成摘要，保留最近 2 条原始消息。实验结果：8 条消息压缩为 3 条（1 条摘要 + 2 条最近消息）。

**向量检索记忆原理**：将历史对话分段存入向量数据库（Chroma），用户新提问时检索最相关片段作为 context 注入 prompt，而非传入完整历史。适合长对话/知识库场景。

### LangGraph 状态图的存储形式

StateGraph 的图结构本身**存在于内存中的 Python 对象**，不是文件：

1. **图结构（编译前）**：`StateGraph` 是内存中的图构建器，通过 `add_node()`、`add_edge()` 注册节点和边，`compile()` 后返回 `CompiledGraph` 对象。全程不写文件。
2. **运行状态（invoke 时）**：state 是内存中的 dict，调用完即丢失。

**图结构本身不需要持久化**——它是代码定义的（哪些节点、哪些边、条件路由逻辑），每次程序启动时通过代码重建。这和定义函数一样，源码存在，运行时自然能重建。

**需要持久化的是运行时的状态数据**：对话历史、用户偏好、中间变量等。Checkpointer 存到数据库的不是"图"，而是**这张图在某个 `thread_id` 下运行到的状态快照**，大致结构：

```
checkpoint 表:
  thread_id="abc123"  node="llm"   state={messages: [...], user_preferences: {...}}
  thread_id="abc123"  node="tools" state={messages: [...]}
```

重启后的恢复流程：代码重建图结构 → checkpointer 按 `thread_id` 从数据库加载最新状态快照 → 从中断处继续执行。

**如需持久化**（中断恢复、时间旅行），需配置 **checkpointer**：

| Checkpointer | 存储位置 | 用途 |
|-------------|---------|------|
| `MemorySaver` | 内存 dict | 进程内持久，重启丢失 |
| `SqliteSaver` | 本地 SQLite 文件 | 跨重启持久化 |
| `PostgresSaver` | PostgreSQL 数据库 | 生产级持久化 |

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "abc123"}})
```

`thread_id` 用于区分不同的对话线程，checkpointer 会按 thread_id 保存状态快照。

> **注意**：LangGraph checkpointer 存的是**对话状态快照**，Phase 3 学的 Chroma 向量库存的是**知识数据**——两者是不同层面的东西。

### 遇到的问题

**LangGraph 状态传递 bug**：最初实现时，每轮对话 Agent 都返回同样的回复。原因是把完整消息历史 + 新消息一起 `invoke`，导致 LLM 每次都看到完全一样的上下文（包含之前的 AI 回复），无法区分"已处理"和"新消息"。修复方案是让 `call_llm` 节点检查最后一条消息是否已经是完成的 AI 回复，如果是则跳过处理。

---

## Step 4: 多 Agent 协作

### 多 Agent 模式

#### Sequential 链式模式

固定的线性流程，A 的输出作为 B 的输入，B 的输出作为 C 的输入：

```python
graph.add_edge(START, "Writer")
graph.add_edge("Writer", "Reviewer")
graph.add_edge("Reviewer", "Summarizer")
graph.add_edge("Summarizer", END)
```

适合：流程固定的任务，不需要动态路由。

#### Supervisor 主管模式

一个 Supervisor 节点根据当前状态决定下一步跳转到哪个子 Agent：

```python
graph.add_edge(START, "Supervisor")
graph.add_conditional_edges("Supervisor", route_supervisor, {...})
# 子 Agent 执行完后都回到 Supervisor
graph.add_edge("Writer", "Supervisor")
graph.add_edge("Reviewer", "Supervisor")
graph.add_edge("Summarizer", "Supervisor")
```

关键设计点：
- **路由函数**：根据 state 中的字段（如 `writer_output` 是否存在）判断下一步
- **安全限制**：必须加最大步数计数器，防止 Supervisor 陷入无限循环
- **状态更新**：节点函数只返回需要更新的字段（如 `{"writer_output": "..."}`），不是完整 state

### 多 Agent vs 单 Agent

| 维度 | 单 Agent | 多 Agent (Supervisor) |
|------|---------|---------------------|
| LLM 调用次数 | 1 次 | 3-4 次 |
| 响应延迟 | 低（~5s） | 高（~15-20s） |
| Token 成本 | 低 | 高（3-4 倍） |
| 专业化程度 | 通用，可能都不精 | 每个 Agent 专精一项 |
| 审查独立性 | 自己写自己审，可能忽略 bug | 独立 Agent 审查，更客观 |
| 可维护性 | 改 prompt 影响全局 | 每个 Agent 独立修改 |

### 多 Agent 的真正优势

1. **关注点分离**：每个 Agent 有独立 system prompt，专注一个领域
2. **独立审查**：不同 prompt 角度的审查比"自己审自己"更容易发现问题
3. **可组合性**：可以动态插入新 Agent（如 TestAgent），或给 Writer 分配多个 Reviewer
4. **失败隔离**：Writer 失败不影响 Reviewer，可单独重试

### 多 Agent 的缺点

1. **成本高**：3-4 倍 LLM 调用
2. **延迟大**：串行执行多个 Agent
3. **调试难**：Agent 间交互可能导致意外行为（互相误导、循环）
4. **过度设计**：简单任务用单 Agent 即可，多 Agent 反而增加复杂度
5. **Agent 之间可能互相误导**：Writer 输出错误代码，Reviewer 没看出来，Summarizer 基于错误信息汇总

### 多 Agent 可以调用不同的 LLM

每个 Agent 节点本质上就是一个 Python 函数，在函数内部调用哪个 LLM 实例完全由代码决定。给不同 Agent 注入不同的 LLM 实例即可：

```python
# 不同 Agent 使用不同模型
write_llm = ChatOpenAI(model="qwen-plus", ...)       # 普通任务
review_llm = ChatOpenAI(model="qwen-max", ...)       # 关键审查用最贵模型
summarize_llm = ChatOpenAI(model="qwen-turbo", ...)  # 简单整理用便宜模型

# 甚至可以混用不同厂商
writer_llm = ChatOpenAI(model="gpt-4o")              # OpenAI
review_llm = ChatAnthropic(model="claude-sonnet-4-6")  # Anthropic
```

**适用场景**：

| 场景 | 策略 | 原因 |
|------|------|------|
| 代码审查 | 用最强模型 | 安全性/正确性不能妥协 |
| 文本摘要 | 用便宜模型 | 任务简单，没必要花大钱 |
| 数据提取 | 用结构化输出强的模型 | 不同模型在 JSON 输出上能力差异大 |
| 降低成本 | 关键步骤贵模型，非关键便宜 | 3-4 次调用全用最贵成本太高 |
| 避免单点故障 | 不同 Agent 用不同供应商 | 一家 API 挂了另一家还能跑 |

实际生产中很常见——比如 Supervisor 用 GPT-4o 做路由，Writer 用 Claude 写代码，Summarizer 用小模型做格式化。总成本可能只有全部用 GPT-4o 的一半。

### 多 Agent 微服务架构

生产环境中，多 Agent 通常部署为**独立微服务**，每个 Agent 是一个独立的 HTTP 服务，Supervisor 通过 API 远程调用：

```
Supervisor (API 服务)
    │
    ├── POST /agent/writer ──────►  Writer 微服务 (Python, qwen-plus)
    ├── POST /agent/reviewer ────►  Reviewer 微服务 (Go, Claude)
    └── POST /agent/summarizer ──►  Summarizer 微服务 (Node.js, qwen-turbo)
```

**与单机进程内的对比**：

| 维度 | 单机进程内（学习阶段） | 微服务架构（生产环境） |
|------|----------------------|---------------------|
| 通信方式 | 函数调用，无网络延迟 | HTTP/gRPC 远程调用，增加 RTT |
| 部署 | 一个进程，部署简单 | 每个 Agent 独立部署 |
| 扩缩容 | 所有 Agent 共享资源 | 可按需独立扩缩容 |
| 技术栈 | 同一语言/框架 | 每个 Agent 可用不同语言 |
| 故障隔离 | 进程级隔离 | 服务级隔离，更彻底 |
| 调试 | 本地断点即可 | 需要分布式 tracing |

**在 LangGraph 中实现**：图结构不变，只需将节点函数从本地调用改为 HTTP 调用：

```python
def writer_agent_remote(state: dict) -> dict:
    import requests
    resp = requests.post("http://writer-svc:8080/generate", json={"task": state["current_task"]})
    return {"writer_output": resp.json()["content"]}
```

**主流框架都支持这种模式**：LangGraph 节点函数可以是任意代码（本地/远程），CrewAI 支持 Agent 通过 API 注册为远程服务，AutoGen 原生支持多进程/远程 Agent 通信。

**学习路径建议**：先在单机进程内理解概念（当前做法），再迁移到微服务架构。核心区别只是"节点函数是直接调用还是 HTTP 远程调用"，LangGraph 的图结构完全不变。

---

## Step 5: 手写 Agent vs 框架 Agent 对比

### 手写 ReAct Agent 核心逻辑

不依赖任何框架，纯 `openai` SDK + for 循环：

```python
messages = [system_msg, user_msg]
for step in range(max_steps):
    response = client.chat.completions.create(messages=messages, tools=TOOLS)
    if response.choices[0].message.tool_calls:
        # 执行工具 → 构造 tool message → 追加到 messages
        for tc in tool_calls:
            result = execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        return response.choices[0].message.content  # LLM 直接回复，结束
```

### 手写 Agent 实验结果

| 测试用例 | 迭代次数 | 耗时 | 行为 |
|---------|---------|------|------|
| 北京天气 | 2 | 6.5s | 调 get_weather → 回复 |
| 123 * 456 | 2 | 2.3s | 调 calculate → 回复 |
| 上海+深圳天气+平均温度 | 3 | 5.8s | 并行调 2 个 get_weather → 调 calculate → 回复 |
| 什么是 RAG | 1 | 8.1s | 不需要工具，直接回复 |

手写 Agent 能正确完成 3+ 工具的调度，包括并行工具调用。

### 手写 vs LangGraph 架构对比

手写 Agent 做的工作：
1. 手动构造 `messages` 列表（`{"role": "system/user/tool", ...}`）
2. 手动调用 OpenAI API（`client.chat.completions.create`）
3. 手动解析 `tool_calls`
4. 手动执行工具并构造 `tool message`
5. 手动控制循环和退出条件

LangGraph 做的工作：
1. 自动生成 tool schema（从 `@tool` 装饰器）
2. 自动调用 LLM（通过 `ChatModel` 抽象）
3. 自动解析 `tool_calls` 和执行工具
4. 自动构造 `ToolMessage` 对象
5. 通过状态图自动控制循环和退出

### 框架哪些封装是真正有价值的？

**高价值**：
1. **Tool Schema 自动生成** — 从函数签名和 docstring 自动生成 JSON schema，手写需要大量样板代码
2. **状态图可视化** — `get_graph().draw_ascii()` 自动生成执行流程图，调试时非常有用
3. **持久化 Checkpointer** — 一行代码配置，手写需要自己实现序列化/反序列化
4. **多 Agent 编排** — `add_conditional_edges` 一行搞定路由逻辑，手写需要自己写状态机

**可能是过度设计**：
1. **LCEL（`|` 操作符链式调用）** — 可读性下降，debug 时难以定位哪一段出问题
2. **RunnableParallel / RunnableLambda** — 增加理解成本，大多数场景用不到这么复杂的组合
3. **BaseChatModel 继承体系** — 为了统一不同 LLM 引入大量抽象类，直接调 API 更直观

**总结**：框架的核心价值在于**工具管理 + 状态管理 + 多 Agent 编排**；过度设计主要在于链式语法和过度抽象的 Runnable 体系。

---

## Step 6: Agent 的错误处理与健壮性

### 5 种异常场景 + 处理策略

| 异常场景 | 处理策略 |
|---------|---------|
| 工具调用失败（网络超时/内部错误） | 指数退避重试（2次）→ 返回错误信息 |
| LLM 返回不存在的工具名 | 捕获 ValueError，返回可用工具列表给 LLM |
| LLM 返回非法 JSON 参数 | 捕获 JSONDecodeError，把错误信息返回给 LLM |
| LLM 缺少必要参数 | 工具内部校验参数，返回错误提示 |
| Agent 陷入循环（反复调用同一工具） | LoopDetector 检测重复调用，达到阈值后强制退出 |
| LLM API 调用失败 | 捕获异常，重试 N 次后返回降级回复 |

### LoopDetector 循环检测

```python
class LoopDetector:
    def __init__(self, max_same_action=3):
        self.action_history = []  # (tool_name, args_json)

    def record(self, tool_name, arguments):
        key = (tool_name, json.dumps(arguments, sort_keys=True))
        self.action_history.append(key)

    def is_looping(self):
        recent = self.action_history[-self.max_same_action:]
        return all(a == recent[-1] for a in recent)
```

记录每次工具调用，当最近 N 次完全相同时判定为循环，强制退出。

### 核心原则

**Agent 永远不会 crash**，最多返回一条有用的错误信息给用户。所有异常都有 try/except 包裹，不会让 Python 的 traceback 暴露给终端用户。

---

## 阶段验收：多工具多步骤代码助手 Agent

### 验收项目

实现一个"审查 → 建议 → 修改"完整流程的代码助手 Agent。

**测试代码**：
```python
def calculate(expression):
    result = eval(expression)  # eval 安全风险
    return result

def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except:  # 裸 except
        return None
```

### 验收结果

**手写版**（`phase4_acceptance_handwritten.py`）和**框架版**（`phase4_acceptance_framework.py`）均通过。

| 验收标准 | 结果 |
|---------|------|
| 完成"审查 → 建议 → 修改"完整流程 | ✅ code_review → suggest_fix → rewrite_code，4 步完成 |
| 工具选择正确率 >= 80% | ✅ 100%，所有场景都正确选择了 3 个工具 |
| 有异常处理和日志 | ✅ 空代码校验、JSON 解析失败、API 调用失败、循环检测 |
| 能解释 Agent 的每一步决策 | ✅ 有执行轨迹 trace 打印 |
| 代码结构清晰，手写版和框架版都有 | ✅ 两个文件 |

**实际执行轨迹**：
```
用户: "请修改这段代码，修复所有问题"
  Step 1: [code_review] → eval() 安全风险, except 裸捕获
  Step 2: [suggest_fix] → 替换为 ast.literal_eval, except Exception
  Step 3: [rewrite_code] → 修改后代码返回
  Step 4: [Final Answer] → 整理报告回复用户
```

**框架版执行图**（`get_graph().draw_ascii()`）：
```
       __start__
          │
       ┌──────┐
       │ llm  │
       └──────┘
       ╱      ╲
  __end__    tools
              │
              └──→ 回到 llm
```
