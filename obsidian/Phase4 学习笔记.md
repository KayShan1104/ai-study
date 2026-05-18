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
