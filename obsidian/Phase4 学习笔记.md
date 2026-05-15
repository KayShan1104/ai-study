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
