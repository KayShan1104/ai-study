"""
Phase 4 Step 5: 手写 Agent vs 框架 Agent 对比
学习目标:
1. 不依赖 LangChain/LangGraph，手写一个 ReAct 风格的 Agent
2. 支持多个工具、最大迭代次数限制
3. 和 LangGraph 的实现对比代码量、可读性、可调试性、扩展性
4. 理解框架底层做了什么
"""

import io
import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"


# ── 工具定义（纯 Python，不依赖任何框架） ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '2 + 3 * 4'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中搜索相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或问题"},
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """执行工具调用，纯 Python 路由，不依赖框架"""
    if tool_name == "get_weather":
        weather_data = {
            "北京": "晴天，25°C，湿度 40%，东南风 2 级",
            "上海": "多云，28°C，湿度 65%，南风 3 级",
            "深圳": "小雨，26°C，湿度 80%，无风",
            "杭州": "晴天，30°C，湿度 55%，西南风 1 级",
        }
        return weather_data.get(arguments["city"], f"未收录 {arguments['city']} 的天气数据")

    elif tool_name == "calculate":
        try:
            expr = arguments["expression"].replace(" ", "")
            if not all(c in "0123456789+-*/.()" for c in expr):
                return f"错误: 表达式包含非法字符"
            result = eval(expr)  # noqa: S307 — 学习用途
            return f"{arguments['expression']} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    elif tool_name == "search_knowledge":
        kb = {
            "Python": "Python 是一种广泛使用的高级编程语言，以代码可读性著称。",
            "API": "API（应用程序编程接口）是不同软件系统之间交互的契约。",
            "RAG": "RAG（检索增强生成）结合向量检索和 LLM 生成，减少幻觉。",
            "Agent": "Agent 是能自主使用工具完成多步任务的 AI 系统。",
        }
        for key, value in kb.items():
            if key.lower() in arguments["query"].lower():
                return f"[知识库] {key}: {value}"
        return f"[知识库] 未找到与 '{arguments['query']}' 直接匹配的内容"

    return f"错误: 未知工具 '{tool_name}'"


# ── 手写 ReAct Agent ──

def react_agent(question: str, max_steps: int = 5) -> dict:
    """
    手写 ReAct Agent（Thought → Action → Observation 循环）
    不依赖任何 Agent 框架，纯 API 调用 + 循环控制。

    返回: {"messages": list, "steps": list}
    """
    messages = [
        {"role": "system", "content": "你是一个助手，可以用工具解决问题。每一步先思考，需要时调用工具。"},
        {"role": "user", "content": question},
    ]

    trace = []  # 记录每一步的执行过程

    for step in range(max_steps):
        trace.append({"step": step + 1, "phase": "thought", "tools": []})

        # 1. 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message

        # 2. 判断是否调用了工具
        if message.tool_calls:
            trace[-1]["phase"] = "action"
            tool_calls = message.tool_calls

            # 把 LLM 的回复加入消息历史
            messages.append({
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            })

            # 3. 执行工具
            for tc in tool_calls:
                tool_name = tc.function.name
                arguments = json.loads(tc.function.arguments)
                result = execute_tool(tool_name, arguments)

                trace[-1]["tools"].append({
                    "name": tool_name,
                    "args": arguments,
                    "result": result,
                })

                # 把工具结果加入消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        else:
            # 4. LLM 直接回复，循环结束
            trace[-1]["phase"] = "final_answer"
            trace[-1]["answer"] = message.content
            return {"messages": messages, "steps": trace, "answer": message.content}

    # 达到最大迭代次数
    return {"messages": messages, "steps": trace, "answer": "达到最大迭代次数，无法完成。"}


def print_trace(result: dict):
    """打印 Agent 执行轨迹"""
    print(f"用户问题: {result['steps'][0].get('question', '见上文')}\n")

    for step in result["steps"]:
        step_num = step["step"]
        phase = step["phase"]
        if phase == "action":
            for tool in step.get("tools", []):
                print(f"  Step {step_num}: [Action] 调用 {tool['name']}({json.dumps(tool['args'], ensure_ascii=False)})")
                print(f"           [Result] {tool['result'][:80]}")
        elif phase == "final_answer":
            print(f"  Step {step_num}: [Final Answer] {step.get('answer', '')[:100]}")
    print()

    print(f"最终回复: {result['answer']}\n")


# ── Demo: 手写 Agent ──

def handwritten_demo():
    """Demo 1: 手写 ReAct Agent"""
    print("=" * 60)
    print("Demo 1: 手写 ReAct Agent（无框架依赖）")
    print("=" * 60)

    test_cases = [
        "北京天气怎么样？",
        "计算一下 123 * 456 等于多少？",
        "上海和深圳的天气哪个更好？帮我算一下两地温度的平均值",
        "什么是 RAG？",
    ]

    for q in test_cases:
        print(f"\n{'─' * 50}")
        print(f"用户: {q}")
        print(f"{'─' * 50}")

        start = time.time()
        result = react_agent(q)
        elapsed = time.time() - start

        print_trace(result)
        print(f"耗时: {elapsed:.1f}s, 迭代次数: {len(result['steps'])}")


# ── 对比分析 ──

def comparison_analysis():
    """Demo 2: 手写 vs 框架对比"""
    print("=" * 60)
    print("Demo 2: 手写 Agent vs LangGraph Agent 架构对比")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                    手写 Agent (本文件)                           │
│                                                                 │
│  messages = [system, user]                                      │
│  for step in range(max_steps):                                  │
│      response = client.chat.completions.create(...)             │
│      if response.tool_calls:                                    │
│          for tc in tool_calls:                                  │
│              result = execute_tool(tc)                          │
│              messages.append({"role": "tool", ...})             │
│      else:                                                      │
│          return response.content                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph Agent (Step 2)                        │
│                                                                 │
│  builder = StateGraph(AgentState)                               │
│  builder.add_node("llm", call_llm)                              │
│  builder.add_node("tools", ToolNode(tools))                     │
│  builder.add_edge(START, "llm")                                 │
│  builder.add_conditional_edges("llm", should_continue, ...)     │
│  builder.add_edge("tools", "llm")                               │
│  graph = builder.compile()                                      │
│  result = graph.invoke({"messages": [...]})                     │
└─────────────────────────────────────────────────────────────────┘

手写 Agent 做的工作:
  1. 手动构造 messages 列表
  2. 手动调用 OpenAI API
  3. 手动解析 tool_calls
  4. 手动执行工具并构造 tool message
  5. 手动控制循环和退出条件

LangGraph 做的工作:
  1. 自动生成 tool schema（从 @tool 装饰器）
  2. 自动调用 LLM（通过 ChatModel 抽象）
  3. 自动解析 tool_calls 和执行工具
  4. 自动构造 tool message（ToolMessage 对象）
  5. 通过状态图自动控制循环和退出
""")


def pros_cons_analysis():
    """Demo 3: 框架封装的价值分析"""
    print("=" * 60)
    print("Demo 3: 框架哪些封装是真正有价值的？哪些是过度设计？")
    print("=" * 60)

    print("""
【真正有价值的封装】

1. Tool Schema 自动生成（@tool 装饰器）
   手写: 需要手动构造 JSON schema 描述每个工具
   框架: 从函数签名和 docstring 自动生成
   价值: 高 — 减少重复劳动，且自动保持同步

2. ToolMessage 对象封装
   手写: 需要自己构造 {"role": "tool", "tool_call_id": "...", "content": "..."}
   框架: ToolMessage(content=..., tool_call_id=...)
   价值: 中 — 减少出错概率（拼写、格式）

3. 状态图可视化（get_graph().draw_ascii()）
   手写: 完全没有
   框架: 自动生执行流程图
   价值: 高 — 调试时非常有用

4. 持久化 Checkpointer
   手写: 需要自己实现状态序列化/反序列化
   框架: 一行代码配置 SqliteSaver/PostgresSaver
   价值: 高 — 生产环境必备

5. 多 Agent 编排（StateGraph + 条件边）
   手写: 需要自己写路由逻辑和状态管理
   框架: add_conditional_edges 一行搞定
   价值: 高 — 复杂流程时优势明显

【可能是过度设计的封装】

1. LCEL（| 操作符链式调用）
   手写: prompt → llm → parser 三行代码
   框架: prompt | llm | parser
   问题: 可读性下降（对新手不友好），debug 时难以定位哪一段出问题

2. RunnableParallel / RunnableLambda 等抽象
   问题: 增加了理解成本，但大多数场景用不到这么复杂的组合

3. 各种 BaseChatModel 继承体系
   问题: 为了统一不同 LLM 的接口，引入了大量抽象类，
         实际上直接调 OpenAI API 更直观

【总结】
  框架的核心价值在于: 工具管理 + 状态管理 + 多 Agent 编排
  过度设计主要在于: 链式语法和过度抽象的 Runnable 体系
""")


if __name__ == "__main__":
    handwritten_demo()
    comparison_analysis()
    pros_cons_analysis()
