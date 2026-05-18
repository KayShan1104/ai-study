"""
Phase 4 Step 2: LangChain 的 Tool 与 Agent
学习目标:
1. 理解 LangChain 的 Tool 定义方式
2. 用 LangGraph 的 create_react_agent 创建 Agent
3. 对比手写 Agent Loop 与 LangGraph Agent 的差异
"""

import io
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# --- LLM 初始化 ---
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# --- 1. 定义工具 ---
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。参数 city 为城市名称，如 "北京"、"上海"。"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%，东南风 2 级",
        "上海": "多云，28°C，湿度 65%，南风 3 级",
        "深圳": "小雨，26°C，湿度 80%，无风",
        "杭州": "晴天，30°C，湿度 55%，西南风 1 级",
    }
    return weather_data.get(city, f"未收录 {city} 的天气数据（模拟数据）")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。参数 expression 为合法的数学表达式，如 "2 + 3 * 4"。"""
    try:
        # 只允许数字和运算符，防止代码注入
        cleaned = expression.replace(" ", "")
        if not all(c in "0123456789+-*/.()" for c in cleaned):
            return f"错误: 表达式 '{expression}' 包含非法字符"
        result = eval(cleaned)  # noqa: S307 — 已做白名单过滤，仅学习用途
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_knowledge(query: str) -> str:
    """在知识库中搜索相关信息。参数 query 为搜索关键词或问题。"""
    knowledge_base = {
        "Python": "Python 是一种广泛使用的高级编程语言，以代码可读性著称。",
        "API": "API（应用程序编程接口）是不同软件系统之间交互的契约。",
        "RAG": "RAG（检索增强生成）结合向量检索和 LLM 生成，减少幻觉。",
        "Agent": "Agent 是能自主使用工具完成多步任务的 AI 系统。",
        "LangChain": "LangChain 是构建 LLM 应用的 Python 框架，提供链式调用和 Agent 能力。",
    }
    for key, value in knowledge_base.items():
        if key.lower() in query.lower() or query.lower() in key.lower():
            return f"[知识库] {key}: {value}"
    return f"[知识库] 未找到与 '{query}' 直接匹配的内容（模拟知识库）"


# --- 2. LangGraph ReAct Agent ---
from langchain.agents import create_agent


def react_agent_demo():
    """演示用 LangGraph 创建 ReAct Agent"""
    print("=" * 60)
    print("Demo 1: LangGraph ReAct Agent")
    print("=" * 60)

    tools = [get_weather, calculate, search_knowledge]

    # create_agent 内部构建 ReAct loop:
    # Thought → Action → Observation → Thought → ... → Final Answer
    agent = create_agent(llm, tools)

    # 测试场景: 需要 Agent 选择并组合多个工具
    test_cases = [
        "北京天气怎么样？",
        "计算一下 123 * 456 + 789 等于多少？",
        "什么是 RAG？",
        "上海和深圳的天气哪个更好？我想算一下 (25 + 26) / 2 的平均温度",
    ]

    for query in test_cases:
        print(f"\n用户: {query}")
        print("-" * 40)
        result = agent.invoke({"messages": [("user", query)]})

        # 提取最终回复
        last_message = result["messages"][-1]
        print(f"AI: {last_message.content}")
        print()


def execution_trace_demo():
    """演示如何查看 Agent 的执行过程（调试用）"""
    print("=" * 60)
    print("Demo 2: 查看 Agent 执行过程")
    print("=" * 60)

    tools = [get_weather, calculate, search_knowledge]
    agent = create_agent(llm, tools)

    query = "杭州天气如何？然后算一下 30 + 25 是多少"
    print(f"\n用户: {query}")
    print("-" * 40)

    result = agent.invoke({"messages": [("user", query)]})

    # 打印完整的消息流
    for msg in result["messages"]:
        role = type(msg).__name__
        content_preview = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
        if role == "HumanMessage":
            print(f"  [User] {content_preview}")
        elif role == "AIMessage":
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            if has_tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                print(f"  [AI → Tool] 调用: {tool_names}")
            else:
                print(f"  [AI] {content_preview}")
        elif role == "ToolMessage":
            print(f"  [Tool Result] {content_preview}")
        print()


def compare_handwritten_vs_framework():
    """对比手写 Agent Loop 与 LangGraph Agent"""
    print("=" * 60)
    print("Demo 3: 手写 vs 框架 对比")
    print("=" * 60)

    print("""
手写 Agent Loop (Phase 1 风格):
  - 需要手动构造 system prompt 描述工具
  - 需要自己解析 LLM 返回的 tool_calls
  - 需要自己执行工具、拼接结果、循环调用
  - 需要自己处理异常和最大迭代次数
  - 代码量: ~100-150 行

LangGraph Agent:
  - 用 @tool 装饰器定义工具，描述自动生成
  - create_react_agent 内部处理 ReAct loop
  - 自动解析 tool_calls、执行工具、回传结果
  - 内置错误处理和迭代限制
  - 代码量: ~10 行（不含工具定义）

LangGraph 更方便的 3 个点:
  1. Tool 描述的 schema 自动生成（从函数签名和 docstring）
  2. ReAct loop 的"调用-执行-回传"循环内置，不需要手写
  3. 消息格式统一，不需要自己维护 "role": "tool" 等字段
""")


if __name__ == "__main__":
    react_agent_demo()
    execution_trace_demo()
    compare_handwritten_vs_framework()
