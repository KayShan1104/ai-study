"""
Phase 4 Step 3: Agent 的记忆与状态管理
学习目标:
1. 理解 LangGraph 的 State、Node、Edge 三个概念
2. 实现有状态的 Agent，能在多轮对话中记住用户偏好
3. 对比三种记忆方式：完整历史、摘要记忆、向量检索记忆
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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# --- 1. LangGraph 状态机模型：State、Node、Edge ---
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import operator


class AgentState(dict):
    """State: Agent 的当前状态，包含消息历史和其他变量"""
    messages: Annotated[list, add_messages]
    # 自定义状态字段
    user_preferences: dict


def call_llm(state: AgentState) -> dict:
    """Node: LLM 调用节点，接收状态、调用模型、返回新状态"""
    messages = state["messages"]

    # 如果最后一条已经是 AI 回复（且无 tool_calls），说明已完成，直接返回
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return {"messages": []}

    # 注入用户偏好到 system message
    prefs = state.get("user_preferences", {})
    if prefs:
        pref_text = ", ".join(f"{k}: {v}" for k, v in prefs.items())
        system_msg = SystemMessage(content=f"用户偏好: {pref_text}。请在回复中考虑这些偏好。")
        messages = [system_msg] + list(messages)

    response = llm.bind_tools([get_weather, calculate]).invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Edge 路由条件: 决定是调工具还是直接回复"""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def memory_agent_demo():
    """Demo 1: 用 LangGraph 状态机实现有记忆的 Agent"""
    print("=" * 60)
    print("Demo 1: 有状态的 Agent（记住用户偏好）")
    print("=" * 60)

    # 构建图
    graph_builder = StateGraph(AgentState)

    # 添加节点
    graph_builder.add_node("llm", call_llm)
    graph_builder.add_node("tools", ToolNode([get_weather, calculate]))

    # 添加边
    graph_builder.add_edge(START, "llm")
    graph_builder.add_conditional_edges(
        "llm",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "llm")

    graph = graph_builder.compile()

    # 多轮对话测试
    conversation = [
        ("我叫小明，我喜欢用 Markdown 格式，我是一名程序员。", "存储偏好"),
        ("我叫什么名字？", "检查记忆"),
        ("我喜欢什么格式？", "检查偏好"),
        ("北京天气怎么样？", "工具调用 + 偏好"),
    ]

    # 初始状态
    state = {"messages": [], "user_preferences": {}}

    for user_msg, label in conversation:
        print(f"\n[用户] {user_msg}")
        print(f"[标签] {label}")
        print("-" * 40)

        # 从消息中提取偏好（模拟：前几条消息设置偏好）
        if "小明" in user_msg:
            state["user_preferences"] = {
                "名字": "小明",
                "格式偏好": "Markdown",
                "职业": "程序员",
            }

        # 传入完整状态，graph 内部用 add_messages 追加新消息
        state = graph.invoke(state)
        # 追加用户的真实输入到消息历史
        state["messages"].append(HumanMessage(content=user_msg))

        # 提取最终回复（最后一个非 tool 的 AI 消息）
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"[AI] {msg.content[:200]}")
                break


def summarize_messages(messages: list, max_tokens: int = 500) -> list:
    """用 LLM 对历史消息做摘要（压缩上下文）"""
    if len(messages) <= 2:
        return messages

    # 保留最近 2 条消息，其余摘要
    recent = messages[-2:]
    history_text = "\n".join(
        f"{msg.__class__.__name__}: {msg.content}" for msg in messages[:-2]
    )

    summary_prompt = f"请用 3-5 句话总结以下对话:\n{history_text}"
    result = llm.invoke([HumanMessage(content=summary_prompt)])

    summary_msg = SystemMessage(content=f"[对话摘要] {result.content}")
    return [summary_msg] + recent


def summary_memory_demo():
    """Demo 2: 摘要记忆 — 压缩长对话历史"""
    print("\n" + "=" * 60)
    print("Demo 2: 摘要记忆（压缩历史）")
    print("=" * 60)

    # 模拟长对话
    messages = [
        HumanMessage(content="你好，我喜欢 Python 编程。"),
        AIMessage(content="你好！Python 是一门非常优秀的语言。"),
        HumanMessage(content="我平时用 Django 做 web 开发。"),
        AIMessage(content="Django 是 Python 生态中最流行的 web 框架之一。"),
        HumanMessage(content="我还喜欢用 Markdown 写文档，用 Git 管理代码。"),
        AIMessage(content="Markdown 和 Git 是现代开发者的标配工具。"),
        HumanMessage(content="我最近在学习 LangChain 和 RAG 技术。"),
        AIMessage(content="LangChain 和 RAG 是 LLM 应用的热门方向。"),
    ]

    print(f"\n原始消息数: {len(messages)}")
    for msg in messages:
        print(f"  {msg.__class__.__name__}: {msg.content[:40]}...")

    compressed = summarize_messages(messages)
    print(f"\n压缩后消息数: {len(compressed)}")
    for msg in compressed:
        print(f"  {msg.__class__.__name__}: {msg.content[:80]}...")


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。参数 city 为城市名称。"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%，东南风 2 级",
        "上海": "多云，28°C，湿度 65%，南风 3 级",
        "深圳": "小雨，26°C，湿度 80%，无风",
        "杭州": "晴天，30°C，湿度 55%，西南风 1 级",
    }
    return weather_data.get(city, f"未收录 {city} 的天气数据（模拟数据）")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。参数 expression 为合法的数学表达式。"""
    try:
        cleaned = expression.replace(" ", "")
        if not all(c in "0123456789+-*/.()" for c in cleaned):
            return f"错误: 表达式包含非法字符"
        result = eval(cleaned)  # noqa: S307 — 学习用途，已做白名单过滤
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def vector_memory_demo():
    """Demo 3: 向量存储记忆 — 从历史对话中检索相关内容"""
    print("\n" + "=" * 60)
    print("Demo 3: 向量检索记忆（从向量库检索相关历史）")
    print("=" * 60)

    print("""
原理:
  1. 将历史对话分段，存入向量数据库（如 Phase 3 学的 Chroma）
  2. 用户新提问时，在向量库中检索最相关的历史片段
  3. 将检索结果作为 context 注入 prompt，而非传入完整历史

优势:
  - 上下文窗口不随对话增长（固定大小）
  - 能检索到很久以前但相关的内容
  - 适合长对话或知识库场景

劣势:
  - 可能漏掉上下文中的重要信息
  - 检索质量依赖 embedding 质量
  - 需要额外的向量存储和检索开销

适用场景对比:
  完整历史: 短对话 (< 20 轮)，需要完整上下文
  摘要记忆: 中等长度对话，保留整体脉络
  向量检索: 长对话/知识库，只需要相关内容
""")


def compare_three_memory_approaches():
    """Demo 4: 三种记忆方式对比总结"""
    print("=" * 60)
    print("Demo 4: 三种记忆方式对比")
    print("=" * 60)

    print("""
+----------------+------------------+------------------+------------------+
| 维度           | 完整消息历史     | 摘要记忆         | 向量检索记忆     |
+----------------+------------------+------------------+------------------+
| 实现难度       | 最简单           | 中等             | 最复杂           |
| Token 消耗     | 线性增长         | 固定（摘要）     | 固定（top-k）    |
| 上下文完整性   | 100% 保留        | 有信息损失       | 仅保留相关内容   |
| 长期记忆能力   | 差（超限即丢）   | 中（摘要会漂移） | 好（持久化存储） |
| 适用场景       | 短对话/简单任务  | 中等长度对话     | 长对话/知识库    |
| 成本           | 高（长上下文）   | 中（摘要调用LLM）| 中（向量检索）   |
+----------------+------------------+------------------+------------------+

总结：
- 大多数场景：摘要记忆是性价比最高的选择
- 简单场景：完整历史足够
- 复杂场景：向量检索 + 摘要混合使用
""")


if __name__ == "__main__":
    memory_agent_demo()
    summary_memory_demo()
    vector_memory_demo()
    compare_three_memory_approaches()
