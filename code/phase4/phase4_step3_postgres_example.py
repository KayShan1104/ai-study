"""
Phase 4 Step 3 补充: 持久化 Checkpointer 示例
演示如何用 PostgreSQL / SQLite 持久化 LangGraph Agent 的对话状态，实现断电重启后恢复。

前置依赖:
  - PostgreSQL: pip install langgraph-checkpoint-postgres psycopg[binary]
  - SQLite:     pip install langgraph-checkpoint-sqlite (langgraph 自带)

使用 Docker 快速启动本地 Postgres:
  docker run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
"""

import io
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated


# ── 工具定义 ──

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
    }
    return weather_data.get(city, f"未收录 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。"""
    try:
        cleaned = expression.replace(" ", "")
        if not all(c in "0123456789+-*/.()" for c in cleaned):
            return f"错误: 包含非法字符"
        result = eval(cleaned)  # noqa: S307 — 学习用途
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ── LLM 初始化 ──

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ── 状态定义 ──

class AgentState(dict):
    messages: Annotated[list, add_messages]


# ── 节点和路由 ──

def call_llm(state: AgentState) -> dict:
    """Node: LLM 调用节点"""
    messages = state["messages"]
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return {"messages": []}
    response = llm.bind_tools([get_weather, calculate]).invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """条件边: 判断是否需要调工具"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── 构建图 ──

builder = StateGraph(AgentState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode([get_weather, calculate]))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")


# ── 辅助函数 ──

def extract_ai_reply(result: dict) -> str:
    """从结果中提取最后一条 AI 回复"""
    msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)]
    return msgs[-1].content if msgs else ""


def build_graph(checkpointer):
    """构建带持久化的图"""
    return builder.compile(checkpointer=checkpointer)


# ── 演示 ──

DSN = os.getenv(
    "LANGGRAPH_PG_DSN",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)


def demo_with_postgres():
    """用 PostgresSaver 持久化（需要外部 PostgreSQL 服务）"""
    print("=" * 60)
    print("Demo 1: PostgresSaver 持久化")
    print("=" * 60)

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        print("[跳过] 未安装 langgraph-checkpoint-postgres")
        print("  安装: pip install langgraph-checkpoint-postgres psycopg[binary]")
        return

    import psycopg
    try:
        with PostgresSaver.from_conn_string(DSN) as checkpointer:
            checkpointer.setup()
            graph = build_graph(checkpointer)

            config_a = {"configurable": {"thread_id": "session_A"}}

            print("\n--- 线程 A - 第1轮 ---")
            result = graph.invoke({"messages": [HumanMessage(content="我叫小明，我是一名程序员。")]}, config_a)
            print(f"用户: 我叫小明，我是一名程序员。")
            print(f"AI: {extract_ai_reply(result)}")

            print("\n--- 线程 A - 第2轮（检查记忆）---")
            result = graph.invoke({"messages": [HumanMessage(content="我叫什么名字？")]}, config_a)
            print(f"用户: 我叫什么名字？")
            print(f"AI: {extract_ai_reply(result)}")

            print("\n[1] 数据已写入 PostgreSQL，关机重启后仍可恢复。\n")

    except psycopg.OperationalError:
        print(f"\n连接失败（Postgres 未启动）: {DSN}")
        print("启动 Docker 容器: docker run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16")


def demo_with_sqlite():
    """用 SqliteSaver 持久化（不需要外部数据库，开箱即用）"""
    print("=" * 60)
    print("Demo 2: SqliteSaver 持久化（本地文件，无需数据库）")
    print("=" * 60)

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = os.path.join(os.path.dirname(__file__), "checkpoints_demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        config_a = {"configurable": {"thread_id": "session_A"}}

        print("\n--- 线程 A - 第1轮 ---")
        result = graph.invoke({"messages": [HumanMessage(content="我叫小明，我喜欢用 Markdown。")]}, config_a)
        print(f"用户: 我叫小明，我喜欢用 Markdown。")
        print(f"AI: {extract_ai_reply(result)}")

        print("\n--- 线程 A - 第2轮（检查记忆）---")
        result = graph.invoke({"messages": [HumanMessage(content="我刚才说了我喜欢什么？")]}, config_a)
        print(f"用户: 我刚才说了我喜欢什么？")
        print(f"AI: {extract_ai_reply(result)}")

        print(f"\n[2] 数据已保存到 {db_path}")
        print(f"    文件大小: {os.path.getsize(db_path) / 1024:.1f} KB\n")

    # 模拟"重启"：重新打开同一个数据库文件
    print("--- 模拟重启后恢复 ---")
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        result = graph.invoke({"messages": [HumanMessage(content="你还记得我是谁吗？")]}, config_a)
        print(f"用户: 你还记得我是谁吗？")
        print(f"AI: {extract_ai_reply(result)}")
        print("\n[3] 重启后成功恢复对话，数据来自本地 SQLite 文件。\n")

    # 列出所有存档线程
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        checkpointer.setup()
        # list() 需要配置，这里用通配符查找所有
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"[4] 数据库中共有 {len(thread_ids)} 个存档线程: {sorted(thread_ids)}")
        print(f"    (删除 {db_path} 可清理所有存档)")


if __name__ == "__main__":
    # Postgres 演示（需要外部服务）
    demo_with_postgres()
    time.sleep(0.5)
    # SQLite 演示（开箱即用）
    demo_with_sqlite()
