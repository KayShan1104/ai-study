"""
Phase 4 Step 4: 多 Agent 协作
学习目标:
1. 用 LangGraph 实现 Supervisor 模式（主管分配任务给子 Agent）
2. 用 LangGraph 实现 Sequential 链式模式（A 的输出作为 B 的输入）
3. 对比单 Agent 和多 Agent 在同一个任务上的表现
4. 理解多 Agent 的真正优势和缺点
"""

import io
import os
import sys
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# ── 子 Agent 定义 ──

def writer_agent(state: dict) -> dict:
    """Writer: 负责写代码"""
    task = state.get("current_task", "")
    messages = [
        SystemMessage(content="你是一个资深的 Python 程序员，负责编写高质量代码。"),
        HumanMessage(content=f"请完成以下任务: {task}"),
    ]
    response = llm.invoke(messages)
    return {"writer_output": response.content}


def reviewer_agent(state: dict) -> dict:
    """Reviewer: 负责审查代码"""
    code = state.get("writer_output", "")
    messages = [
        SystemMessage(content="你是一个严格的代码审查员。从正确性、性能、可读性、安全性四个维度审查代码。"),
        HumanMessage(content=f"请审查以下代码:\n\n{code}"),
    ]
    response = llm.invoke(messages)
    return {"review_output": response.content}


def summarizer_agent(state: dict) -> dict:
    """Summarizer: 汇总所有结果"""
    writer = state.get("writer_output", "")
    review = state.get("review_output", "")
    messages = [
        SystemMessage(content="你是一个技术写作助手，负责将代码和审查结果整理成用户友好的格式。"),
        HumanMessage(content=f"【代码】\n{writer}\n\n【审查意见】\n{review}\n\n请整合为一份完整报告。"),
    ]
    response = llm.invoke(messages)
    return {"final_report": response.content}


# ── 状态定义 ──

class MultiAgentState(dict):
    messages: Annotated[list, add_messages]
    user_input: str
    current_task: str
    writer_output: str
    review_output: str
    final_report: str
    step_count: int


# ── 模式 1: Sequential 链式模式 ──

def build_sequential_graph():
    """Sequential: Writer → Reviewer → Summarizer，线性执行"""
    graph = StateGraph(MultiAgentState)
    graph.add_node("Writer", writer_agent)
    graph.add_node("Reviewer", reviewer_agent)
    graph.add_node("Summarizer", summarizer_agent)
    graph.add_edge(START, "Writer")
    graph.add_edge("Writer", "Reviewer")
    graph.add_edge("Reviewer", "Summarizer")
    graph.add_edge("Summarizer", END)
    return graph.compile()


def sequential_demo():
    """Demo 1: Sequential 链式模式"""
    print("=" * 60)
    print("Demo 1: Sequential 链式模式 (Writer → Reviewer → Summarizer)")
    print("=" * 60)

    graph = build_sequential_graph()

    init_state = {
        "messages": [],
        "user_input": "写一个 Python 函数，实现二分查找算法",
        "current_task": "实现二分查找算法",
        "writer_output": "",
        "review_output": "",
        "final_report": "",
        "step_count": 0,
    }

    print(f"\n用户: {init_state['user_input']}\n")
    result = graph.invoke(init_state)

    print(f"【Writer 输出】\n{result['writer_output'][:200]}...\n")
    print(f"【Reviewer 输出】\n{result['review_output'][:200]}...\n")
    print(f"【Summarizer 报告】\n{result['final_report'][:200]}...\n")
    print("[完成] Sequential 模式执行完毕\n")


# ── 模式 2: Supervisor 主管模式 ──

def supervisor_node(state: dict) -> dict:
    """Supervisor: 根据当前进度决定下一步"""
    has_writer = bool(state.get("writer_output", "").strip())
    has_review = bool(state.get("review_output", "").strip())
    has_report = bool(state.get("final_report", "").strip())
    step_count = state.get("step_count", 0)

    # 安全限制：最多 5 步
    if step_count >= 5 or has_report:
        return {"next_action": "FINISH", "step_count": step_count + 1}

    if not has_writer:
        next_action = "Writer"
    elif not has_review:
        next_action = "Reviewer"
    elif not has_report:
        next_action = "Summarizer"
    else:
        next_action = "FINISH"

    return {"next_action": next_action, "step_count": step_count + 1}


def route_supervisor(state: dict) -> str:
    """路由: 根据 next_action 跳转到对应节点"""
    action = state.get("next_action", "FINISH")
    mapping = {"Writer": "Writer", "Reviewer": "Reviewer", "Summarizer": "Summarizer", "FINISH": END}
    return mapping.get(action, END)


def build_supervisor_graph():
    """Supervisor 模式: 主管动态分配任务"""
    graph = StateGraph(MultiAgentState)
    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("Writer", writer_agent)
    graph.add_node("Reviewer", reviewer_agent)
    graph.add_node("Summarizer", summarizer_agent)

    graph.add_edge(START, "Supervisor")
    graph.add_conditional_edges("Supervisor", route_supervisor, {
        "Writer": "Writer", "Reviewer": "Reviewer", "Summarizer": "Summarizer", END: END,
    })
    graph.add_edge("Writer", "Supervisor")
    graph.add_edge("Reviewer", "Supervisor")
    graph.add_edge("Summarizer", "Supervisor")

    return graph.compile()


def supervisor_demo():
    """Demo 2: Supervisor 主管模式"""
    print("=" * 60)
    print("Demo 2: Supervisor 主管模式 (Supervisor 动态分配)")
    print("=" * 60)

    graph = build_supervisor_graph()

    init_state = {
        "messages": [],
        "user_input": "写一个冒泡排序函数，审查它的性能，然后整理成报告",
        "current_task": "实现冒泡排序并审查性能",
        "writer_output": "",
        "review_output": "",
        "final_report": "",
        "step_count": 0,
        "next_action": "",
    }

    print(f"\n用户: {init_state['user_input']}\n")
    result = graph.invoke(init_state)

    steps = result["step_count"]
    print(f"Supervisor 执行了 {steps} 步决策")
    print(f"Writer 输出: {'有' if result.get('writer_output') else '无'}")
    print(f"Reviewer 输出: {'有' if result.get('review_output') else '无'}")
    print(f"最终报告: {'有' if result.get('final_report') else '无'}")
    print(f"\n【最终报告】\n{result.get('final_report', '')[:300]}...\n")
    print("[完成] Supervisor 模式执行完毕\n")


# ── 模式 3: 单 Agent 对比 ──

def single_agent_demo():
    """Demo 3: 单 Agent 做同样的任务"""
    print("=" * 60)
    print("Demo 3: 单 Agent 对比（完成同样的任务）")
    print("=" * 60)

    task = "请完成以下任务: 1) 写一个 Python 冒泡排序函数 2) 审查它的性能 3) 整理成报告"

    messages = [
        SystemMessage(content="你是一个全能的 AI 助手，负责写代码、审查代码和整理报告。"),
        HumanMessage(content=task),
    ]

    print(f"\n用户: {task}\n")
    response = llm.invoke(messages)
    print(f"单 Agent 回复:\n{response.content[:300]}...\n")

    print("对比分析:")
    print("""
+------------------+----------------------+----------------------+
| 维度             | 单 Agent             | 多 Agent (Supervisor) |
+------------------+----------------------+----------------------+
| LLM 调用次数     | 1 次                 | 3-4 次               |
| 响应延迟         | 低（~5s）            | 高（~15-20s）        |
| Token 成本       | 低                   | 高（3-4 倍）         |
| 专业化程度       | 通用，可能都不精     | 每个 Agent 专精一项  |
| 审查独立性       | 自己写自己审，可能   | 独立 Agent 审查，     |
|                  | 忽略自己的 bug       | 更客观               |
| 可维护性         | 改 prompt 影响全局   | 每个 Agent 独立修改  |
+------------------+----------------------+----------------------+
""")


# ── 对比总结 ──

def multi_agent_pros_cons():
    """Demo 4: 多 Agent 的真正优势和缺点"""
    print("=" * 60)
    print("Demo 4: 多 Agent 的真正优势和缺点")
    print("=" * 60)

    print("""
多 Agent 的真正优势（不仅仅是"先后调用"）:

1. **关注点分离（Separation of Concerns）**
   - 每个 Agent 有独立的 system prompt，专注一个领域
   - 修改 Writer 的 prompt 不会影响 Reviewer 的行为

2. **独立审查（Independent Review）**
   - 单 Agent 自己写自己审，容易"自我偏见"
   - 独立 Reviewer Agent 有不同的 system prompt，更容易发现问题

3. **可组合性（Composability）**
   - 可以把 Writer 同时给多个 Reviewer（一个查安全、一个查性能）
   - 可以动态插入新 Agent（比如加一个 TestAgent）

4. **失败隔离（Failure Isolation）**
   - Writer 失败了不影响 Reviewer 的执行
   - 可以针对单个 Agent 做重试、降级

多 Agent 的缺点:

1. **成本高**: 3 个 Agent = 3-4 次 LLM 调用，Token 消耗是单 Agent 的 3-4 倍
2. **延迟大**: 串行执行 3-4 个 Agent，用户等的时间更长
3. **调试难**: Agent 之间交互可能导致意外行为（互相误导、循环）
4. **过度设计风险**: 简单任务用单 Agent 就够了
5. **Agent 之间可能互相误导**: Writer 输出错误代码，Reviewer 没看出来
""")


if __name__ == "__main__":
    sequential_demo()
    supervisor_demo()
    single_agent_demo()
    multi_agent_pros_cons()
