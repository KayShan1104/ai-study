"""
Phase 4 阶段验收：多工具多步骤代码助手 Agent（框架版 - LangGraph）
功能：
1. 用户给代码 → Agent 自动审查（bug、性能、安全）
2. 用户问"怎么修" → Agent 给出修改建议
3. 用户说"帮我改" → Agent 输出修改后的代码
4. 至少 3 个工具：code_review、suggest_fix、rewrite_code
5. 支持多轮对话
6. 有工具调用失败时的降级处理
"""

import io
import os
import sys
import json
import logging
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("agent")


# ── 工具定义（@tool 装饰器，自动生成 schema） ──

@tool
def code_review(code: str) -> str:
    """审查代码，检查 bug、性能问题和安全隐患。"""
    if not code.strip():
        return "[错误] 代码为空"

    issues = []
    if "eval(" in code:
        issues.append("⚠️ 安全风险: 使用了 eval()，可能导致代码注入")
    if "except:" in code and "except Exception" not in code:
        issues.append("⚠️ 代码规范: 使用了裸 except，应改为 except Exception")
    if "import *" in code:
        issues.append("⚠️ 代码规范: 使用了 import *，应明确导入")
    if "while True" in code and "break" not in code:
        issues.append("⚠️ 潜在问题: while True 可能导致死循环，需确保有 break")
    if "global " in code:
        issues.append("⚠️ 设计问题: 使用了 global 变量，建议改用类或参数传递")
    if not issues:
        issues.append("✅ 代码整体质量良好，未发现明显 bug、性能或安全问题")
    return "\n".join(issues)


@tool
def suggest_fix(code: str, review_result: str) -> str:
    """根据代码审查结果给出修改建议。"""
    if not code.strip():
        return "[错误] 代码为空"

    suggestions = []
    if "eval()" in review_result:
        suggestions.append("建议: 将 eval() 替换为 ast.literal_eval() 或 json.loads()，避免代码注入风险")
    if "except" in review_result and "裸" in review_result:
        suggestions.append("建议: 将 except: 改为 except Exception as e:，避免捕获 KeyboardInterrupt 等异常")
    if "import *" in review_result:
        suggestions.append("建议: 将 import * 改为明确的 import xxx, yyy")
    if "while True" in review_result:
        suggestions.append("建议: 为 while True 添加 break 条件或最大迭代次数限制")
    if "global" in review_result:
        suggestions.append("建议: 将 global 变量改为函数参数或类属性")
    if not suggestions:
        suggestions.append("✅ 代码无需修改，质量良好")
    return "\n".join(suggestions)


@tool
def rewrite_code(code: str, review_result: str, fix_suggestion: str) -> str:
    """根据审查结果和建议，重写/修改代码。"""
    if not code.strip():
        return "[错误] 代码为空"

    rewritten = code
    if "eval(" in code:
        rewritten = rewritten.replace("eval(", "ast.literal_eval(")
        if "import ast" not in rewritten:
            rewritten = "import ast\n" + rewritten
    if "except:" in rewritten:
        rewritten = rewritten.replace("except:", "except Exception as e:")
    if "while True" in rewritten and "break" not in rewritten:
        rewritten = rewritten.replace("while True:", "while True:  # TODO: 添加 break 条件")
    return f"【修改后代码】\n```\n{rewritten}\n```"


TOOLS = [code_review, suggest_fix, rewrite_code]


# ── 状态定义 ──

class AgentState(dict):
    messages: Annotated[list, add_messages]
    code_context: str


# ── 节点和路由 ──

def call_llm(state: AgentState) -> dict:
    """LLM 节点"""
    messages = state["messages"]
    from langchain_core.messages import AIMessage
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return {"messages": []}

    code_ctx = state.get("code_context", "")
    # 插入系统提示
    system_prompt = f"""你是代码助手，有三个工具:
- code_review(code): 审查代码
- suggest_fix(code, review_result): 根据审查结果给修改建议
- rewrite_code(code, review_result, fix_suggestion): 重写代码

代码上下文:
{code_ctx}
"""
    full_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.bind_tools(TOOLS).invoke(full_messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """条件边: 判断是否继续调工具"""
    messages = state["messages"]
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """构建 LangGraph 图"""
    builder = StateGraph(AgentState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "llm")
    return builder.compile()


# ── 演示 ──

def extract_ai_reply(result: dict) -> str:
    from langchain_core.messages import AIMessage
    msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)]
    return msgs[-1].content if msgs else ""


TEST_CODE = """
def calculate(expression):
    result = eval(expression)
    return result

def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except:
        return None
"""


def demo_framework_review():
    """测试 1: 代码审查"""
    print("=" * 60)
    print("框架版 测试 1: 代码审查")
    print("=" * 60)
    print(f"代码:\n{TEST_CODE}")

    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="请审查以下代码:\n" + TEST_CODE)],
        "code_context": TEST_CODE,
    })

    print(f"\n最终回复:\n{extract_ai_reply(result)}\n")


def demo_framework_fix():
    """测试 2: 修改建议 + 重写"""
    print("=" * 60)
    print("框架版 测试 2: 修改建议 + 重写代码")
    print("=" * 60)

    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="请帮我修改这段代码，修复所有问题:\n" + TEST_CODE)],
        "code_context": TEST_CODE,
    })

    # 打印工具调用轨迹
    print("执行轨迹:")
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  → {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:80]})")
        elif hasattr(msg, "name") and msg.name:
            content_preview = msg.content[:100].replace("\n", " ")
            print(f"    结果: {content_preview}...")

    print(f"\n最终回复:\n{extract_ai_reply(result)}\n")


def demo_framework_flow_chart():
    """打印执行流程图"""
    print("=" * 60)
    print("框架版 Agent 执行流程图")
    print("=" * 60)

    graph = build_graph()
    print(graph.get_graph().draw_ascii())


if __name__ == "__main__":
    demo_framework_review()
    demo_framework_fix()
    demo_framework_flow_chart()
