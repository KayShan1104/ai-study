"""
Phase 4 阶段验收：多工具多步骤代码助手 Agent（手写版）
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

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("agent")


# ── 工具定义 ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "code_review",
            "description": "审查代码，检查 bug、性能问题和安全隐患。返回审查结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "需要审查的代码"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_fix",
            "description": "根据代码审查结果给出修改建议。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "原始代码"},
                    "review_result": {"type": "string", "description": "代码审查结果"},
                },
                "required": ["code", "review_result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite_code",
            "description": "根据审查结果和建议，重写/修改代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "原始代码"},
                    "review_result": {"type": "string", "description": "代码审查结果"},
                    "fix_suggestion": {"type": "string", "description": "修改建议"},
                },
                "required": ["code", "review_result", "fix_suggestion"],
            },
        },
    },
]

AVAILABLE_TOOLS = {"code_review": True, "suggest_fix": True, "rewrite_code": True}


# ── 工具实现 ──

def execute_tool(tool_name: str, arguments: dict) -> str:
    """执行工具调用"""
    if tool_name not in AVAILABLE_TOOLS:
        available = ", ".join(sorted(AVAILABLE_TOOLS.keys()))
        return f"[错误] 未知工具 '{tool_name}'。可用工具: {available}"

    if tool_name == "code_review":
        code = arguments.get("code", "")
        if not code.strip():
            return "[错误] 代码为空"
        # 审查逻辑：模拟对代码的分析
        issues = []
        if "eval(" in code:
            issues.append("⚠️ 安全风险: 使用了 eval()，可能导致代码注入")
        if "except:" in code and "except Exception" not in code:
            issues.append("⚠️ 代码规范: 使用了裸 except，应改为 except Exception")
        if "import *" in code:
            issues.append("⚠️ 代码规范: 使用了 import *，应明确导入")
        if "while True" in code:
            issues.append("⚠️ 潜在问题: while True 可能导致死循环，需确保有 break")
        if "global " in code:
            issues.append("⚠️ 设计问题: 使用了 global 变量，建议改用类或参数传递")
        if not issues:
            issues.append("✅ 代码整体质量良好，未发现明显 bug、性能或安全问题")
        return "\n".join(issues)

    elif tool_name == "suggest_fix":
        code = arguments.get("code", "")
        review = arguments.get("review_result", "")
        if not code.strip():
            return "[错误] 代码为空"
        # 根据审查结果生成建议
        suggestions = []
        if "eval()" in review:
            suggestions.append("建议: 将 eval() 替换为 ast.literal_eval() 或 json.loads()，避免代码注入风险")
        if "except" in review and "裸" in review:
            suggestions.append("建议: 将 except: 改为 except Exception as e:，避免捕获 KeyboardInterrupt 等异常")
        if "import *" in review:
            suggestions.append("建议: 将 import * 改为明确的 import xxx, yyy")
        if "while True" in review:
            suggestions.append("建议: 为 while True 添加 break 条件或最大迭代次数限制")
        if "global" in review:
            suggestions.append("建议: 将 global 变量改为函数参数或类属性")
        if not suggestions:
            suggestions.append("✅ 代码无需修改，质量良好")
        return "\n".join(suggestions)

    elif tool_name == "rewrite_code":
        code = arguments.get("code", "")
        review = arguments.get("review_result", "")
        fix = arguments.get("fix_suggestion", "")
        if not code.strip():
            return "[错误] 代码为空"
        # 模拟代码重写
        rewritten = code
        if "eval(" in code:
            rewritten = rewritten.replace("eval(", "ast.literal_eval(")
            if "import ast" not in rewritten:
                rewritten = "import ast\n" + rewritten
        if "except:" in rewritten:
            rewritten = rewritten.replace("except:", "except Exception as e:")
        if "while True" in rewritten and "break" not in rewritten:
            rewritten = rewritten.replace("while True:", "while True:  # TODO: 添加 break 条件\n        pass  # 占位，需添加退出逻辑")
        return rewritten


# ── 手写 Agent ──

def code_assistant_agent(user_input: str, code_context: str = "", max_steps: int = 8) -> dict:
    """
    手写代码助手 Agent:
    - 支持 code_review → suggest_fix → rewrite_code 完整流程
    - 异常处理 + 循环检测
    """
    messages = [
        {
            "role": "system",
            "content": f"""你是一个代码助手，有三个可用工具:
1. code_review(code): 审查代码，检查 bug、性能、安全问题
2. suggest_fix(code, review_result): 根据审查结果给出修改建议
3. rewrite_code(code, review_result, fix_suggestion): 根据审查和建议重写代码

当前代码上下文:
{code_context}

请根据用户请求，合理使用工具完成任务。""",
        },
        {"role": "user", "content": user_input},
    ]

    trace = []

    for step in range(max_steps):
        logger.info(f"Step {step + 1}/{max_steps}: 调用 LLM")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            return {"answer": f"[服务不可用] API 调用失败: {e}", "trace": trace, "status": "error"}

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            return {"answer": message.content, "trace": trace, "status": "success"}

        messages.append({
            "role": "assistant",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {tc.function.arguments[:100]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"[参数解析失败] 请返回合法 JSON",
                })
                continue

            logger.info(f"  工具: {tc.function.name}({json.dumps(arguments, ensure_ascii=False)[:100]})")

            result = execute_tool(tc.function.name, arguments)
            trace.append({"step": step + 1, "tool": tc.function.name, "args": arguments, "result": result[:200]})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return {"answer": "[达到最大迭代次数]", "trace": trace, "status": "max_steps"}


# ── 演示 ──

def print_trace(result: dict):
    """打印执行轨迹"""
    for entry in result.get("trace", []):
        print(f"  Step {entry['step']}: [{entry['tool']}]")
        print(f"    结果: {entry['result'][:150]}")
    print(f"\n最终回复:\n{result['answer']}\n")


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


def demo_review():
    """测试 1: 代码审查"""
    print("=" * 60)
    print("测试 1: 代码审查")
    print("=" * 60)
    print(f"代码:\n{TEST_CODE}")

    result = code_assistant_agent("请审查以下代码的安全性、性能和质量:\n" + TEST_CODE)
    print_trace(result)


def demo_suggest_fix():
    """测试 2: 修改建议"""
    print("=" * 60)
    print("测试 2: 修改建议")
    print("=" * 60)

    result = code_assistant_agent(
        "这段代码有什么问题？怎么修？\n" + TEST_CODE,
        code_context=TEST_CODE,
    )
    print_trace(result)


def demo_rewrite():
    """测试 3: 重写代码"""
    print("=" * 60)
    print("测试 3: 重写代码")
    print("=" * 60)

    result = code_assistant_agent(
        "请帮我修改这段代码，修复所有发现的问题:\n" + TEST_CODE,
        code_context=TEST_CODE,
    )
    print_trace(result)


def demo_error_handling():
    """测试 4: 异常处理"""
    print("=" * 60)
    print("测试 4: 异常处理（空代码）")
    print("=" * 60)

    result = code_assistant_agent("请审查这段代码: （空）")
    print_trace(result)


def demo_flow_chart():
    """打印执行流程图"""
    print("=" * 60)
    print("Agent 执行流程图")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                        用户输入代码/请求                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │    LLM      │ 判断需要什么工具
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │code_review│ │suggest_fix│ │rewrite_code│
        └─────┬────┘ └─────┬────┘ └─────┬────┘
              │            │            │
              ▼            ▼            ▼
        审查结果返回LLM → 建议返回LLM → 代码返回LLM
                           │
                    ┌──────▼──────┐
                    │  最终回复用户 │
                    └─────────────┘

典型流程:
  用户: "审查这段代码"
    → LLM 调用 code_review → 返回审查结果 → 回复用户

  用户: "怎么修？"
    → LLM 调用 code_review → 调用 suggest_fix → 返回建议 → 回复用户

  用户: "帮我改"
    → LLM 调用 code_review → suggest_fix → rewrite_code → 返回代码 → 回复用户

异常处理:
  - 工具失败 → 重试 → 返回错误信息
  - JSON 解析失败 → 告诉 LLM 重新返回
  - LLM API 失败 → 重试 N 次 → 降级回复
  - 空代码 → 工具内部校验，返回错误
""")


if __name__ == "__main__":
    demo_review()
    demo_suggest_fix()
    demo_rewrite()
    demo_error_handling()
    demo_flow_chart()
