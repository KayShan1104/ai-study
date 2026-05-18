"""
Phase 4 Step 6: Agent 的错误处理与健壮性
学习目标:
1. 测试 5 种异常场景并实现处理策略
2. Agent 在异常时不会 crash，而是给用户有用的提示
3. 有异常处理日志记录
"""

import io
import os
import sys
import json
import time
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

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")


# ── 工具定义 ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
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
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

AVAILABLE_TOOL_NAMES = {"get_weather", "calculate"}


# ── 异常处理：工具调用 ──

def execute_tool_with_retry(tool_name: str, arguments: dict, max_retries: int = 2) -> str:
    """带重试的工具执行"""
    for attempt in range(max_retries + 1):
        try:
            return execute_tool(tool_name, arguments)
        except Exception as e:
            logger.warning(f"工具 {tool_name} 第 {attempt + 1} 次调用失败: {e}")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))  # 指数退避
            else:
                return f"[工具调用失败] {tool_name} 执行失败（已重试 {max_retries} 次）: {e}"


def execute_tool(tool_name: str, arguments: dict) -> str:
    """执行工具调用（可能抛出异常）"""
    if tool_name not in AVAILABLE_TOOL_NAMES:
        # 异常 1: LLM 返回了不存在的工具名
        available = ", ".join(sorted(AVAILABLE_TOOL_NAMES))
        raise ValueError(f"未知工具 '{tool_name}'。可用工具: {available}")

    if tool_name == "get_weather":
        city = arguments.get("city", "")
        if not city:
            raise ValueError("缺少必要参数: city")
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "深圳": "小雨，26°C",
            "杭州": "晴天，30°C",
        }
        return weather_data.get(city, f"未收录 {city} 的天气数据")

    elif tool_name == "calculate":
        expr = arguments.get("expression", "").replace(" ", "")
        if not all(c in "0123456789+-*/.()" for c in expr):
            return f"错误: 表达式 '{arguments.get('expression', '')}' 包含非法字符"
        result = eval(expr)  # noqa: S307
        return f"{arguments['expression']} = {result}"


# ── 异常处理：循环检测 ──

class LoopDetector:
    """检测 Agent 是否陷入循环（反复调用同一工具同一参数）"""

    def __init__(self, max_same_action: int = 3):
        self.max_same_action = max_same_action
        self.action_history: list[tuple[str, str]] = []  # (tool_name, args_json)

    def record(self, tool_name: str, arguments: dict):
        key = (tool_name, json.dumps(arguments, sort_keys=True))
        self.action_history.append(key)

    def is_looping(self) -> bool:
        if len(self.action_history) < self.max_same_action:
            return False
        last = self.action_history[-1]
        # 检查最近 N 次是否完全相同
        recent = self.action_history[-self.max_same_action:]
        return all(a == last for a in recent)

    def get_hint(self) -> str:
        return (f"检测到循环：你已重复调用 {self.action_history[-1][0]} "
                f"{self.max_same_action} 次。请改变策略或直接回复用户。")


# ── 健壮的 ReAct Agent ──

def robust_react_agent(question: str, max_steps: int = 8) -> dict:
    """
    带错误处理的 ReAct Agent:
    - 工具调用失败重试
    - 未知工具名时告知可用工具列表
    - 循环检测并强制退出
    - 所有异常都有日志记录
    """
    messages = [
        {"role": "system", "content": "你是一个助手，可以用工具解决问题。"},
        {"role": "user", "content": question},
    ]

    loop_detector = LoopDetector(max_same_action=3)
    trace = []
    error_count = 0
    max_errors = 3  # 最多允许 N 次错误

    for step in range(max_steps):
        logger.info(f"Step {step + 1}/{max_steps}: 调用 LLM")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as e:
            # 异常 2: LLM API 调用失败（网络超时等）
            logger.error(f"LLM API 调用失败: {e}")
            error_count += 1
            if error_count >= max_errors:
                return {
                    "answer": f"[服务不可用] LLM API 连续失败 {max_errors} 次，请稍后重试。",
                    "trace": trace,
                    "status": "llm_error",
                }
            messages.append({"role": "system", "content": f"API 调用出错，请重试。错误: {e}"})
            continue

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            # 正常回复，结束
            return {"answer": message.content, "trace": trace, "status": "success"}

        # 有工具调用
        messages.append({
            "role": "assistant",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # 异常 3: LLM 返回了非法的 JSON
                logger.warning(f"工具参数 JSON 解析失败: {tc.function.arguments}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"[参数解析失败] 请返回合法的 JSON 格式。你返回的是: {tc.function.arguments[:100]}",
                })
                continue

            # 循环检测
            loop_detector.record(tool_name, arguments)
            if loop_detector.is_looping():
                logger.warning("检测到 Agent 陷入循环，强制退出")
                hint = loop_detector.get_hint()
                messages.append({"role": "system", "content": hint})
                return {
                    "answer": f"[循环检测] Agent 陷入重复调用 {tool_name}，已强制退出。{hint}",
                    "trace": trace,
                    "status": "loop_detected",
                }

            logger.info(f"  执行工具: {tool_name}({json.dumps(arguments, ensure_ascii=False)})")

            # 带重试执行
            result = execute_tool_with_retry(tool_name, arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return {
        "answer": "[达到最大迭代次数] 无法在限定步数内完成任务。",
        "trace": trace,
        "status": "max_steps_reached",
    }


# ── Demo: 测试各种异常场景 ──

def demo_normal():
    """Demo 1: 正常流程"""
    print("=" * 60)
    print("Demo 1: 正常流程（北京天气 + 计算）")
    print("=" * 60)

    result = robust_react_agent("北京天气怎么样？然后算一下 25 + 26")
    print(f"\n状态: {result['status']}")
    print(f"回复: {result['answer']}\n")


def demo_unknown_tool():
    """Demo 2: LLM 返回未知工具名（通过注入 system prompt 模拟）"""
    print("=" * 60)
    print("Demo 2: 未知工具名处理")
    print("=" * 60)

    # 直接测试 execute_tool 的错误处理
    try:
        result = execute_tool("search_google", {"query": "test"})
        print(f"结果: {result}")
    except ValueError as e:
        print(f"捕获异常: {e}")
        print("✓ Agent 会把这个错误信息返回给 LLM，LLM 就能知道可用工具列表\n")


def demo_missing_param():
    """Demo 3: 缺少必要参数"""
    print("=" * 60)
    print("Demo 3: 缺少必要参数处理")
    print("=" * 60)

    try:
        result = execute_tool("get_weather", {})
        print(f"结果: {result}")
    except ValueError as e:
        print(f"捕获异常: {e}")
        print("✓ 错误信息会返回给 LLM，让其补充参数\n")


def demo_invalid_json():
    """Demo 4: LLM 返回非法 JSON"""
    print("=" * 60)
    print("Demo 4: JSON 解析失败处理")
    print("=" * 60)

    bad_json = '{"city": "北京", "extra": }'
    try:
        arguments = json.loads(bad_json)
    except json.JSONDecodeError as e:
        print(f"捕获异常: {e}")
        print(f"原始内容: {bad_json}")
        print("✓ Agent 会把解析错误告诉 LLM，让其重新返回合法 JSON\n")


def demo_tool_failure_retry():
    """Demo 5: 工具失败重试"""
    print("=" * 60)
    print("Demo 5: 工具失败重试")
    print("=" * 60)

    # 模拟一个会失败的工具
    def failing_tool():
        raise ConnectionError("网络连接超时")

    result = execute_tool_with_retry("failing_tool", {}, max_retries=2)
    print(f"重试后结果: {result}")
    print("✓ 重试 2 次后返回错误信息，Agent 不会 crash\n")


def demo_error_summary():
    """Demo 6: 异常处理策略总结"""
    print("=" * 60)
    print("Demo 6: 5 种异常场景 + 处理策略总结")
    print("=" * 60)

    print("""
┌──────────────────────────────────────────────────────────────────┐
│ 异常场景                  │ 处理策略                             │
├───────────────────────────┼──────────────────────────────────────┤
│ 1. 工具调用失败            │ 指数退避重试 (2次) → 返回错误信息   │
│    (网络超时/内部错误)     │                                      │
├───────────────────────────┼──────────────────────────────────────┤
│ 2. LLM 返回不存在的工具名  │ 捕获 ValueError，返回可用工具列表    │
│                           │ 给 LLM，让其重新选择                 │
├───────────────────────────┼──────────────────────────────────────┤
│ 3. LLM 返回非法 JSON 参数  │ 捕获 JSONDecodeError，把错误信息    │
│                           │ 返回给 LLM 让其重试                  │
├───────────────────────────┼──────────────────────────────────────┤
│ 4. LLM 缺少必要参数        │ 工具内部校验参数，返回错误提示       │
│                           │ 给 LLM 让其补充                      │
├───────────────────────────┼──────────────────────────────────────┤
│ 5. Agent 陷入循环          │ LoopDetector 检测重复调用，达到     │
│    (反复调用同一工具)      │ 阈值后强制退出，返回有用提示         │
├───────────────────────────┼──────────────────────────────────────┤
│ 6. LLM API 调用失败        │ 捕获 OpenAI API 异常，重试 N 次后   │
│    (网络超时/服务不可用)   │ 返回降级回复，不 crash               │
└───────────────────────────┴──────────────────────────────────────┘

核心原则: Agent 永远不会 crash，最多返回一条有用的错误信息给用户。
""")


if __name__ == "__main__":
    demo_normal()
    demo_unknown_tool()
    demo_missing_param()
    demo_invalid_json()
    demo_tool_failure_retry()
    demo_error_summary()
