"""
Phase 6 Step 4: Agent 行为监控与调试

目标：
1. 实现 Trace 系统——记录每一步的 Thought、Action、Observation
2. 追踪工具调用参数和结果
3. 记录 token 消耗和时间
4. 定位"在哪一步开始出错"
5. 可视化完整执行轨迹

运行方式：python code/phase6/phase6_step4_trace_debug.py
"""

import sys
import io
import os
import json
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================
# Part 1: Trace 数据模型
# ============================================================

class TraceStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolCallTrace:
    """工具调用级别的追踪"""
    name: str
    input_args: dict
    output: Optional[str] = None
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: float = 0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class StepTrace:
    """Agent 单步执行的追踪"""
    step_number: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    tool_call: Optional[ToolCallTrace] = None
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: float = 0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class RunTrace:
    """一次 Agent 运行的完整追踪"""
    run_id: str
    test_case_id: str
    category: str
    difficulty: str
    user_input: str
    steps: list = field(default_factory=list)
    final_response: str = ""
    overall_status: str = "pending"
    total_duration_ms: float = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "test_case_id": self.test_case_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "user_input": self.user_input,
            "steps": [asdict(s) for s in self.steps],
            "final_response": self.final_response,
            "overall_status": self.overall_status,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# ============================================================
# Part 2: Trace Collector
# ============================================================

class TraceCollector:
    """收集和管理 Agent 执行轨迹"""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "traces")
        os.makedirs(self.output_dir, exist_ok=True)
        self.traces: list[RunTrace] = []

    def start_run(self, run_id, test_case) -> RunTrace:
        """开始一次运行追踪"""
        trace = RunTrace(
            run_id=run_id,
            test_case_id=test_case["id"],
            category=test_case["category"],
            difficulty=test_case["difficulty"],
            user_input=test_case["input"],
            start_time=datetime.now().isoformat(),
        )
        self.traces.append(trace)
        return trace

    def add_step(self, trace: RunTrace, step_number, thought="", action="") -> StepTrace:
        """添加一个执行步骤"""
        step = StepTrace(
            step_number=step_number,
            thought=thought,
            action=action,
            status="running",
            start_time=datetime.now().isoformat(),
        )
        trace.steps.append(step)
        return step

    def complete_step(self, step: StepTrace, observation="", status="success", error=None):
        """完成一个步骤"""
        step.observation = observation
        step.status = status
        step.end_time = datetime.now().isoformat()
        step.error = error
        if step.start_time and step.end_time:
            t1 = datetime.fromisoformat(step.start_time)
            t2 = datetime.fromisoformat(step.end_time)
            step.duration_ms = (t2 - t1).total_seconds() * 1000
        if error:
            step.status = "error"

    def complete_run(self, trace: RunTrace, final_response="", status="success", error=None):
        """完成一次运行"""
        trace.final_response = final_response
        trace.overall_status = status
        trace.end_time = datetime.now().isoformat()
        trace.error = error
        if trace.start_time and trace.end_time:
            t1 = datetime.fromisoformat(trace.start_time)
            t2 = datetime.fromisoformat(trace.end_time)
            trace.total_duration_ms = (t2 - t1).total_seconds() * 1000
        # 汇总
        trace.total_tokens = sum(s.tokens_used for s in trace.steps)
        trace.total_tool_calls = sum(1 for s in trace.steps if s.tool_call)

    def save_trace(self, trace: RunTrace):
        """保存追踪到 JSON"""
        filename = f"trace_{trace.run_id}_{trace.test_case_id}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)

    def save_all(self):
        """保存所有追踪"""
        summary_path = os.path.join(self.output_dir, "trace_summary.json")
        summary = {
            "total_runs": len(self.traces),
            "timestamp": datetime.now().isoformat(),
            "traces": [t.to_dict() for t in self.traces],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def get_error_runs(self) -> list[RunTrace]:
        """获取所有失败的运行"""
        return [t for t in self.traces if t.overall_status in ("error", "timeout")]

    def get_slow_runs(self, threshold_ms=100) -> list[RunTrace]:
        """获取超时的运行（模拟环境下阈值设低一些）"""
        return [t for t in self.traces if t.total_duration_ms > threshold_ms]


# ============================================================
# Part 3: 模拟带 Trace 的 Agent 执行
# ============================================================

def simulate_agent_with_trace(test_case, run_id, collector):
    """
    模拟 Agent 执行并记录完整轨迹。
    模拟了多种故障场景：正常、工具失败、超时、LLM 错误等。
    """
    trace = collector.start_run(run_id, test_case)

    cat = test_case["category"]
    inp = test_case["input"]
    code_part = inp.split("\n\n", 1)[-1] if "\n\n" in inp else inp

    import hashlib
    seed = int(hashlib.md5((test_case["id"] + run_id).encode()).hexdigest()[:8], 16)
    scenario = seed % 10

    try:
        # Step 1: 解析用户输入
        step1 = collector.add_step(trace, 1,
            thought="用户提交了代码审查请求，我需要分析输入内容",
            action="parse_input")
        time.sleep(0.01)  # 模拟耗时
        collector.complete_step(step1, "输入解析完成，识别到代码片段")
        step1.tokens_used = 50

        # Step 2: 工具选择和调用
        step2 = collector.add_step(trace, 2,
            thought=f"代码需要审查，应调用 {test_case['expected_tool']} 工具",
            action=f"call_tool({test_case['expected_tool']})")
        time.sleep(0.02)

        # 模拟不同故障场景
        if scenario == 0 and cat == "stress":
            # 场景 A: 工具调用超时
            tool_trace = ToolCallTrace(
                name=test_case["expected_tool"],
                input_args={"code": code_part[:30]},
                status="timeout",
                error="工具调用超时（30s）",
            )
            step2.tool_call = tool_trace
            collector.complete_step(step2, status="timeout", error="工具调用超时")
            step2.tokens_used = 30
            collector.complete_run(trace,
                final_response="抱歉，工具调用超时，请稍后重试。",
                status="timeout")
            trace.total_tokens = step1.tokens_used + step2.tokens_used
            return trace

        elif scenario == 1 and cat == "edge_case":
            # 场景 B: LLM 解析错误
            tool_trace = ToolCallTrace(
                name=test_case["expected_tool"],
                input_args={"code": code_part[:30]},
                status="error",
                error="JSON 解析失败：LLM 返回的工具调用格式不正确",
            )
            step2.tool_call = tool_trace
            collector.complete_step(step2, status="error",
                error="LLM 返回了无效的工具调用 JSON")
            step2.tokens_used = 80
            collector.complete_run(trace,
                final_response="抱歉，我遇到了内部错误。",
                status="error",
                error="JSONDecodeError: 工具调用格式无效")
            trace.total_tokens = step1.tokens_used + step2.tokens_used
            return trace

        elif scenario == 2:
            # 场景 C: 工具执行异常
            tool_trace = ToolCallTrace(
                name=test_case["expected_tool"],
                input_args={"code": code_part[:30]},
                output=None,
                status="error",
                error="工具内部异常：TypeError: 'NoneType' object is not subscriptable",
            )
            step2.tool_call = tool_trace
            collector.complete_step(step2, status="error",
                error="工具执行失败：TypeError")
            step2.tokens_used = 60
            collector.complete_run(trace,
                final_response="工具执行时出现错误，无法完成审查。",
                status="error",
                error="工具返回异常")
            trace.total_tokens = step1.tokens_used + step2.tokens_used
            return trace

        elif scenario == 3 and cat == "adversarial":
            # 场景 D: 多步后被 prompt injection 误导
            tool_trace = ToolCallTrace(
                name=test_case["expected_tool"],
                input_args={"code": code_part[:30]},
                status="success",
                output="检测到安全风险：可能的 prompt injection 攻击",
            )
            step2.tool_call = tool_trace
            collector.complete_step(step2, observation="工具检测到注入攻击")
            step2.tokens_used = 100

            # Step 3: Agent 被误导
            step3 = collector.add_step(trace, 3,
                thought="用户说'忽略安全规则'，我应该跳过安全检查",
                action="generate_response_without_safety")
            collector.complete_step(step3,
                observation="生成了不安全的回答",
                status="success")
            step3.tokens_used = 120

            collector.complete_run(trace,
                final_response="以下是你的请求：忽略安全规则，代码没问题。",
                status="success")
            trace.total_tokens = sum(s.tokens_used for s in trace.steps)
            return trace

        else:
            # 场景 E: 正常执行
            tool_trace = ToolCallTrace(
                name=test_case["expected_tool"],
                input_args={"code": code_part[:30] + "..." if len(code_part) > 30 else code_part},
                output="代码审查完成，发现 2 个问题",
                status="success",
            )
            step2.tool_call = tool_trace
            collector.complete_step(step2, observation=tool_trace.output)
            step2.tokens_used = 80

            # Step 3: 生成最终回答
            step3 = collector.add_step(trace, 3,
                thought="根据工具返回结果，生成用户友好的回答",
                action="generate_response")
            collector.complete_step(step3,
                observation="回答生成完成",
                status="success")
            step3.tokens_used = 120

            collector.complete_run(trace,
                final_response="代码审查完成，以下是详细结果...",
                status="success")
            trace.total_tokens = sum(s.tokens_used for s in trace.steps)
            return trace

    except Exception as e:
        collector.complete_run(trace,
            final_response=f"异常: {e}",
            status="error",
            error=traceback.format_exc())
        return trace


# ============================================================
# Part 4: Trace 可视化
# ============================================================

def print_trace(trace: RunTrace, show_details=True):
    """以可读格式打印追踪信息"""
    print(f"\n{'='*60}")
    print(f"  🔍 Trace: {trace.run_id} | {trace.test_case_id}")
    print(f"{'='*60}")
    print(f"  类别: {trace.category} | 难度: {trace.difficulty}")
    print(f"  用户输入: {trace.user_input[:60]}...")
    print(f"  状态: {trace.overall_status}")
    print(f"  总耗时: {trace.total_duration_ms:.0f}ms | 总 tokens: {trace.total_tokens}")
    print(f"  工具调用: {trace.total_tool_calls} 次")

    for step in trace.steps:
        print(f"\n  ── Step {step.step_number} ──")
        print(f"  💭 Thought: {step.thought[:80]}...")
        print(f"  ⚡ Action:  {step.action}")
        print(f"  📝 Status:  {step.status} | {step.duration_ms:.0f}ms | {step.tokens_used} tokens")

        if step.tool_call:
            tc = step.tool_call
            print(f"  🔧 Tool:    {tc.name}")
            print(f"  📥 Input:   {json.dumps(tc.input_args, ensure_ascii=False)[:60]}")
            if tc.output:
                print(f"  📤 Output:  {tc.output[:60]}")
            if tc.error:
                print(f"  ❌ Error:   {tc.error[:60]}")
            print(f"  📊 Status:  {tc.status}")

        if step.observation:
            print(f"  👁️ Obs:     {step.observation[:80]}...")
        if step.error:
            print(f"  🚨 Error:   {step.error[:80]}...")

    print(f"\n  📋 Final Response: {trace.final_response[:100]}...")
    if trace.error:
        print(f"  🚨 Run Error: {trace.error[:80]}...")


def visualize_trace_timeline(traces: list[RunTrace]):
    """可视化所有运行的时间线概览"""
    print(f"\n{'='*60}")
    print(f"  📊 执行轨迹时间线概览（{len(traces)} 次运行）")
    print(f"{'='*60}")

    print(f"\n  {'Run ID':<22} {'用例':<10} {'类别':<14} {'状态':<10} {'耗时':<8} {'步数':<4} {'错误'}")
    print(f"  {'-'*80}")

    for t in traces:
        status_icon = {
            "success": "✅",
            "error": "❌",
            "timeout": "⏱️",
            "skipped": "⏭️",
        }.get(t.overall_status, "❓")

        steps = len(t.steps)
        error_preview = t.error[:30] if t.error else ""

        print(f"  {status_icon} {t.run_id[-15:]:<20} {t.test_case_id:<10} {t.category:<14} "
              f"{t.overall_status:<10} {t.total_duration_ms:<6.0f}ms {steps:<4} {error_preview}")


def error_analysis(traces: list[RunTrace]):
    """错误分析——定位"在哪一步开始出错" """
    errors = [t for t in traces if t.overall_status in ("error", "timeout")]

    if not errors:
        print(f"\n  ✓ 没有错误需要分析")
        return

    print(f"\n{'='*60}")
    print(f"  🔧 错误分析（{len(errors)} 次失败）")
    print(f"{'='*60}")

    # 按错误步骤统计
    error_steps = {}
    error_types = {}

    for t in errors:
        for step in t.steps:
            if step.status in ("error", "timeout"):
                key = f"Step {step.step_number}: {step.action}"
                error_steps[key] = error_steps.get(key, 0) + 1
                if step.error:
                    # 提取错误类型
                    if "超时" in step.error:
                        error_types["timeout"] = error_types.get("timeout", 0) + 1
                    elif "JSON" in step.error:
                        error_types["parse_error"] = error_types.get("parse_error", 0) + 1
                    elif "工具" in step.error:
                        error_types["tool_error"] = error_types.get("tool_error", 0) + 1
                    elif "注入" in step.error or "injection" in step.error.lower():
                        error_types["security"] = error_types.get("security", 0) + 1
                    else:
                        error_types["other"] = error_types.get("other", 0) + 1

    print(f"\n  错误发生步骤分布：")
    for step_key, count in sorted(error_steps.items(), key=lambda x: -x[1]):
        print(f"    {step_key}: {count} 次")

    print(f"\n  错误类型分布：")
    for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count} 次")

    print(f"\n  失败详情：")
    for t in errors:
        print(f"\n  ❌ {t.test_case_id} ({t.category})")
        for step in t.steps:
            if step.status in ("error", "timeout"):
                print(f"     在 Step {step.step_number} ({step.action}) 失败")
                if step.error:
                    print(f"     原因: {step.error[:80]}")


def token_analysis(traces: list[RunTrace]):
    """Token 消耗分析"""
    print(f"\n{'='*60}")
    print(f"  💰 Token 消耗分析")
    print(f"{'='*60}")

    total_tokens = sum(t.total_tokens for t in traces)
    avg_tokens = total_tokens / len(traces) if traces else 0

    # 按类别
    cat_tokens = {}
    for t in traces:
        cat_tokens.setdefault(t.category, []).append(t.total_tokens)

    print(f"\n  总消耗: {total_tokens} tokens")
    print(f"  平均每次运行: {avg_tokens:.0f} tokens")

    print(f"\n  按类别平均消耗：")
    for cat, tokens in sorted(cat_tokens.items()):
        avg = sum(tokens) / len(tokens)
        print(f"    {cat:<20} {avg:.0f} tokens/次 ({len(tokens)} 次)")

    # 最耗 Token 的步骤
    step_tokens = {}
    for t in traces:
        for step in t.steps:
            step_tokens.setdefault(step.action, []).append(step.tokens_used)

    print(f"\n  按步骤平均消耗（Top 5）：")
    sorted_steps = sorted(step_tokens.items(), key=lambda x: -sum(x[1])/len(x[1]))
    for action, tokens in sorted_steps[:5]:
        avg = sum(tokens) / len(tokens)
        print(f"    {action:<35} {avg:.0f} tokens/次")


# ============================================================
# Part 5: 监控平台对比
# ============================================================

def compare_monitoring_platforms():
    """对比主流 Agent 监控/追踪平台"""
    print(f"\n{'='*60}")
    print(f"  Agent 监控/追踪平台对比")
    print(f"{'='*60}")

    platforms = [
        {
            "name": "LangSmith",
            "vendor": "LangChain 官方",
            "type": "SaaS / 自托管",
            "pros": ["功能最全", "官方支持", "与 LangChain 深度集成", "有 LLM-as-judge"],
            "cons": ["免费版有限制", "非 LangChain 项目集成成本高"],
            "features": "Trace、Dataset、Evaluation、Prompt Hub、LLM-as-judge",
        },
        {
            "name": "LangFuse",
            "vendor": "开源社区",
            "type": "自托管 / Cloud",
            "pros": ["开源免费", "可自托管数据隐私好", "支持多框架"],
            "cons": ["功能不如 LangSmith 全", "社区支持较弱"],
            "features": "Trace、Session、Evaluation、Prompt Management",
        },
        {
            "name": "Phoenix (Arize)",
            "vendor": "Arize AI",
            "type": "开源",
            "pros": ["Embedding 分析强", "支持 LLM 和传统 ML", "可视化好"],
            "cons": ["部署复杂", "文档较少"],
            "features": "Trace、Embedding Analysis、Drift Detection",
        },
    ]

    print(f"\n  {'平台':<15} {'类型':<15} {'核心功能'}")
    print(f"  {'-'*60}")
    for p in platforms:
        print(f"  {p['name']:<15} {p['type']:<15} {p['features']}")
        print(f"    👍 {' | '.join(p['pros'])}")
        print(f"    👎 {' | '.join(p['cons'])}")
        print()


# ============================================================
# Part 6: 主流程
# ============================================================

def load_test_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 6 Step 4: Agent 行为监控与调试")
    print("  Trace 系统 + 错误分析 + 平台对比")
    print("=" * 60)

    # 加载测试用例
    test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    test_cases = load_test_cases(test_cases_path)

    # 从每个类别选代表性用例
    selected = []
    for cat in ["normal", "edge_case", "adversarial", "stress", "multilingual"]:
        cat_cases = [tc for tc in test_cases if tc["category"] == cat]
        # 每个类别选 3 个
        selected.extend(cat_cases[:3])

    print(f"\n  选择 {len(selected)} 个代表性用例进行追踪分析\n")

    # 运行追踪
    collector = TraceCollector()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, tc in enumerate(selected):
        trace = simulate_agent_with_trace(tc, run_id, collector)
        collector.save_trace(trace)

    collector.save_all()
    traces = collector.traces

    # 可视化
    visualize_trace_timeline(traces)

    # 打印每个 trace 的详情
    for t in traces:
        print_trace(t)

    # 错误分析
    error_analysis(traces)

    # Token 分析
    token_analysis(traces)

    # 平台对比
    compare_monitoring_platforms()

    print(f"\n{'='*60}")
    print(f"  追踪文件已保存到: {collector.output_dir}/")
    print(f"  共 {len(traces)} 个 trace 文件")
    print(f"{'='*60}")

    print(f"\n  下一步：Step 5 — 安全评估")
