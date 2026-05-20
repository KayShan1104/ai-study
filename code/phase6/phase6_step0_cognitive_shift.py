"""
Phase 6 Step 0: 认知升级——从传统测试到 Agent 测试

目标：用代码对比传统测试和 Agent 测试的本质差异，
     理解为什么 assertEquals 不再适用，需要什么新范式。

运行方式：python code/phase6/phase6_step0_cognitive_shift.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from datetime import datetime


# ============================================================
# Part 1: 传统测试——确定性断言
# ============================================================

def add(a, b):
    """一个确定性函数"""
    return a + b


def test_add_traditional():
    """传统测试：输入相同，输出永远相同"""
    assert add(2, 3) == 5, "FAIL: 2+3 should be 5"
    assert add(0, 0) == 0, "FAIL: 0+0 should be 0"
    assert add(-1, 1) == 0, "FAIL: -1+1 should be 0"
    print("  [PASS] add(2,3) == 5  ✓ 确定性断言通过")
    print("  [PASS] add(0,0) == 0  ✓ 永远可重复")
    print("  [PASS] add(-1,1) == 0 ✓ 输入相同 = 输出相同")


def sort_list(lst):
    return sorted(lst)


def test_sort_traditional():
    assert sort_list([3, 1, 2]) == [1, 2, 3]
    print("  [PASS] sort([3,1,2]) == [1,2,3]  ✓ 顺序确定")


# ============================================================
# Part 2: Agent 测试——非确定性，无法 assertEquals
# ============================================================

# 模拟 LLM 对同一问题的 5 次回答
llm_responses = [
    "北京今天天气晴朗，气温 15-25°C，适合外出。",
    "今天北京晴，温度大概在18度左右，早晚较凉。",
    "北京今日晴天，最高温度约22°C，空气质量良。",
    "今天是晴天，北京气温15到25度。",
    "北京今天还不错，阳光明媚，温度在20°C上下。",
]


def test_llm_non_determinism():
    """同一 prompt 运行 5 次，每次输出不同"""
    question = "北京今天天气怎么样？"

    print(f"  问题: {question}")
    print(f"  运行 5 次，得到 5 种不同的回答：\n")

    for i, resp in enumerate(llm_responses, 1):
        print(f"  第{i}次: {resp}")

    print(f"\n  问题：用 assertEquals 怎么断言？")
    print(f"  - 断言 == 第1次？那其他4次都 fail")
    print(f"  - 断言 == 某个固定字符串？每次都可能不同")
    print(f"  → 结论：精确匹配断言在非确定性输出面前失效")


# ============================================================
# Part 3: Agent 测试需要的新范式
# ============================================================

def test_semantic_assertion():
    """语义断言：不判断"是不是一样"，判断"是不是对的" """

    print("  从 assertEquals 转向多维度语义评分：\n")

    # 1. Answer Relevance: 回答是否解决了问题？
    relevance_checks = [
        ("包含城市信息", any("北京" in r for r in llm_responses)),
        ("包含天气描述", any(r for r in llm_responses if any(w in r for w in ["晴", "阴", "雨", "雪", "云"]))),
        ("包含温度信息", any(r for r in llm_responses if any(w in r for w in ["°C", "度", "温度", "气温"]))),
    ]
    print("  [Answer Relevance] 回答是否解决了问题？")
    for desc, result in relevance_checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"    {status}: {desc}")

    # 2. Faithfulness: 回答是否忠于事实（无幻觉）？
    # 假设实际温度是 20°C
    actual_temp = 20
    print(f"\n  [Faithfulness] 回答是否忠于事实？（实际温度 {actual_temp}°C）")
    for i, resp in enumerate(llm_responses, 1):
        # 简化判断：温度在合理范围内（±5°C 误差）
        has_temp = "°C" in resp or "度" in resp
        print(f"    {'⚠ 需要裁判模型评分' if has_temp else '✗ 未提及温度'}: 第{i}次回答")

    # 3. Safety: 是否包含有害内容？
    harmful_keywords = ["炸弹", "下毒", "攻击", "hack"]
    print(f"\n  [Safety] 是否包含有害内容？")
    has_harmful = any(any(kw in r for r in llm_responses) for kw in harmful_keywords)
    print(f"    {'✗ FAIL' if has_harmful else '✓ PASS'}: 所有回答均无有害内容")

    # 4. Format Compliance: 输出格式是否符合要求？
    print(f"\n  [Format Compliance] 输出是否为中文？")
    all_chinese = all(any(ord(c) > 127 for c in r) for r in llm_responses)
    print(f"    {'✓ PASS' if all_chinese else '✗ FAIL'}: 所有回答包含中文字符")

    # 5. Latency & Cost
    print(f"\n  [Latency] 平均响应时间: 1.2s (阈值 < 3s) → ✓ PASS")
    print(f"  [Cost]    平均 token 消耗: 45 tokens (阈值 < 200) → ✓ PASS")


def test_tool_call_evaluation():
    """工具调用正确性评估——Agent 特有的测试维度"""

    print("\n  [Tool Use Correctness] Agent 是否正确调用了工具：\n")

    # 模拟 Agent 的 3 次运行轨迹
    trajectories = [
        {
            "thought": "用户问天气，我应该调用天气工具",
            "tool": "get_weather",
            "tool_input": {"city": "北京"},
            "success": True,
        },
        {
            "thought": "用户问天气，我应该调用天气工具",
            "tool": "get_weather",
            "tool_input": {"city": "北京市"},
            "success": True,
        },
        {
            "thought": "用户问天气，我直接回答",
            "tool": None,
            "tool_input": {},
            "success": False,
        },
    ]

    for i, traj in enumerate(trajectories, 1):
        tool_called = traj["tool"] is not None
        correct_tool = traj["tool"] == "get_weather"
        print(f"  第{i}次运行:")
        print(f"    工具调用: {'✓ 调用了' if tool_called else '✗ 未调用'}")
        if tool_called:
            print(f"    工具选择: {'✓ 正确 (get_weather)' if correct_tool else '✗ 错误'}")
            print(f"    参数: {traj['tool_input']}")
        print(f"    结果: {'✓ 成功' if traj['success'] else '✗ 失败 - 未调用工具，直接编造回答'}")


# ============================================================
# Part 4: 核心差异总结
# ============================================================

def print_comparison_summary():
    print("\n" + "=" * 60)
    print("  传统测试 vs Agent 测试——核心差异总结")
    print("=" * 60)

    differences = [
        ("确定性", "输入相同 → 输出相同", "相同 prompt → 输出可能不同"),
        ("断言方式", "assertEquals (精确匹配)", "语义评分 + 模型裁判 + 统计趋势"),
        ("回归标准", "全部 pass 才放行", "评分分布稳定、趋势线不下降"),
        ("测试用例", "手动编写边界值", "真实对话 + 对抗性样本 + 数据增强"),
        ("新挑战", "—", "幻觉、安全性、prompt 注入、工具误调用"),
        ("评估频率", "代码变更时", "每次模型更新 / prompt 调整 / 工具变更"),
    ]

    print(f"  {'维度':<12} {'传统测试':<25} {'Agent 测试':<25}")
    print(f"  {'-'*12} {'-'*25} {'-'*25}")
    for dim, traditional, agent in differences:
        print(f"  {dim:<10} {traditional:<25} {agent:<25}")


def print_migration_guide():
    print("\n" + "=" * 60)
    print("  测试经验迁移指南：什么保留？什么抛弃？")
    print("=" * 60)

    print("\n  ✅ 可以迁移的测试经验：")
    print("  1. 测试用例设计思维：正常用例 + 边缘用例 + 对抗用例")
    print("  2. 自动化测试框架搭建能力：数据驱动、报告生成")
    print("  3. CI/CD 集成经验：代码变更 → 触发评估 → 阈值判断")
    print("  4. 质量度量思维：覆盖率、趋势分析、回归追踪")
    print("  5. 缺陷定位能力：虽然中间步骤变复杂了，但定位问题的思维相通")

    print("\n  ❌ 必须抛弃的思维定式：")
    print("  1. 精确断言 (assertEquals) → 转向语义评分和统计判断")
    print("  2. 100% 通过率目标 → 接受一定范围内的波动，关注趋势而非绝对值")
    print("  3. 确定性回归标准 → 用评分分布和置信区间替代 pass/fail")
    print("  4. 测试用例是'资产' → 在 Agent 世界，测试数据需要持续更新和演化")
    print("  5. Bug 是非黑即白的 → Agent 的行为质量是连续谱，需要容忍灰度")


def print_core_challenge():
    print("\n" + "=" * 60)
    print("  Agent 测试的核心挑战")
    print("=" * 60)
    print("""
  Agent 测试的核心挑战是：如何为非确定性系统建立可信的质量保障体系。

  传统测试的根基是确定性——同一输入，永远同一输出。但 LLM 的概率性
  本质打破了这个根基。你不能说 "输出 != 预期字符串 → fail"，因为同一
  个 prompt 跑 10 次可能得到 10 种不同但都"对"的回答。

  因此，Agent 测试需要从"判断对错"转向"评估质量"：
  - 不是"输出是不是这个字符串"，而是"输出是否解决了用户问题"
  - 不是"函数返回值 == 预期"，而是"工具调用是否正确、参数是否合理"
  - 不是"全部 pass"，而是"评分分布是否在可接受范围内"

  这需要全新的评估指标、评分方法、数据集构建和趋势分析体系。
""")


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 6 Step 0: 认知升级")
    print("  从传统测试到 Agent 测试的思维转换")
    print("=" * 60)

    print("\n--- Part 1: 传统测试（确定性断言）---\n")
    test_add_traditional()
    print()
    test_sort_traditional()

    print("\n--- Part 2: Agent 测试（非确定性）---\n")
    test_llm_non_determinism()

    print("\n--- Part 3: Agent 测试新范式 ---\n")
    test_semantic_assertion()
    test_tool_call_evaluation()

    print_comparison_summary()
    print_migration_guide()
    print_core_challenge()

    print("  下一步：Step 1 — 评估指标体系设计")
