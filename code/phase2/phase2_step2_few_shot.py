"""
Phase 2 Step 2: Few-Shot Prompting

学习目标:
- 理解 few-shot 的原理：给模型 2-5 个输入-输出示例，让它模仿模式
- 对比 zero-shot vs few-shot 的分类准确率
- 尝试 1-shot、3-shot、5-shot，观察准确率变化

运行: python code/phase2/phase2_step2_few_shot.py
"""

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"

# ============================================================
# 意图分类任务：将用户输入分类为 query / command / chitchat / creative
# ============================================================

# 10 个测试用例（输入, 期望类别, 期望置信度区间）
# 前 10 个是简单 case，后 10 个是模糊/边缘 case
TEST_CASES = [
    # 简单 case
    ("明天北京天气怎么样", "query", 0.9),
    ("打开空调", "command", 0.9),
    ("你好呀", "chitchat", 0.95),
    ("帮我查一下上海到广州的机票", "query", 0.9),
    ("写一首关于春天的诗", "creative", 0.9),
    ("把屏幕亮度调到最低", "command", 0.9),
    ("你今天心情好吗", "chitchat", 0.9),
    ("Python 中列表和元组有什么区别", "query", 0.9),
    ("讲个笑话", "creative", 0.85),
    ("今天星期几", "query", 0.9),
    # 模糊/边缘 case（真正需要 few-shot 来区分）
    ("帮我关掉卧室的灯", "command", 0.85),      # "帮我" 开头，可能被误判为 query
    ("我有点无聊", "chitchat", 0.7),              # 没有明确请求，暗示性闲聊
    ("北京和上海哪个城市更大", "query", 0.85),     # 比较类查询
    ("用 Python 写一个快速排序", "command", 0.8), # 写代码，边界：creative vs command
    ("你觉得我应该学 Go 还是 Rust", "chitchat", 0.6),  # 征求建议，边界
    ("现在几点了", "query", 0.8),                  # 简单事实查询
    ("帮我分析一下这个日志", "command", 0.7),      # 模糊指令
    ("今天天气真好啊", "chitchat", 0.8),           # 感叹句，非查询
    ("给我推荐几部科幻电影", "query", 0.7),        # 推荐类，边界 query vs creative
    ("太感谢你了！", "chitchat", 0.95),            # 纯情感表达
]


def build_zero_shot_prompt(user_input):
    """Zero-shot: 只给任务描述，不给示例"""
    return f"""将用户输入分类为以下类别：
- query: 查询类问题（天气、知识查询等）
- command: 命令类操作（设备控制、功能调用等）
- chitchat: 日常闲聊（打招呼、闲聊等）
- creative: 创意类请求（写诗、写故事、讲笑话等）

请用 JSON 格式输出：{{"category": "类别", "confidence": 0.0-1.0之间的小数}}
只输出 JSON，不要其他文字。

输入："{user_input}"
"""


def build_few_shot_prompt(user_input, examples):
    """Few-shot: 给 N 个示例 + 待分类输入"""
    examples_text = "\n".join(
        f'输入："{inp}" -> {{"category": "{cat}", "confidence": {conf}}}'
        for inp, cat, conf in examples
    )
    return f"""将用户输入分类为以下类别：
- query: 查询类问题（天气、知识查询等）
- command: 命令类操作（设备控制、功能调用等）
- chitchat: 日常闲聊（打招呼、闲聊等）
- creative: 创意类请求（写诗、写故事、讲笑话等）

示例：
{examples_text}

请用 JSON 格式输出：{{"category": "类别", "confidence": 0.0-1.0之间的小数}}
只输出 JSON，不要其他文字。

输入："{user_input}"
"""


def call_llm(prompt_text):
    """调用 LLM 并解析 JSON 输出"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {content[:100]}"


def run_tests(prompt_builder, label, examples=None):
    """跑 10 个测试用例，统计正确率"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    correct = 0
    results = []

    for user_input, expected_cat, _ in TEST_CASES:
        if examples:
            prompt = prompt_builder(user_input, examples)
        else:
            prompt = prompt_builder(user_input)

        data, error = call_llm(prompt)

        if error:
            actual_cat = f"解析失败 ({error})"
            is_correct = False
        else:
            actual_cat = data.get("category", "unknown")
            is_correct = actual_cat == expected_cat

        if is_correct:
            correct += 1

        results.append({
            "input": user_input,
            "expected": expected_cat,
            "actual": actual_cat,
            "correct": is_correct,
        })

        icon = "✅" if is_correct else "❌"
        print(f"  {icon} 输入: {user_input!r}")
        print(f"       期望: {expected_cat}, 实际: {actual_cat}")

    print(f"\n  正确率: {correct}/{len(TEST_CASES)} = {correct/len(TEST_CASES)*100:.0f}%")
    return correct, results


# ============================================================
# Few-shot 示例池（每个类别选 1 个典型示例）
# ============================================================

EXAMPLES_1_SHOT = [
    ("明天北京天气怎么样", "query", 0.95),
]

EXAMPLES_3_SHOT = [
    ("明天北京天气怎么样", "query", 0.95),
    ("打开空调", "command", 0.9),
    ("你好呀", "chitchat", 0.99),
]

EXAMPLES_5_SHOT = [
    ("明天北京天气怎么样", "query", 0.95),
    ("打开空调", "command", 0.9),
    ("你好呀", "chitchat", 0.99),
    ("写一首关于春天的诗", "creative", 0.95),
    ("Python 中列表和元组有什么区别", "query", 0.9),
]


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("Phase 2 Step 2: Few-Shot Prompting")
    print(f"任务: 意图分类（query / command / chitchat / creative）")
    print(f"模型: {MODEL}")
    print(f"测试用例数: {len(TEST_CASES)}（前10个简单，后10个模糊/边缘）")

    # Zero-shot
    zero_correct, zero_results = run_tests(
        build_zero_shot_prompt, "Zero-Shot（0 个示例）"
    )

    # 1-shot
    one_correct, one_results = run_tests(
        build_few_shot_prompt, "1-Shot（1 个示例）", EXAMPLES_1_SHOT
    )

    # 3-shot
    three_correct, three_results = run_tests(
        build_few_shot_prompt, "3-Shot（3 个示例）", EXAMPLES_3_SHOT
    )

    # 5-shot
    five_correct, five_results = run_tests(
        build_few_shot_prompt, "5-Shot（5 个示例）", EXAMPLES_5_SHOT
    )

    # 汇总对比
    print("\n" + "=" * 60)
    print("  汇总对比")
    print("=" * 60)
    print(f"""
| 方式 | 正确数 | 正确率 |
|------|--------|--------|
| Zero-Shot | {zero_correct}/{len(TEST_CASES)} | {zero_correct/len(TEST_CASES)*100:.0f}% |
| 1-Shot | {one_correct}/{len(TEST_CASES)} | {one_correct/len(TEST_CASES)*100:.0f}% |
| 3-Shot | {three_correct}/{len(TEST_CASES)} | {three_correct/len(TEST_CASES)*100:.0f}% |
| 5-Shot | {five_correct}/{len(TEST_CASES)} | {five_correct/len(TEST_CASES)*100:.0f}% |

关键观察点:
  - Few-shot 是否比 zero-shot 准确率更高？
  - 增加到 5-shot 后是否有边际递减？
  - 哪些 case 在所有方式下都错了？是 prompt 问题还是模型能力问题？
""")
