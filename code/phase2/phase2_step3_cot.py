"""
Phase 2 Step 3: Chain-of-Thought (CoT)

学习目标:
- 理解 CoT 的核心：让模型"一步步思考"而不是直接给答案
- 对比直接回答 vs CoT 在需要推理的题目上的准确率
- 了解 Zero-shot CoT vs Few-shot CoT 的区别
- 量化 CoT 的 token 代价

运行: python code/phase2/phase2_step3_cot.py
"""

import os
import sys
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
# 5 道需要多步推理的题目
# ============================================================

TEST_PROBLEMS = [
    {
        "question": "一个农场有鸡和兔子，共有35个头、94只脚。鸡有2只脚，兔子有4只脚。问鸡和兔子各有多少只？",
        "answer": "鸡23只，兔子12只",
    },
    {
        "question": "小明今年8岁，爸爸的年龄是小明的4倍。请问爸爸比小明大多少岁？5年后爸爸比小明大多少岁？",
        "answer": "爸爸比小明大24岁，5年后仍然大24岁",
    },
    {
        "question": "一个水池有进水管和出水管。进水管单独开6小时可以注满水池，出水管单独开8小时可以排空水池。如果两管同时打开，需要多少小时才能注满水池？",
        "answer": "24小时",
    },
    {
        "question": "甲乙两人同时从A、B两地相向而行，甲每小时走5公里，乙每小时走4公里。两人在距中点3公里处相遇。求A、B两地的距离。",
        "answer": "54公里",
    },
    {
        "question": "有3个开关控制3个灯泡，3个开关在房间A，3个灯泡在房间B（看不到）。你只能去房间B一次。如何判断哪个开关控制哪个灯泡？",
        "answer": "先开开关1等几分钟关掉，再开开关2，然后去房间B：亮的灯对应开关2，热的灯对应开关1，不亮不热的对应开关3",
    },
]


def call_llm(messages, label=""):
    """调用 LLM，返回回复内容和 token 消耗"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    usage = response.usage
    return content, usage


def run_direct(problem):
    """方式 1：直接问，要求直接给答案"""
    prompt = problem["question"] + "\n直接给出答案。"
    content, usage = call_llm(
        [{"role": "user", "content": prompt}], "direct"
    )
    return content, usage


def run_zero_shot_cot(problem):
    """方式 2：Zero-shot CoT —— 只加"请一步步思考" """
    prompt = problem["question"] + "\n请一步步思考，先列出已知条件和思路，再求解。"
    content, usage = call_llm(
        [{"role": "user", "content": prompt}], "zero-shot CoT"
    )
    return content, usage


def run_few_shot_cot(problem):
    """方式 3：Few-shot CoT —— 示例中包含推理步骤"""
    example = """问题：一个笼子里有若干只鸡和鸭，共有10个头、28只脚。鸡有2只脚，鸭有2只脚。问鸡和鸭各有多少只？

思考过程：
1. 已知条件：总头数 = 10，总脚数 = 28
2. 设鸡有 x 只，鸭有 y 只
3. 列方程：
   x + y = 10  （头的数量）
   2x + 2y = 28 （脚的数量）
4. 简化：2(x + y) = 28 → x + y = 14，但 x + y = 10，矛盾。
   说明题目数据有误。但如果假设鸡有2只脚，鸭有2只脚，那么10只动物就是20只脚，不可能有28只脚。

答案：题目数据矛盾，无解。"""

    prompt = f"""请一步步思考以下数学题，先列出已知条件和思路，再求解。

示例：
{example}

问题：{problem["question"]}"""

    content, usage = call_llm(
        [{"role": "user", "content": prompt}], "few-shot CoT"
    )
    return content, usage


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("Phase 2 Step 3: Chain-of-Thought (CoT)")
    print(f"模型: {MODEL}")
    print(f"测试题数: {len(TEST_PROBLEMS)}")

    methods = [
        ("Direct（直接回答）", run_direct),
        ("Zero-shot CoT", run_zero_shot_cot),
        ("Few-shot CoT", run_few_shot_cot),
    ]

    results = {}
    for method_name, method_fn in methods:
        print(f"\n{'='*60}")
        print(f"  {method_name}")
        print(f"{'='*60}")

        correct = 0
        total_tokens = 0

        for i, problem in enumerate(TEST_PROBLEMS):
            content, usage = method_fn(problem)
            total_tokens += usage.total_tokens

            print(f"\n  --- 题 {i+1} ---")
            print(f"  问题: {problem['question'][:50]}...")
            print(f"  期望答案: {problem['answer']}")
            # 打印回复的前 200 字符
            preview = content[:200]
            print(f"  回复: {preview}{'...' if len(content) > 200 else ''}")

            # 简单判断：期望答案的关键字是否在回复中
            # 注意：这是粗略判断，实际应该更严格
            keywords = [problem["answer"].split("各")[0],
                        problem["answer"].split("大")[0] if "大" in problem["answer"] else problem["answer"][:3]]
            is_correct = any(kw in content for kw in keywords if len(kw) >= 2)
            # 对于第5题（开关灯泡），单独判断
            if i == 4:
                is_correct = ("热" in content and "亮" in content) or "开" in content

            if is_correct:
                correct += 1
                print(f"  判断: ✅ 正确")
            else:
                print(f"  判断: ❌ 错误")

            print(f"  tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")

        print(f"\n  合计: {correct}/{len(TEST_PROBLEMS)} 正确")
        print(f"  总 token 消耗: {total_tokens}")
        results[method_name] = {"correct": correct, "total_tokens": total_tokens}

    # 汇总
    print("\n" + "=" * 60)
    print("  汇总对比")
    print("=" * 60)

    print(f"\n| 方式 | 正确率 | 总 Token | 单题平均 Token |")
    print(f"|------|--------|----------|----------------|")
    for name, data in results.items():
        avg = data["total_tokens"] // len(TEST_PROBLEMS)
        print(f"| {name} | {data['correct']}/{len(TEST_PROBLEMS)} | {data['total_tokens']} | {avg} |")

    print(f"""
关键观察点:
  - CoT 是否提升了正确率？
  - Zero-shot CoT vs Few-shot CoT 哪个更好？
  - CoT 的 token 代价是多少？（completion tokens 的增长）
  - 哪些题目在 direct 模式下错了，CoT 模式下对了？
""")
