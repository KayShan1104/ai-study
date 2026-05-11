"""
Phase 2 Step 1: Prompt 的基本结构

学习目标:
- 理解 prompt 的 5 个核心要素：角色/人设、任务描述、上下文、输出格式要求、示例
- 对比实验：同一任务，3 个不同质量等级的 prompt，观察输出差异

运行方式: python code/phase2/phase2_step1_prompt_structure.py
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

# Windows terminal: force UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"

# ============================================================
# 测试任务：分析一段代码的问题
# ============================================================

CODE_SAMPLE = """
def get_user_data(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    result = cursor.fetchone()
    return result
"""

# ============================================================
# 3 个不同质量等级的 prompt
# ============================================================

# Level 1: 只有问题，没有任何结构
def build_level1_prompt(code):
    return f"这段代码有什么问题？\n\n{code}"


# Level 2: 有角色和任务，但缺少格式要求和示例
def build_level2_prompt(code):
    return f"""你是一个有经验的 Python 工程师。请分析以下代码中存在的问题。

代码：
{code}"""


# Level 3: 完整结构 — 角色 + 任务 + 上下文 + 输出格式 + 示例
def build_level3_prompt(code):
    return f"""你是一个资深 Python 代码审查员。请从以下 4 个维度审查代码：
1. 安全性（如 SQL 注入、硬编码等）
2. 健壮性（如异常处理、资源释放）
3. 性能（如低效操作、不必要的开销）
4. 可维护性（如命名规范、代码结构）

请用 JSON 格式输出，包含以下字段：
{{
  "security_issues": ["问题描述"],
  "robustness_issues": ["问题描述"],
  "performance_issues": ["问题描述"],
  "maintainability_issues": ["问题描述"],
  "overall_score": 0-10之间的整数,
  "summary": "一句话总结"
}}

示例：
输入：
def read_file(path):
    f = open(path)
    return f.read()

输出：
{{
  "security_issues": ["文件路径未校验，可能读取任意系统文件"],
  "robustness_issues": ["未使用 with 语句，文件未被关闭；无 try/except 处理文件不存在的情况"],
  "performance_issues": [],
  "maintainability_issues": ["函数名过于通用，缺少类型注解"],
  "overall_score": 4,
  "summary": "资源泄漏和路径校验是主要问题"
}}

待审查代码：
{code}

只输出 JSON，不要其他文字。"""


# ============================================================
# 调用 API
# ============================================================

def call_llm(prompt_text, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # 打印 prompt 结构分析
    elements = []
    if any(kw in prompt_text.lower() for kw in ["你", "role", "角色", "engineer", "审查员"]):
        elements.append("角色/人设")
    if any(kw in prompt_text.lower() for kw in ["请", "分析", "审查", "问题"]):
        elements.append("任务描述")
    if "代码" in prompt_text or "code" in prompt_text.lower():
        elements.append("上下文/背景")
    if "json" in prompt_text.lower() or "格式" in prompt_text:
        elements.append("输出格式要求")
    if "示例" in prompt_text or "example" in prompt_text.lower():
        elements.append("示例 (few-shot)")

    print(f"Prompt 包含要素: {', '.join(elements) if elements else '无'}")
    print(f"Prompt 长度: {len(prompt_text)} 字符")
    print("-" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()
    usage = response.usage

    print(content)
    print("-" * 60)
    print(f"Token 消耗: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
    print(f"费用估算: ¥{usage.prompt_tokens / 1000 * 0.0008 + usage.completion_tokens / 1000 * 0.002:.5f}")

    # 尝试解析 JSON
    if "json" in prompt_text.lower():
        try:
            data = json.loads(content)
            print("JSON 解析: 成功")
        except json.JSONDecodeError as e:
            print(f"JSON 解析: 失败 ({e})")

    return content


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("Phase 2 Step 1: Prompt 的基本结构")
    print(f"测试任务: 代码审查")
    print(f"模型: {MODEL}")

    prompts = [
        ("Level 1 — 只有问题", build_level1_prompt(CODE_SAMPLE)),
        ("Level 2 — 角色 + 任务", build_level2_prompt(CODE_SAMPLE)),
        ("Level 3 — 完整结构", build_level3_prompt(CODE_SAMPLE)),
    ]

    for label, prompt_text in prompts:
        call_llm(prompt_text, label)

    print("\n" + "=" * 60)
    print("  对比总结")
    print("=" * 60)
    print("""
Prompt 的 5 个核心要素:
  1. 角色/人设      — 设定 AI 的身份和行为边界
  2. 任务描述       — 明确要做什么
  3. 上下文/背景    — 提供必要信息（代码、数据、约束）
  4. 输出格式要求   — 控制输出结构，便于后续程序处理
  5. 示例(few-shot) — 让模型模仿模式，提升输出一致性

观察要点:
  - Level 1 的输出通常随意、不完整、难以程序化处理
  - Level 2 比 Level 1 更有针对性，但仍然没有结构约束
  - Level 3 的输出最规范，JSON 可直接被下游系统消费
""")
