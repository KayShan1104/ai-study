"""
Phase 2 Step 4: Structured Output 进阶

学习目标:
- 对比 prompt 口头要求 JSON vs response_format 强制 JSON 的区别
- 学习使用 OpenAI 兼容格式的 JSON Schema 约束输出
- 实现代码审查场景的结构化输出

运行: python code/phase2/phase2_step4_structured_output.py
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
# 测试代码样本（故意包含多种问题）
# ============================================================

CODE_SAMPLES = [
    # 样本 1：SQL 注入 + 资源泄漏
    """
def get_user(user_id):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    sql = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(sql)
    return cursor.fetchone()
""",
    # 样本 2：密码硬编码 + 无异常处理
    """
def connect_db():
    import pymysql
    conn = pymysql.connect(
        host="192.168.1.100",
        user="root",
        password="admin123",
        database="production"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders")
    return cursor.fetchall()
""",
    # 样本 3：XSS + 无输入验证
    """
@app.route('/search')
def search():
    query = request.args.get('q')
    return f"<h1>搜索结果: {query}</h1>"
""",
]

# ============================================================
# 方式 1：Prompt 口头要求 JSON（Phase 1 的做法）
# ============================================================

CODE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "security_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "问题类型，如 SQL Injection, XSS, Hardcoded Password"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "description": {"type": "string"},
                    "line_hint": {"type": "string", "description": "问题所在的代码片段"},
                    "fix_suggestion": {"type": "string"},
                },
                "required": ["type", "severity", "description", "fix_suggestion"],
                "additionalProperties": False,
            },
        },
        "code_quality": {
            "type": "object",
            "properties": {
                "maintainability_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["maintainability_score", "suggestions"],
            "additionalProperties": False,
        },
        "overall_verdict": {"type": "string", "enum": ["approve", "request_changes", "reject"]},
    },
    "required": ["security_issues", "code_quality", "overall_verdict"],
    "additionalProperties": False,
}


def build_prompt_review(code):
    """方式 1：Prompt 中口头要求 JSON"""
    return f"""你是一个资深代码审查员。请审查以下代码，从安全性和代码质量两个维度给出评估。

请用 JSON 格式输出，包含以下字段：
- security_issues: 数组，每个元素包含 type（问题类型）、severity（严重程度：critical/high/medium/low）、description（描述）、line_hint（问题代码片段）、fix_suggestion（修复建议）
- code_quality: 对象，包含 maintainability_score（1-10 的整数）、suggestions（改进建议数组）
- overall_verdict: 字符串，值为 approve / request_changes / reject 之一

代码：
{code}

只输出 JSON，不要其他文字。"""


def run_prompt_review(code, label):
    """调用方式 1"""
    messages = [
        {"role": "user", "content": build_prompt_review(code)},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    usage = response.usage

    # 尝试解析 JSON
    try:
        data = json.loads(content)
        parsed = True
    except json.JSONDecodeError:
        # 尝试提取 JSON 块（有时模型会包裹在 ```json ... ``` 中）
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            parsed = True
        except (ValueError, json.JSONDecodeError):
            data = None
            parsed = False

    return content, data, parsed, usage


# ============================================================
# 方式 2：使用 response_format JSON Schema 约束
# ============================================================

def run_schema_review(code, label):
    """方式 2：使用 response_format JSON Schema"""
    messages = [
        {"role": "system", "content": "你是一个资深代码审查员。请审查代码，从安全性和代码质量两个维度给出评估。"},
        {"role": "user", "content": f"请审查以下代码：\n{code}"},
    ]

    # qwen-plus 支持 response_format，但 JSON Schema mode 需要测试
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "code_review",
                    "schema": CODE_REVIEW_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content.strip()
        usage = response.usage
        data = json.loads(content)
        return content, data, True, usage, None
    except Exception as e:
        # 如果 JSON Schema mode 不支持，降级到普通 JSON mode
        print(f"  ⚠ JSON Schema mode 不可用: {e}")
        print(f"  降级使用 response_format: {{'type': 'json_object'}}")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            usage = response.usage
            data = json.loads(content)
            return content, data, True, usage, "降级为 json_object mode"
        except Exception as e2:
            return "", None, False, None, str(e2)


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    print("Phase 2 Step 4: Structured Output 进阶")
    print(f"模型: {MODEL}")
    print(f"测试代码样本数: {len(CODE_SAMPLES)}")

    # 方式 1：Prompt 口头要求 JSON
    print("\n" + "=" * 60)
    print("  方式 1：Prompt 口头要求 JSON")
    print("=" * 60)

    prompt_results = []
    for i, code in enumerate(CODE_SAMPLES):
        print(f"\n  --- 样本 {i+1} ---")
        preview = code.strip()[:60]
        print(f"  代码: {preview}...")
        content, data, parsed, usage = run_prompt_review(code, f"样本 {i+1}")
        print(f"  JSON 解析: {'成功' if parsed else '失败'}")
        print(f"  tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")

        if data:
            verdict = data.get("overall_verdict", "N/A")
            issues_count = len(data.get("security_issues", []))
            score = data.get("code_quality", {}).get("maintainability_score", "N/A")
            print(f"  审查结果: verdict={verdict}, 安全问题={issues_count}个, 质量分={score}")

        prompt_results.append(parsed)

    # 方式 2：response_format JSON Schema
    print("\n" + "=" * 60)
    print("  方式 2：response_format JSON Schema 约束")
    print("=" * 60)

    schema_results = []
    for i, code in enumerate(CODE_SAMPLES):
        print(f"\n  --- 样本 {i+1} ---")
        preview = code.strip()[:60]
        print(f"  代码: {preview}...")
        content, data, parsed, usage, fallback = run_schema_review(code, f"样本 {i+1}")
        if fallback:
            print(f"  降级: {fallback}")
        print(f"  JSON 解析: {'成功' if parsed else '失败'}")
        if usage:
            print(f"  tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")

        if data:
            verdict = data.get("overall_verdict", "N/A")
            issues_count = len(data.get("security_issues", []))
            score = data.get("code_quality", {}).get("maintainability_score", "N/A")
            print(f"  审查结果: verdict={verdict}, 安全问题={issues_count}个, 质量分={score}")

        schema_results.append(parsed)

    # 10 次连续调用测试（验证稳定性）
    print("\n" + "=" * 60)
    print("  稳定性测试：连续 10 次调用（使用方式 1）")
    print("=" * 60)

    stability_count = 0
    for j in range(10):
        code = CODE_SAMPLES[j % len(CODE_SAMPLES)]
        _, _, parsed, usage = run_prompt_review(code, f"第{j+1}次")
        if parsed:
            stability_count += 1
        icon = "✅" if parsed else "❌"
        print(f"  {icon} 第 {j+1} 次调用: {'解析成功' if parsed else '解析失败'}")

    # 汇总
    print("\n" + "=" * 60)
    print("  汇总对比")
    print("=" * 60)

    prompt_ok = sum(prompt_results)
    schema_ok = sum(schema_results)
    print(f"""
| 方式 | 3 个样本解析成功数 | 10 次连续调用解析成功数 |
|------|---------------------|-------------------------|
| Prompt 口头要求 JSON | {prompt_ok}/3 | {stability_count}/10 |
| response_format JSON Schema | {schema_ok}/3 | — |

additionalProperties: false 的作用:
  - 禁止模型输出 schema 中未定义的额外字段
  - 确保输出严格符合预期结构，防止模型"自由发挥"
  - 这是构建可靠 Agent 的关键——下游代码依赖固定字段名
""")
