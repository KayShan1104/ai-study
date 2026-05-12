"""
Phase 2 阶段验收：结构化代码审查助手

要求：
  1. 接受 Python 代码作为输入（stdin 或文件路径）
  2. 使用 few-shot + CoT 提示策略
  3. 输出严格的结构化 JSON（bug 列表、性能问题、安全建议、总体评分）
  4. 对常见代码模式（循环、文件 IO、网络请求）有特定的审查规则
  5. 跑 20 个测试用例，结构化输出 100% 可解析

用法:
  python code_reviewer.py < code.py          # 从 stdin 读取
  python code_reviewer.py path/to/code.py    # 从文件读取
  python code_reviewer.py --test             # 跑 20 个测试用例
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
# JSON Schema — 严格定义输出结构
# ============================================================

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "security_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "问题类型，如 SQL Injection、XSS、Command Injection、Hardcoded Credentials、Insecure Deserialization"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "line_hint": {"type": "string", "description": "问题所在的代码片段"},
                    "description": {"type": "string"},
                    "fix_suggestion": {"type": "string"},
                },
                "required": ["type", "severity", "description", "fix_suggestion"],
                "additionalProperties": False,
            },
        },
        "bug_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Bug 类型，如 KeyError、TypeError、ZeroDivisionError、Mutable Default Argument"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "line_hint": {"type": "string", "description": "问题所在的代码片段"},
                    "description": {"type": "string"},
                    "fix_suggestion": {"type": "string"},
                },
                "required": ["type", "severity", "description", "fix_suggestion"],
                "additionalProperties": False,
            },
        },
        "performance_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "性能问题类型，如 O(n^2) Complexity、Memory Inefficiency、Unnecessary Loop"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "line_hint": {"type": "string", "description": "问题所在的代码片段"},
                    "description": {"type": "string"},
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
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "overall_verdict": {"type": "string", "enum": ["approve", "request_changes", "reject"]},
    },
    "required": ["security_issues", "bug_issues", "performance_issues", "code_quality", "overall_score", "overall_verdict"],
    "additionalProperties": False,
}

# ============================================================
# System Prompt + Few-Shot CoT 示例
# ============================================================

SYSTEM_PROMPT = """你是一个资深代码审查员和安全专家。请审查 Python 代码，从安全性、Bug、性能和代码质量四个维度给出评估。

【审查规则】
- 安全：SQL 注入、XSS、命令注入、路径穿越、不安全反序列化（pickle）、硬编码密码/密钥、弱加密（MD5/SHA1）
- Bug：可变默认参数（def f(x=[])）、未捕获异常（KeyError/ZeroDivisionError）、资源泄漏（文件/数据库连接未关闭）
- 性能：O(n^2) 嵌套循环、大文件一次性读取（readlines）、可以用列表推导替代的循环 append
- 代码质量：单一职责原则（类职责过多）、冗余代码（如 if x: return True else: return False）

【Verdict 规则】
- critical/high 安全问题或严重 Bug → reject
- medium 问题（性能/风格）→ request_changes
- 无明显问题 → approve"""

FEW_SHOT_EXAMPLES = """--- 示例 1：安全问题 ---
输入代码：
def get_user(user_id):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    sql = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(sql)
    return cursor.fetchone()

思考：
1. 第5行：字符串拼接构造 SQL → SQL 注入，critical
2. 第3-6行：数据库连接未用 context manager，异常时不会关闭 → 资源泄漏，medium

输出：
{
  "security_issues": [
    {"type": "SQL Injection", "severity": "critical", "line_hint": "\"SELECT * FROM users WHERE id = \" + user_id", "description": "使用字符串拼接构造 SQL 查询，存在 SQL 注入风险", "fix_suggestion": "使用参数化查询：cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"}
  ],
  "bug_issues": [],
  "performance_issues": [
    {"type": "Resource Leak", "severity": "medium", "line_hint": "conn = sqlite3.connect('users.db')", "description": "数据库连接未使用 with 语句管理，异常时不会自动关闭", "fix_suggestion": "使用 context manager: with sqlite3.connect('users.db') as conn:"}
  ],
  "code_quality": {"maintainability_score": 3, "suggestions": ["添加类型提示和 docstring", "将 SQL 查询逻辑封装为独立函数"]},
  "overall_score": 30,
  "overall_verdict": "reject"
}

--- 示例 2：正常代码 ---
输入代码：
squares = [x ** 2 for x in range(100)]
total = sum(squares)

思考：
1. 使用列表推导式，Pythonic 风格，无安全问题，无 bug，性能合理。

输出：
{
  "security_issues": [],
  "bug_issues": [],
  "performance_issues": [],
  "code_quality": {"maintainability_score": 8, "suggestions": ["如果只需要 sum，可以用生成器表达式省内存: sum(x ** 2 for x in range(100))"]},
  "overall_score": 90,
  "overall_verdict": "approve"
}"""

# ============================================================
# 审查函数
# ============================================================

def review_code(code: str) -> tuple:
    """
    审查代码，返回 (原始回复, 解析后的 JSON, 是否解析成功, usage)。
    """
    user_prompt = f"""请按以下步骤审查代码：
1. 通读代码，识别主要功能和使用的库
2. 逐行检查安全问题
3. 检查潜在 bug
4. 评估性能和代码质量

{FEW_SHOT_EXAMPLES}

--- 请审查以下代码 ---
{code}

只输出 JSON，不要其他文字。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "code_review",
                "schema": REVIEW_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content.strip()
    usage = response.usage

    try:
        data = json.loads(content)
        return content, data, True, usage
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            return content, data, True, usage
        except (ValueError, json.JSONDecodeError):
            return content, None, False, usage


# ============================================================
# 格式化输出
# ============================================================

def print_review(data: dict):
    """格式化打印审查结果。"""
    verdict_colors = {"approve": "\033[92m", "request_changes": "\033[93m", "reject": "\033[91m"}
    reset = "\033[0m"
    verdict = data.get("overall_verdict", "N/A")
    score = data.get("overall_score", "N/A")

    print(f"\n  总体评分: {score}/100")
    print(f"  审查结论: {verdict_colors.get(verdict, '')}{verdict.upper()}{reset}")

    for category, key in [("安全问题", "security_issues"), ("Bug", "bug_issues"), ("性能", "performance_issues")]:
        issues = data.get(key, [])
        if issues:
            print(f"\n  [{category}]")
            for issue in issues:
                sev = issue.get("severity", "?")
                print(f"    [{sev.upper()}] {issue['type']}: {issue['description']}")
                print(f"      建议: {issue['fix_suggestion']}")

    quality = data.get("code_quality", {})
    if quality.get("suggestions"):
        print(f"\n  [代码质量] 可维护性评分: {quality.get('maintainability_score', 'N/A')}/10")
        for s in quality["suggestions"]:
            print(f"    - {s}")


# ============================================================
# 测试用例
# ============================================================

TEST_CASES = [
    {
        "id": "TC-001", "desc": "SQL 注入",
        "code": """def get_user(user_id):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    sql = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(sql)
    return cursor.fetchone()""",
    },
    {
        "id": "TC-002", "desc": "XSS 漏洞",
        "code": """@app.route('/search')
def search():
    query = request.args.get('q')
    return f"<h1>搜索结果: {query}</h1>""",
    },
    {
        "id": "TC-003", "desc": "密码硬编码",
        "code": """def connect_db():
    import pymysql
    conn = pymysql.connect(
        host="192.168.1.100",
        user="root",
        password="admin123",
        database="production"
    )
    return conn""",
    },
    {
        "id": "TC-004", "desc": "命令注入",
        "code": """import subprocess
def run_command(cmd):
    result = subprocess.call(cmd, shell=True)
    return result""",
    },
    {
        "id": "TC-005", "desc": "不安全反序列化",
        "code": """import pickle
def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)""",
    },
    {
        "id": "TC-006", "desc": "路径穿越",
        "code": """@app.route('/download')
def download():
    filename = request.args.get('file')
    return send_file(f'/data/{filename}')""",
    },
    {
        "id": "TC-007", "desc": "MD5 弱加密",
        "code": """import hashlib
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()""",
    },
    {
        "id": "TC-008", "desc": "硬编码 API 密钥",
        "code": """import requests
def call_api():
    headers = {"Authorization": "Bearer sk-abc123secret456"}
    return requests.get("https://api.example.com/data", headers=headers)""",
    },
    {
        "id": "TC-009", "desc": "可变默认参数",
        "code": """def add_item(item, items=[]):
    items.append(item)
    return items""",
    },
    {
        "id": "TC-010", "desc": "KeyError 未处理",
        "code": """data = {"users": [{"name": "Alice"}]}
name = data["users"][0]["name"]
address = data["users"][0]["address"]""",
    },
    {
        "id": "TC-011", "desc": "除零未处理",
        "code": """def divide(a, b):
    return a / b

result = divide(10, 0)""",
    },
    {
        "id": "TC-012", "desc": "文件未关闭",
        "code": """def read_config():
    f = open('config.json', 'r')
    data = f.read()
    return data""",
    },
    {
        "id": "TC-013", "desc": "异常处理过于宽泛",
        "code": """def process():
    try:
        do_something()
    except Exception:
        pass""",
    },
    {
        "id": "TC-014", "desc": "循环 append 低效",
        "code": """result = []
for i in range(10000):
    result.append(i * 2)""",
    },
    {
        "id": "TC-015", "desc": "O(n^2) 查找",
        "code": """def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates""",
    },
    {
        "id": "TC-016", "desc": "大文件全量读取",
        "code": """data = []
with open('big_file.csv') as f:
    data = f.readlines()
for line in data:
    process(line)""",
    },
    {
        "id": "TC-017", "desc": "冗余 if-else",
        "code": """def is_positive(x):
    if x > 0:
        return True
    else:
        return False""",
    },
    {
        "id": "TC-018", "desc": "God class",
        "code": """class DataProcessor:
    def __init__(self):
        self.data = []
        self.config = {}
        self.cache = {}
        self.logger = None
        self.db = None
        self.api = None
        self.queue = []
        self.lock = None
        self.count = 0
        self.name = ""
        self.version = "1.0"

    def process(self): pass
    def save(self): pass
    def load(self): pass
    def validate(self): pass
    def transform(self): pass""",
    },
    {
        "id": "TC-019", "desc": "正常代码（列表推导 + sum）",
        "code": """squares = [x ** 2 for x in range(100)]
total = sum(squares)""",
    },
    {
        "id": "TC-020", "desc": "正常代码（实例变量独立）",
        "code": """class User:
    def __init__(self, name):
        self.name = name
        self.score = 0

    def update_score(self, points):
        self.score = self.score + points

u1 = User("Alice")
u2 = User("Bob")
u1.update_score(10)
print(u2.score)""",
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    if "--test" in sys.argv:
        run_tests()
        return

    # 从 stdin 或文件读取代码
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        filepath = sys.argv[1]
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        print(f"审查文件: {filepath}")
    else:
        print("请粘贴 Python 代码（Ctrl+D / Ctrl+Z 结束）:")
        code = sys.stdin.read()

    if not code.strip():
        print("错误：未输入代码")
        sys.exit(1)

    print(f"模型: {MODEL}")
    print(f"代码长度: {len(code)} 字符")
    print("\n正在审查...")

    content, data, parsed, usage = review_code(code)
    print(f"\n  [Token] prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")

    if parsed:
        print_review(data)
    else:
        print(f"\n  JSON 解析失败，原始回复:")
        print(f"  {content}")


def run_tests():
    """跑 20 个测试用例。"""
    print("Phase 2 阶段验收：结构化代码审查助手")
    print(f"模型: {MODEL}")
    print(f"测试用例数: {len(TEST_CASES)}")
    print()

    parsed_count = 0
    total_tokens = 0

    for i, tc in enumerate(TEST_CASES):
        content, data, parsed, usage = review_code(tc["code"])
        total_tokens += usage.total_tokens

        icon = "✅" if parsed else "❌"
        print(f"  {icon} [{tc['id']}] {tc['desc']:<20s}  JSON: {'成功' if parsed else '失败'}  "
              f"tokens: {usage.total_tokens}")

        if parsed:
            parsed_count += 1
            verdict = data.get("overall_verdict", "N/A")
            score = data.get("overall_score", "N/A")
            sec = len(data.get("security_issues", []))
            bugs = len(data.get("bug_issues", []))
            perf = len(data.get("performance_issues", []))
            print(f"        verdict={verdict}, score={score}, 安全={sec}, Bug={bugs}, 性能={perf}")
        else:
            print(f"        原始回复: {content[:100]}...")

    print(f"\n{'='*60}")
    print(f"  测试结果")
    print(f"{'='*60}")
    print(f"  JSON 解析成功率: {parsed_count}/{len(TEST_CASES)} ({parsed_count/len(TEST_CASES)*100:.0f}%)")
    print(f"  总 Token 消耗: {total_tokens}")
    print(f"  目标: 100% 可解析")
    print(f"  {'✅ 通过' if parsed_count == len(TEST_CASES) else '❌ 未通过'}")


if __name__ == "__main__":
    main()
