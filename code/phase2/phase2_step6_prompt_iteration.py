"""
Phase 2 Step 6: Prompt 调试与迭代

学习目标:
- 建立系统化的 prompt 调优方法
- 通过 3 轮以上迭代，量化提升代码审查 prompt 的效果
- 记录每轮失败 case 和改进说明

任务：代码审查 prompt 调优
- 准备 20 个不同难度的 Python 代码样本
- 每轮迭代后统计：问题检出率、严重问题漏报率、误报率、结构化输出成功率
- 目标：准确率从 ~60% → 85%+

运行: python code/phase2/phase2_step6_prompt_iteration.py
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
# 20 个测试代码样本
# ============================================================
# 每个样本包含：
#   code: 代码片段
#   expected_issues: 应该被检出的问题类型列表
#   should_reject: 是否应该整体 reject（reject / request_changes / approve）
# ============================================================

CODE_SAMPLES = [
    # --- 安全类问题（8 个） ---
    {
        "id": "SEC-001",
        "code": """def get_user(user_id):
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    sql = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(sql)
    return cursor.fetchone()""",
        "expected_issues": ["sql_injection", "resource_leak"],
        "should_reject": "reject",
        "desc": "SQL 注入 + 数据库连接未关闭",
    },
    {
        "id": "SEC-002",
        "code": """@app.route('/search')
def search():
    query = request.args.get('q')
    return f"<h1>搜索结果: {query}</h1>""",
        "expected_issues": ["xss"],
        "should_reject": "reject",
        "desc": "XSS 漏洞：未转义用户输入",
    },
    {
        "id": "SEC-003",
        "code": """def connect_db():
    import pymysql
    conn = pymysql.connect(
        host="192.168.1.100",
        user="root",
        password="admin123",
        database="production"
    )
    return conn""",
        "expected_issues": ["hardcoded_password", "sensitive_info"],
        "should_reject": "reject",
        "desc": "密码硬编码 + 明文传输",
    },
    {
        "id": "SEC-004",
        "code": """import subprocess

def run_command(cmd):
    result = subprocess.call(cmd, shell=True)
    return result""",
        "expected_issues": ["command_injection"],
        "should_reject": "reject",
        "desc": "命令注入：shell=True + 未过滤输入",
    },
    {
        "id": "SEC-005",
        "code": """import pickle

def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)""",
        "expected_issues": ["insecure_deserialization"],
        "should_reject": "reject",
        "desc": "不安全的反序列化（pickle）",
    },
    {
        "id": "SEC-006",
        "code": """def verify_token(token):
    secret = "my-secret-key-123"
    return token == secret""",
        "expected_issues": ["hardcoded_password", "weak_auth"],
        "should_reject": "reject",
        "desc": "硬编码密钥 + 弱认证逻辑",
    },
    {
        "id": "SEC-007",
        "code": """import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()""",
        "expected_issues": ["weak_crypto"],
        "should_reject": "request_changes",
        "desc": "使用 MD5 哈希密码（不安全）",
    },
    {
        "id": "SEC-008",
        "code": """@app.route('/download')
def download():
    filename = request.args.get('file')
    return send_file(f'/data/{filename}')""",
        "expected_issues": ["path_traversal"],
        "should_reject": "reject",
        "desc": "路径穿越漏洞",
    },

    # --- Bug 类问题（5 个） ---
    {
        "id": "BUG-001",
        "code": """def add_item(item, items=[]):
    items.append(item)
    return items""",
        "expected_issues": ["mutable_default_arg"],
        "should_reject": "request_changes",
        "desc": "可变默认参数陷阱",
    },
    {
        "id": "BUG-002",
        "code": """for i in range(10):
    if i == 5:
        break
print(i)  # 意图：打印最后一个 i""",
        "expected_issues": [],
        "should_reject": "approve",
        "desc": "无 bug，只是循环变量泄漏（Python 特性）",
    },
    {
        "id": "BUG-003",
        "code": """data = {"users": [{"name": "Alice", "age": 30}]}
name = data["users"][0]["name"]
address = data["users"][0]["address"]""",
        "expected_issues": ["key_error"],
        "should_reject": "request_changes",
        "desc": "KeyError：访问不存在的键",
    },
    {
        "id": "BUG-004",
        "code": """def divide(a, b):
    return a / b

result = divide(10, 0)""",
        "expected_issues": ["zero_division"],
        "should_reject": "request_changes",
        "desc": "除以零未处理",
    },
    {
        "id": "BUG-005",
        "code": """class User:
    def __init__(self, name):
        self.name = name
        self.score = 0

    def update_score(self, points):
        self.score = self.score + points

u1 = User("Alice")
u2 = User("Bob")
u1.update_score(10)
print(u2.score)  # 期望输出 0""",
        "expected_issues": [],
        "should_reject": "approve",
        "desc": "无 bug，实例变量独立（正常代码）",
    },

    # --- 性能类问题（4 个） ---
    {
        "id": "PERF-001",
        "code": """result = []
for i in range(10000):
    result.append(i * 2)""",
        "expected_issues": ["inefficient_loop"],
        "should_reject": "request_changes",
        "desc": "应该用列表推导式替代循环 append",
    },
    {
        "id": "PERF-002",
        "code": """def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates""",
        "expected_issues": ["o_n_squared"],
        "should_reject": "request_changes",
        "desc": "O(n²) 查找重复，可用 set 优化",
    },
    {
        "id": "PERF-003",
        "code": """data = []
with open('big_file.csv') as f:
    data = f.readlines()
for line in data:
    process(line)""",
        "expected_issues": ["memory_inefficient"],
        "should_reject": "request_changes",
        "desc": "大文件一次性读取到内存",
    },
    {
        "id": "PERF-004",
        "code": """squares = [x ** 2 for x in range(100)]
total = sum(squares)""",
        "expected_issues": [],
        "should_reject": "approve",
        "desc": "正常代码，列表推导 + sum（Pythonic）",
    },

    # --- 风格/最佳实践（3 个） ---
    {
        "id": "STYLE-001",
        "code": """def f(x):
    if x > 0:
        return True
    else:
        return False""",
        "expected_issues": ["redundant_code"],
        "should_reject": "request_changes",
        "desc": "冗余的 if-else，可直接 return x > 0",
    },
    {
        "id": "STYLE-002",
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

    def process(self):
        pass""",
        "expected_issues": ["god_class"],
        "should_reject": "request_changes",
        "desc": "God class：太多职责，应拆分",
    },
    {
        "id": "STYLE-003",
        "code": """def calculate_total(items):
    total = 0
    for item in items:
        total = total + item
    return total""",
        "expected_issues": [],
        "should_reject": "approve",
        "desc": "简单求和，虽然可以用 sum() 但没有原则性问题",
    },
]

# ============================================================
# 评分规则
# ============================================================

def evaluate_review(review_data, sample):
    """
    评估审查结果的质量。

    返回 dict:
      - expected_count: 该样本预期问题数
      - detected_count: 实际检出的问题数
      - missed: 漏报的问题列表
      - false_positives: 误报的问题列表
      - verdict_correct: 总体 verdict 是否正确
      - parse_ok: JSON 是否解析成功
    """
    if review_data is None:
        return {
            "expected_count": len(sample["expected_issues"]),
            "detected_count": 0,
            "missed": list(sample["expected_issues"]),
            "false_positives": [],
            "verdict_correct": False,
            "parse_ok": False,
        }

    missed = []
    false_positives = []

    # 收集所有报告的问题类型（security_issues + code_quality suggestions 中的关键词）
    reported_types = set()
    all_suggestions_text = ""
    for issue in review_data.get("security_issues", []):
        issue_type = issue.get("type", "").lower()
        reported_types.add(issue_type)
        all_suggestions_text += issue.get("type", "") + " " + issue.get("description", "") + " " + issue.get("fix_suggestion", "") + " "

    for suggestion in review_data.get("code_quality", {}).get("suggestions", []):
        all_suggestions_text += suggestion + " "

    # 1. 检出率：预期问题中有多少个被检出
    detected = 0
    for expected in sample["expected_issues"]:
        found = False
        for rt in reported_types:
            if _issue_type_matches(rt, expected):
                found = True
                break
        if not found:
            found = _suggestion_mentions(all_suggestions_text, expected)
        if found:
            detected += 1
        else:
            missed.append(expected)

    # 2. 误报：正常代码（expected_issues 为空）是否被报了安全问题
    if not sample["expected_issues"]:
        if len(review_data.get("security_issues", [])) > 0:
            false_positives = [
                i["type"] for i in review_data["security_issues"]
            ]

    # 3. Verdict 正确性
    expected_verdict = sample["should_reject"]
    actual_verdict = review_data.get("overall_verdict", "")

    if expected_verdict == "approve":
        verdict_correct = actual_verdict == "approve"
    elif expected_verdict == "reject":
        verdict_correct = actual_verdict == "reject"
    else:  # request_changes
        verdict_correct = actual_verdict != "approve"

    return {
        "expected_count": len(sample["expected_issues"]),
        "detected_count": detected,
        "missed": missed,
        "false_positives": false_positives,
        "verdict_correct": verdict_correct,
        "parse_ok": True,
    }


def _suggestion_mentions(text, expected_issue):
    """检查 suggestions 文本是否提及了某个预期问题。"""
    keywords = {
        "sql_injection": ["sql", "inject", "concat", "parametr", "placeholder"],
        "resource_leak": ["close", "leak", "resource", "context manager", "with statement", "connection"],
        "xss": ["xss", "escape", "encode", "html", "sanitize", "markup"],
        "hardcoded_password": ["hardcod", "password", "credential", "secret", "env", "config"],
        "sensitive_info": ["sensitive", "plaintext", "encrypt", "unencrypted"],
        "command_injection": ["command", "subprocess", "shell", "inject", "shlex"],
        "insecure_deserialization": ["pickle", "deserializ", "unpickle", "yaml.safe"],
        "weak_auth": ["token", "auth", "verify", "compare", "constant"],
        "weak_crypto": ["md5", "sha1", "bcrypt", "argon", "hash", "salt", "cryptograph"],
        "path_traversal": ["traversal", "path", "os.path.basename", "sanitize", "directory"],
        "mutable_default_arg": ["mutable", "default", "none", "list default"],
        "key_error": ["key", ".get(", "exist", "missing", "default"],
        "zero_division": ["zero", "division", "divide"],
        "inefficient_loop": ["comprehension", "list comp", "append", "concise"],
        "o_n_squared": ["set", "quadratic", "o(n", "nested", "lookup", "performance"],
        "memory_inefficient": ["memory", "line by line", "iterator", "stream", "chunk", "readlines"],
        "redundant_code": ["redundant", "simplify", "return x >", "concise", "ternary"],
        "god_class": ["responsibility", "srp", "split", "too many", "god class", "single responsibility"],
    }
    text_lower = text.lower()
    words = keywords.get(expected_issue, [expected_issue])
    return any(w in text_lower for w in words)


def _issue_type_matches(reported, expected):
    """模糊匹配问题类型。"""
    mapping = {
        "sql_injection": ["sql", "injection", "sql_injection"],
        "resource_leak": ["resource_leak", "connection", "close", "leak"],
        "xss": ["xss", "cross-site", "escape", "encode"],
        "hardcoded_password": ["hardcoded", "password", "credential", "secret"],
        "sensitive_info": ["sensitive", "plaintext", "unencrypted"],
        "command_injection": ["command", "subprocess", "shell", "injection"],
        "insecure_deserialization": ["pickle", "deserializ", "unpickle"],
        "weak_auth": ["weak_auth", "token", "authentication", "verify"],
        "weak_crypto": ["md5", "weak_crypto", "hash", "cryptograph"],
        "path_traversal": ["path_traversal", "traversal", "directory"],
        "mutable_default_arg": ["mutable_default", "default_arg", "mutable"],
        "key_error": ["key_error", "missing_key", "key", "exist"],
        "zero_division": ["zero_division", "division", "divide by zero"],
        "inefficient_loop": ["inefficient", "list_comprehension", "append", "loop"],
        "o_n_squared": ["o(n²)", "o_n_squared", "quadratic", "nested_loop", "n^2", "performance"],
        "memory_inefficient": ["memory", "large_file", "readlines", "streaming"],
        "redundant_code": ["redundant", "simplify", "unnecessary", "verbose"],
        "god_class": ["god_class", "single_responsibility", "too_many", "srp", "class"],
    }
    keywords = mapping.get(expected, [expected])
    return any(kw in reported for kw in keywords)


# ============================================================
# Prompt 版本
# ============================================================

# --- V1：基础版（只有角色 + 简单任务描述） ---
PROMPT_V1 = """你是一个资深代码审查员。请审查以下代码，从安全性和代码质量两个维度给出评估。

请用 JSON 格式输出，包含以下字段：
- security_issues: 数组，每个元素包含 type、severity（critical/high/medium/low）、description、fix_suggestion
- code_quality: 对象，包含 maintainability_score（1-10）、suggestions
- overall_verdict: approve / request_changes / reject

代码：
{code}

只输出 JSON，不要其他文字。"""

# --- V2：增加了具体问题类型提示 + severity 定义 ---
PROMPT_V2 = """你是一个资深代码审查员和安全专家。请仔细审查以下代码，识别所有潜在问题。

【审查范围】
- 安全问题：SQL 注入、XSS、命令注入、路径穿越、不安全反序列化、硬编码凭据、弱加密、资源泄漏等
- Bug：变量作用域、默认参数、除零、键不存在、类型错误等
- 性能：O(n²) 算法、内存浪费、不必要的循环等
- 代码风格：冗余代码、违反单一职责、不 Pythonic 的写法等

【输出要求】
- security_issues: 数组，每个元素包含 type（具体问题类型）、severity（critical=可导致数据泄露或被攻击、high=严重 bug、medium=性能或风格问题、low=建议性改进）、description（问题描述）、fix_suggestion（修复建议）
- code_quality: 对象，包含 maintainability_score（1-10 的整数）、suggestions（改进建议数组）
- overall_verdict: "reject"（有 critical/high 安全问题）、"request_changes"（有 medium 问题）、"approve"（无问题）

代码：
{code}

只输出 JSON，不要其他文字。"""

# --- V3：增加 CoT + 分步审查指令 ---
PROMPT_V3 = """你是一个资深代码审查员和安全专家。请按以下步骤审查代码：

1. 先通读代码，识别使用的语言、框架、主要功能
2. 逐行检查安全问题（注入、XSS、凭据泄露、不安全反序列化、弱加密、路径穿越）
3. 检查潜在 bug（类型错误、边界条件、异常处理）
4. 评估性能（算法复杂度、内存使用）
5. 评估代码风格（简洁性、可维护性、Pythonic）

【输出要求 - 严格 JSON】
- security_issues: 数组，每个元素包含 type（具体问题类型，如 sql_injection、xss、hardcoded_password）、severity（critical=可导致数据泄露或被攻击、high=严重 bug 或崩溃、medium=性能问题或不良实践、low=风格/建议）、description（问题描述）、fix_suggestion（可操作的修复建议）
- code_quality: 对象，包含 maintainability_score（1-10 整数）、suggestions（改进建议数组）
- overall_verdict: "reject"（存在 critical/high 问题）、"request_changes"（仅 medium）、"approve"（代码健康）

【重要】
- 如果代码没有问题，security_issues 应为空数组，verdict 为 "approve"
- 不要遗漏任何安全问题，即使是很小的隐患也要指出
-  severity 必须准确：真正的安全问题至少是 high，不要把所有问题都标成 critical

代码：
{code}

只输出 JSON。"""

PROMPTS = {
    "V1 基础版": PROMPT_V1,
    "V2 增强版": PROMPT_V2,
    "V3 CoT 分步版": PROMPT_V3,
}


# ============================================================
# 主流程
# ============================================================

def run_review(prompt_label, code, verbose=False):
    """跑一次代码审查。"""
    prompt = PROMPTS[prompt_label].format(code=code)
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    usage = response.usage

    # 解析 JSON
    try:
        data = json.loads(content)
        parsed = True
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            parsed = True
        except (ValueError, json.JSONDecodeError):
            data = None
            parsed = False

    return content, data, parsed, usage


if __name__ == "__main__":
    print("Phase 2 Step 6: Prompt 调试与迭代")
    print(f"模型: {MODEL}")
    print(f"测试样本数: {len(CODE_SAMPLES)}")
    print(f"Prompt 版本数: {len(PROMPTS)}")

    # 跑所有 prompt 版本
    all_results = {}
    for label in PROMPTS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        total_expected = 0
        total_detected = 0
        verdict_ok = 0
        parse_ok = 0
        total_tokens = 0
        all_missed = []
        all_fp = []

        for sample in CODE_SAMPLES:
            content, data, parsed, usage = run_review(label, sample["code"])
            total_tokens += usage.total_tokens

            eval_result = evaluate_review(data, sample)
            total_expected += eval_result["expected_count"]
            total_detected += eval_result["detected_count"]
            verdict_ok += eval_result["verdict_correct"]
            parse_ok += parsed
            all_missed.extend(eval_result["missed"])
            all_fp.extend(eval_result["false_positives"])

            # 图标：有漏报或误报标红，否则绿
            has_problem = len(eval_result["missed"]) > 0 or len(eval_result["false_positives"]) > 0 or not eval_result["verdict_correct"]
            icon = "✅" if not has_problem else "❌"
            print(f"  {icon} [{sample['id']}] {sample['desc'][:30]:30s}  "
                  f"检出 {eval_result['detected_count']}/{eval_result['expected_count']}  "
                  f"verdict={'✅' if eval_result['verdict_correct'] else '❌'}  "
                  f"parse={'✅' if parsed else '❌'}")

            if eval_result["missed"]:
                print(f"      漏报: {', '.join(eval_result['missed'])}")
            if eval_result["false_positives"]:
                print(f"      误报: {', '.join(eval_result['false_positives'])}")

        recall = total_detected / total_expected * 100 if total_expected > 0 else 100
        verdict_rate = verdict_ok / len(CODE_SAMPLES) * 100
        parse_rate = parse_ok / len(CODE_SAMPLES) * 100
        fp_samples = sum(1 for s in CODE_SAMPLES if s["id"] in [s2["id"] for s2 in CODE_SAMPLES])  # just count

        print(f"\n  汇总:")
        print(f"    问题检出率（召回率）: {total_detected}/{total_expected} ({recall:.0f}%)")
        print(f"    Verdict 正确率: {verdict_ok}/{len(CODE_SAMPLES)} ({verdict_rate:.0f}%)")
        print(f"    JSON 解析成功率: {parse_ok}/{len(CODE_SAMPLES)} ({parse_rate:.0f}%)")
        print(f"    漏报总数: {len(all_missed)}")
        print(f"    误报总数: {len(all_fp)}")
        print(f"    总 Token: {total_tokens}")

        all_results[label] = {
            "recall": recall,
            "detected": total_detected,
            "expected": total_expected,
            "verdict_rate": verdict_rate,
            "verdict_ok": verdict_ok,
            "parse_rate": parse_rate,
            "parse_ok": parse_ok,
            "total_missed": len(all_missed),
            "total_fp": len(all_fp),
            "total_tokens": total_tokens,
        }

    # 最终对比
    print("\n" + "=" * 60)
    print("  三轮迭代对比")
    print("=" * 60)

    print(f"\n| 版本 | 问题检出率 | Verdict 正确率 | JSON 解析率 | 漏报 | 误报 | 总 Token |")
    print(f"|------|-----------|----------------|-------------|------|------|----------|")
    for label, data in all_results.items():
        print(f"| {label} | {data['detected']}/{data['expected']} ({data['recall']:.0f}%) | {data['verdict_ok']}/20 ({data['verdict_rate']:.0f}%) | {data['parse_ok']}/20 ({data['parse_rate']:.0f}%) | {data['total_missed']} | {data['total_fp']} | {data['total_tokens']} |")

    print(f"""
总结与反思:
  - 从 V1 → V2：增加了具体问题类型提示和 severity 定义后，漏报是否减少？
  - 从 V2 → V3：增加 CoT 分步指令后，复杂问题的检出率是否提升？
  - 是否有误报增加的情况？（过度敏感也是问题）
  - Token 成本增长了多少？是否值得？
""")
