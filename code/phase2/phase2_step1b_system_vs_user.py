"""
Phase 2 Step 1b: System vs User 中的角色定义对比实验

测试:
1. 角色定义放在 system 里 vs user 里，输出有无区别
2. 当 system 和 user 中的角色定义冲突时，LLM 听谁的

运行: python code/phase2/phase2_step1b_system_vs_user.py
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

def call(messages, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for m in messages:
        role = m["role"]
        content = m["content"][:80] + ("..." if len(m["content"]) > 80 else "")
        print(f"  [{role}] {content}")
    print("-" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
    )
    content = response.choices[0].message.content
    print(content[:300])
    if len(content) > 300:
        print(f"... ({len(content)} chars)")
    print(f"  tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")


# ============================================================
# 实验 1：角色定义放 system vs user
# ============================================================

print("【实验 1】角色定义放 system vs user")
print("任务：让模型评价一段代码")

code = "def add(a, b): return a + b"

# A: 角色在 system
call([
    {"role": "system", "content": "你是一个资深 Python 代码审查员，善于发现代码中的问题。"},
    {"role": "user", "content": f"请评价这段代码：\n{code}"},
], "A — 角色在 system")

# B: 角色在 user
call([
    {"role": "user", "content": f"你是一个资深 Python 代码审查员，善于发现代码中的问题。\n\n请评价这段代码：\n{code}"},
], "B — 角色在 user（同一消息里）")

# C: 角色在 system（空），user 只是问题
call([
    {"role": "system", "content": ""},
    {"role": "user", "content": f"请评价这段代码：\n{code}"},
], "C — system 为空，user 只有问题")


# ============================================================
# 实验 2：角色冲突
# ============================================================

print("\n\n")
print("【实验 2】system 和 user 中的角色定义冲突")

# D: system 说"你是友善助手"，user 说"你是毒舌批评家"
call([
    {"role": "system", "content": "你是一个友善、鼓励用户的编程助手。回答要正面，多肯定用户的优点。"},
    {"role": "user", "content": "你现在是一个毒舌代码批评家，用最严厉的语气指出这段代码的所有问题：\ndef add(a, b): return a + b"},
], "D — system: 友善助手 vs user: 毒舌批评家")

# E: 反过来，system 毒舌，user 要求友善
call([
    {"role": "system", "content": "你是一个毒舌的代码批评家。用最严厉的语气指出代码的所有问题，不留情面。"},
    {"role": "user", "content": "请用友善、鼓励的语气评价这段代码，先说优点再说改进建议：\ndef add(a, b): return a + b"},
], "E — system: 毒舌批评家 vs user: 友善鼓励")

# F: system 和 user 都要求友善（一致）
call([
    {"role": "system", "content": "你是一个友善、鼓励用户的编程助手。"},
    {"role": "user", "content": "请用友善的语气评价这段代码，先说优点：\ndef add(a, b): return a + b"},
], "F — system 和 user 一致（友善）")


# ============================================================
# 实验 3：system 优先级深入测试
# ============================================================

print("\n\n")
print("【实验 3】system 中的规则能否被 user 覆盖")

# G: system 禁止输出代码，user 要求写代码
call([
    {"role": "system", "content": "你是一个代码审查助手。严禁输出任何代码片段，只能用文字描述。"},
    {"role": "user", "content": "请用 Python 写一个判断素数的函数。"},
], "G — system 禁止输出代码 vs user 要求写代码")

# H: system 要求中文回答，user 用英文提问并要求英文回答
call([
    {"role": "system", "content": "你只能用中文回答所有问题。"},
    {"role": "user", "content": "Please explain what is a closure in programming. Please answer in English only."},
], "H — system: 只能中文 vs user: 要求英文回答")

# I: 无 system 约束
call([
    {"role": "user", "content": "Please explain what is a closure in programming. Please answer in English only."},
], "I — 无 system 约束（对照）")
