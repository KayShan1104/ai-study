"""
Phase 2 Step 1c: System Prompt 的工程化管理（补充）

展示 3 种不同复杂度的 system prompt 管理方式：
1. 硬编码字符串 — 最简单的 demo
2. 配置文件/环境变量 — 固定人设的小工具
3. 模板文件 — 需要按场景替换变量的 Agent

运行: python code/phase2/phase2_step1c_prompt_templates.py
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"

# ============================================================
# 方式 1：硬编码字符串（最简单的 demo）
# ============================================================

GAME_SYSTEM_PROMPT = "你是一个资深游戏玩家，你的任务是帮助玩家解决游戏内遇到的问题。"


def chat_v1(user_input):
    """方式 1：system prompt 硬编码，每次调用写死"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": GAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ============================================================
# 方式 2：从文件加载（配置文件分离）
# ============================================================

# 先创建一个 system prompt 模板文件
SYSTEM_PROMPT_DIR = os.path.dirname(__file__)


def load_system_prompt_file(filename):
    """从 .txt 文件加载 system prompt"""
    filepath = os.path.join(SYSTEM_PROMPT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def chat_v2(user_input, system_file="system_game_guide.txt"):
    """方式 2：从文件加载，改人设只需改文件"""
    system_prompt = load_system_prompt_file(system_file)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ============================================================
# 方式 3：模板变量替换（最通用的 Agent 模式）
# ============================================================

class GameAgent:
    """
    游戏攻略 Agent：初始化时加载 system prompt 模板，
    每次 chat 自动组装 messages。
    支持不同游戏、不同角色人设。
    """

    def __init__(self, game_name, character_profile="攻略助手"):
        self.game_name = game_name
        self.character_profile = character_profile
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        template = load_system_prompt_file("system_game_guide_template.txt")
        return template.format(
            game_name=self.game_name,
            role=self.character_profile,
        )

    def chat(self, user_input, context=None):
        """
        聊天接口。context 可选，用于注入游戏状态等额外信息。
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "user", "content": context})
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    # 先创建模板文件
    os.makedirs(SYSTEM_PROMPT_DIR, exist_ok=True)

    # 方式 2 的 prompt 文件
    with open(
        os.path.join(SYSTEM_PROMPT_DIR, "system_game_guide.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("你是一个资深游戏玩家，你的任务是帮助玩家解决游戏内遇到的问题。回答要简洁实用。")

    # 方式 3 的模板文件
    with open(
        os.path.join(SYSTEM_PROMPT_DIR, "system_game_guide_template.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            """你是一个资深{game_name}玩家，角色定位是{role}。
你的职责是帮助玩家解决游戏内遇到的问题。

要求：
1. 回答要简洁、实用，避免长篇大论
2. 如果涉及具体游戏机制，给出操作步骤
3. 如果不确定某个信息，诚实说不知道，不要编造
4. 知道当前游戏版本是最新"""
        )

    question = "新手开局选什么职业比较好？"

    print("=" * 60)
    print("  方式 1：硬编码字符串")
    print("=" * 60)
    print(f"System: {GAME_SYSTEM_PROMPT}")
    print(f"回答: {chat_v1(question)}")

    print("\n" + "=" * 60)
    print("  方式 2：从文件加载")
    print("=" * 60)
    print(f"回答: {chat_v2(question)}")

    print("\n" + "=" * 60)
    print("  方式 3：模板变量 + Agent 类")
    print("=" * 60)

    # 不同的 Agent 实例
    agent1 = GameAgent(game_name="艾尔登法环", character_profile="硬核攻略作者")
    agent2 = GameAgent(game_name="星露谷物语", character_profile="休闲种田向导")

    print(f"\n【{agent1.game_name} — {agent1.character_profile}】")
    print(f"System: {agent1.system_prompt[:100]}...")
    print(f"回答: {agent1.chat(question)}")

    print(f"\n【{agent2.game_name} — {agent2.character_profile}】")
    print(f"System: {agent2.system_prompt[:100]}...")
    print(f"回答: {agent2.chat(question)}")

    print("\n" + "-" * 60)
    print("  三种方式对比")
    print("-" * 60)
    print("""
方式 1 — 硬编码:
  适合: 快速 demo、单次脚本
  缺点: 改人设要改代码

方式 2 — 文件加载:
  适合: 固定人设的小工具
  优点: 改人设只改 .txt 文件，不动代码

方式 3 — 模板 + Agent 类:
  适合: 多游戏、多角色的实际产品
  优点: 初始化一次，后续复用；支持动态变量；
        可扩展 context 注入（游戏状态、背包等）
""")
