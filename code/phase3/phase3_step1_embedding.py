"""
Phase 3 Step 1: 理解 Embedding

学习目标:
- 理解 Embedding 是什么：把文本变成一组数字（向量），捕捉语义
- 生成多段文本的 embedding，计算任意两两之间的余弦相似度
- 观察语义相近的文本向量距离更近

运行: python code/phase3/phase3_step1_embedding.py
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

MODEL = "text-embedding-v3"


def get_embedding(text: str) -> list[float]:
    """调用 Embedding API，返回文本的向量。"""
    response = client.embeddings.create(
        model=MODEL,
        input=text,
    )
    return response.data[0].embedding, response.usage


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度（纯 Python 实现，无第三方依赖）。"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ============================================================
# 测试文本：设计几组语义对比
# ============================================================

TEST_TEXTS = [
    {"id": "T1", "text": "我爱编程", "group": "编程"},
    {"id": "T2", "text": "我喜欢写代码", "group": "编程"},
    {"id": "T3", "text": "Python 是一门好语言", "group": "编程"},
    {"id": "T4", "text": "今天天气很好", "group": "天气"},
    {"id": "T5", "text": "外面阳光灿烂", "group": "天气"},
    {"id": "T6", "text": "明天会下雨", "group": "天气"},
    {"id": "T7", "text": "深度学习在图像识别中很厉害", "group": "AI"},
    {"id": "T8", "text": "神经网络可以拟合复杂函数", "group": "AI"},
    {"id": "T9", "text": "这道菜很好吃", "group": "食物"},
    {"id": "T10", "text": "火锅是我的最爱", "group": "食物"},
]


def main():
    print("Phase 3 Step 1: 理解 Embedding")
    print(f"模型: {MODEL}")
    print()

    # 1. 生成所有文本的 embedding
    print("  [1] 生成 Embedding")
    print("  " + "-" * 50)

    embeddings = {}
    total_tokens = 0

    for item in TEST_TEXTS:
        vec, usage = get_embedding(item["text"])
        embeddings[item["id"]] = {
            "text": item["text"],
            "group": item["group"],
            "vector": vec,
            "dimension": len(vec),
        }
        total_tokens += usage.total_tokens

        print(f"    [{item['id']}] \"{item['text']}\" → 维度={len(vec)}, tokens={usage.total_tokens}")

    print(f"\n  总 token 消耗: {total_tokens}")

    # 2. 打印维度信息
    dim = next(iter(embeddings.values()))["dimension"]
    print(f"\n  Embedding 维度: {dim}")
    print(f"  这意味着每段文本被编码为 {dim} 个浮点数")
    print(f"  这 {dim} 个数字共同捕捉了文本的语义信息")

    # 3. 计算组内相似度（同组文本之间的相似度）
    print(f"\n  [2] 组内相似度（语义相近的文本）")
    print("  " + "-" * 50)

    groups = {}
    for item in TEST_TEXTS:
        groups.setdefault(item["group"], []).append(item["id"])

    for group, ids in groups.items():
        if len(ids) < 2:
            continue
        print(f"\n    【{group}】")
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                sim = cosine_similarity(
                    embeddings[id_a]["vector"],
                    embeddings[id_b]["vector"],
                )
                text_a = embeddings[id_a]["text"]
                text_b = embeddings[id_b]["text"]
                print(f"      {id_a} vs {id_b}: {sim:.4f}")
                print(f"        \"{text_a}\" vs \"{text_b}\"")

    # 4. 计算跨组相似度（不同组文本之间的相似度）
    print(f"\n  [3] 跨组相似度（语义不相关的文本）")
    print("  " + "-" * 50)

    cross_pairs = [
        ("T1", "T4"),  # 编程 vs 天气
        ("T1", "T7"),  # 编程 vs AI
        ("T4", "T9"),  # 天气 vs 食物
        ("T2", "T9"),  # 编程 vs 食物
    ]

    for id_a, id_b in cross_pairs:
        sim = cosine_similarity(
            embeddings[id_a]["vector"],
            embeddings[id_b]["vector"],
        )
        text_a = embeddings[id_a]["text"]
        text_b = embeddings[id_b]["text"]
        group_a = embeddings[id_a]["group"]
        group_b = embeddings[id_b]["group"]
        print(f"    {id_a} vs {id_b}: {sim:.4f}  ({group_a} vs {group_b})")
        print(f"      \"{text_a}\" vs \"{text_b}\"")

    # 5. 总结
    print(f"\n  [4] 总结")
    print("  " + "-" * 50)
    print("""
  观察要点:
    - 同组文本的相似度通常较高（语义相近）
    - 跨组文本的相似度通常较低（语义不相关）
    - 相似度范围在 [-1, 1]，越接近 1 表示语义越相似
    - Embedding 不是关键词匹配，而是理解"意思"
      例如"我爱编程"和"我喜欢写代码"没有共同关键词，但语义相似

  余弦相似度的直观理解:
    - 把 embedding 看作高维空间中的一个方向（向量）
    - 余弦相似度 = 两个向量夹角的余弦值
    - 夹角越小（方向越接近），相似度越高
    - 夹角 0° → 相似度 1.0（完全一致）
    - 夹角 90° → 相似度 0.0（无关）
""")


if __name__ == "__main__":
    main()
