"""
Phase 3 Step 6: RAG 进阶技巧

学习目标:
- 实现 Query Rewriting（查询重写）和 Re-ranking（重排序）
- 对比优化前后的评估分数
- 了解 RAGAS 评估框架

运行: python code/phase3/phase3_step6_advanced.py
"""

import os
import sys
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

EMBEDDING_MODEL = "text-embedding-v3"
CHAT_MODEL = "qwen-plus"

# ============================================================
# 知识库
# ============================================================

KNOWLEDGE_BASE = [
    {"id": "doc_01", "title": "百炼平台模型选型", "content": "百炼平台提供多种模型：qwen-turbo（速度最快，成本最低，适合简单任务）、qwen-plus（平衡性能和成本，推荐默认选择）、qwen-max（能力最强，适合复杂推理和代码生成）、qwen-vl（多模态，支持图像理解）。", "category": "model"},
    {"id": "doc_02", "title": "qwen-plus 计费标准", "content": "计费按输入+输出总 token 数。qwen-plus：输入 0.004 元/千 token，输出 0.012 元/千 token。建议控制 prompt 长度，减少不必要的上下文。", "category": "pricing"},
    {"id": "doc_03", "title": "API 速率限制", "content": "默认 QPS 限制：qwen-turbo 为 100，qwen-plus 为 60，qwen-max 为 30。需要更高并发可在控制台申请提升配额。错误码 429 表示超出速率限制。", "category": "api"},
    {"id": "doc_04", "title": "Embedding API 使用", "content": "使用 text-embedding-v3 模型生成 1024 维向量。调用方式：client.embeddings.create(model='text-embedding-v3', input=text)。可用于语义搜索、聚类、文本分类。", "category": "api"},
    {"id": "doc_05", "title": "OpenAI 兼容格式调用", "content": "使用 openai 库，base_url 设置为 https://dashscope.aliyuncs.com/compatible-mode/v1。代码：client = OpenAI(api_key=KEY, base_url=URL)，然后用 client.chat.completions.create() 调用。", "category": "api"},
    {"id": "doc_06", "title": "常见错误码处理", "content": "400 参数错误（检查输入格式）、401 API Key 无效（检查密钥配置）、429 速率限制（降低请求频率或申请配额）、500 服务端错误（稍后重试）。建议在代码中处理这些异常。", "category": "api"},
    {"id": "doc_07", "title": "Prompt Engineering 最佳实践", "content": "Prompt Engineering 最佳实践包括以下技巧：系统提示词定义角色和行为规范；用户提示词包含具体任务；复杂任务用 few-shot 示例（给出2-3个输入输出对）；JSON 输出用 function calling 保证格式；长上下文放关键信息在开头和结尾（首因+近因效应）。", "category": "prompt"},
    {"id": "doc_08", "title": "Function Calling 使用", "content": "Function Calling 让 LLM 决定调用外部函数。定义函数 schema（name、description、parameters），在 chat.completions.create 中传入 tools 参数。LLM 返回 tool_calls，客户端执行后传回结果。", "category": "api"},
]

TEST_QUERIES = [
    "qwen-plus 多少钱？",
    "百炼平台有哪些模型？",
    "API 调用频率限制是多少？",
    "触发速率限制怎么办？",
    "429 错误码什么意思？",
    "text-embedding-v3 有什么用？",
    "Embedding 向量多少维？",
    "怎么写 Function Calling？",
    "OpenAI 兼容格式的 base_url 是什么？",
    "401 错误怎么处理？",
    "qwen-max 适合什么场景？",
    "Prompt Engineering 有什么最佳实践？",
    "qwen-vl 支持什么功能？",
]


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def build_index(documents: list[dict]) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_data_step6"))
    try:
        chroma_client.delete_collection("rag_docs")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="rag_docs")
    ids = [doc["id"] for doc in documents]
    texts = [doc["content"] for doc in documents]
    metadatas = [{"title": doc["title"], "category": doc["category"]} for doc in documents]
    embeddings = [get_embedding(text) for text in texts]
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return collection


# ============================================================
# 基线：简单检索（Step 4 的方式）
# ============================================================

def baseline_retrieve(collection, query: str, top_k: int = 2) -> list[dict]:
    """基线检索：直接语义搜索。"""
    query_vec = get_embedding(query)
    results = collection.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
    return [{"title": m.get("title",""), "category": m.get("category",""), "content": d, "distance": dist}
            for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]


# ============================================================
# 进阶 1：Query Rewriting（查询重写）
# ============================================================

def rewrite_query(query: str) -> list[str]:
    """让 LLM 把模糊 query 改写为更具体的子问题。"""
    prompt = f"""你是一个搜索优化专家。用户的问题是：
"{query}"
请将其改写为 2-3 个更具体、更利于检索的查询语句。要求：
- 去掉抽象词（如"最佳"、"怎么"、"如何"）
- 使用具体的技术术语
- 覆盖用户问题的核心意图

只返回 JSON 数组，如：["查询1", "查询2"]
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    queries = json.loads(response.choices[0].message.content)
    if isinstance(queries, list):
        return queries
    # 兼容不同返回格式
    for v in queries.values():
        if isinstance(v, list):
            return v
    return [query]


def rewritten_retrieve(collection, query: str, top_k: int = 2) -> list[dict]:
    """Query Rewriting 检索：重写后多路搜索，合并结果。"""
    rewritten = rewrite_query(query)
    all_results = {}

    for q in rewritten:
        query_vec = get_embedding(q)
        results = collection.query(query_embeddings=[query_vec], n_results=top_k + 1, include=["documents", "metadatas", "distances"])
        for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            doc_id = m.get("id", d[:20])
            # 保留最小距离（最相关）的结果
            if doc_id not in all_results or dist < all_results[doc_id]["distance"]:
                all_results[doc_id] = {"title": m.get("title",""), "category": m.get("category",""), "content": d, "distance": dist}

    # 按距离排序取 top_k
    sorted_results = sorted(all_results.values(), key=lambda x: x["distance"])
    return sorted_results[:top_k]


# ============================================================
# 进阶 2：Re-ranking（重排序）
# ============================================================

def rerank_results(query: str, candidates: list[dict]) -> list[dict]:
    """用 LLM 对候选文档做相关度重排序。"""
    prompt = f"""你是一个搜索评估专家。请评估以下文档与用户问题的相关度，从最相关到最不相关排序。

用户问题："{query}"

候选文档：
{json.dumps([{"index": i, "title": c["title"], "content": c["content"]} for i, c in enumerate(candidates)], ensure_ascii=False, indent=2)}

请返回按相关度从高到低排序的索引列表。
只返回 JSON 数组，如：[1, 0, 2]
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    # 解析排序索引 — 兼容 list 和 dict 两种返回格式
    if isinstance(result, list):
        indices = [i for i in result if isinstance(i, int)]
    else:
        indices = []
        for v in result.values():
            if isinstance(v, list):
                indices = [i for i in v if isinstance(i, int)]
                break
    if not indices:
        indices = list(range(len(candidates)))

    # 按排序重新排列
    ranked = []
    for i in indices:
        if i < len(candidates):
            ranked.append(candidates[i])
    return ranked


def reranked_retrieve(collection, query: str, top_k: int = 2, expand_k: int = 5) -> list[dict]:
    """Re-ranking 检索：先放宽召回（expand_k），再精排取 top_k。"""
    query_vec = get_embedding(query)
    results = collection.query(query_embeddings=[query_vec], n_results=expand_k, include=["documents", "metadatas", "distances"])
    candidates = [{"title": m.get("title",""), "category": m.get("category",""), "content": d, "distance": dist}
                  for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
    ranked = rerank_results(query, candidates)
    return ranked[:top_k]


# ============================================================
# 评估：简化版 Context Precision
# ============================================================

def score_retrieval(query: str, retrieved: list[dict]) -> float:
    """简化评分：基于 LLM 判断检索结果是否相关（1-5分）。"""
    context = "\n".join([f"- [{r['title']}] {r['content']}" for r in retrieved])
    prompt = f"""评估以下检索结果是否与用户问题相关。

用户问题：{query}
检索结果：
{context}

给出 1-5 分：1=完全无关，5=高度相关。只返回数字。"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return int(response.choices[0].message.content.strip()[0])
    except (ValueError, IndexError):
        return 0


def generate_answer(query: str, retrieved: list[dict]) -> str:
    context_parts = [f"[{i}] {r['title']}\n{r['content']}" for i, r in enumerate(retrieved, 1)]
    context = "\n\n".join(context_parts)
    prompt = f"""请基于以下参考资料回答用户问题。
参考资料：
{context}
要求：
1. 回答必须基于参考资料，不要编造信息
2. 如果参考资料中没有相关信息，请明确说"没有找到相关信息"
用户问题：{query}"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": "你是一个技术助手。"}, {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def main():
    print("Phase 3 Step 6: RAG 进阶技巧")
    print(f"测试查询数: {len(TEST_QUERIES)}")
    print()

    print("  [1] 构建索引")
    collection = build_index(KNOWLEDGE_BASE)
    print(f"  已索引 {len(KNOWLEDGE_BASE)} 篇文档\n")

    # ============================================================
    # 2. 三种检索策略对比
    # ============================================================

    strategies = [
        ("基线（直接语义搜索）", lambda q: baseline_retrieve(collection, q)),
        ("进阶1（Query Rewriting）", lambda q: rewritten_retrieve(collection, q)),
        ("进阶2（Re-ranking）", lambda q: reranked_retrieve(collection, q)),
    ]

    print("  [2] 三种检索策略对比")
    print("  " + "-" * 50)

    # 先看几个典型 query 的对比
    showcase_queries = [
        "Prompt Engineering 有什么最佳实践？",
        "qwen-plus 多少钱？",
        "怎么写 Function Calling？",
    ]

    for q in showcase_queries:
        print(f"\n  查询: 「{q}」")
        for name, func in strategies:
            results = func(q)
            titles = [r["title"] for r in results]
            print(f"    [{name}] {titles}")

    # ============================================================
    # 3. 全量评分
    # ============================================================

    print(f"\n  [3] 全量评分（13 个测试查询）")
    print("  " + "-" * 50)

    all_scores = {name: [] for name, _ in strategies}

    for q in TEST_QUERIES:
        print(f"\n  「{q}」")
        for name, func in strategies:
            results = func(q)
            score = score_retrieval(q, results)
            all_scores[name].append(score)
            titles = [r["title"] for r in results]
            print(f"    [{name}] {score}/5 — {titles}")

    # ============================================================
    # 4. 汇总
    # ============================================================

    print(f"\n  [4] 汇总对比")
    print("  " + "-" * 50)

    for name in all_scores:
        scores = all_scores[name]
        avg = sum(scores) / len(scores)
        high = sum(1 for s in scores if s >= 4)
        low = sum(1 for s in scores if s <= 2)
        print(f"  {name}")
        print(f"    平均分: {avg:.2f}/5.00")
        print(f"    好结果（≥4分）: {high}/{len(scores)}")
        print(f"    差结果（≤2分）: {low}/{len(scores)}")
        print()

    print("  [5] RAGAS 评估框架简介")
    print("  " + "-" * 50)
    print("""
  RAGAS (RAG Assessment) 是专门的 RAG 评估 Python 库:

  pip install ragas

  核心指标:
  - faithfulness: 回答是否基于检索内容
  - answer_relevance: 回答是否切题
  - context_precision: 检索到的上下文是否相关
  - context_recall: 检索是否召回了所有必要信息

  示例:
  from ragas import evaluate
  from ragas.metrics import faithfulness, answer_relevance, context_precision

  result = evaluate(
      dataset=my_dataset,
      metrics=[faithfulness, answer_relevance, context_precision],
  )

  和我们手写的 LLM-as-Judge 相比:
  - RAGAS 有更完善的评分 rubric 和基准
  - 支持多种 LLM 作为裁判
  - 内置数据集格式和统计报告
  - 但学习成本更高，需要安装额外依赖
""")


if __name__ == "__main__":
    main()
