"""
Phase 3 Step 4: 最简 RAG Pipeline

学习目标:
- 串联 embedding + vector DB + LLM，实现完整的 RAG 流程
- 理解索引阶段和查询阶段的职责分离
- 设计 prompt 让 LLM 基于检索内容回答
- 处理"知识库里没有答案"的情况

运行: python code/phase3/phase3_step4_rag_pipeline.py
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
# 知识库文档（模拟真实技术文档）
# ============================================================

KNOWLEDGE_BASE = [
    {
        "id": "doc_01",
        "title": "百炼平台模型选型",
        "content": "百炼平台提供多种模型：qwen-turbo（速度最快，成本最低，适合简单任务）、qwen-plus（平衡性能和成本，推荐默认选择）、qwen-max（能力最强，适合复杂推理和代码生成）、qwen-vl（多模态，支持图像理解）。",
        "category": "model",
    },
    {
        "id": "doc_02",
        "title": "qwen-plus 计费标准",
        "content": "计费按输入+输出总 token 数。qwen-plus：输入 0.004 元/千 token，输出 0.012 元/千 token。建议控制 prompt 长度，减少不必要的上下文。",
        "category": "pricing",
    },
    {
        "id": "doc_03",
        "title": "API 速率限制",
        "content": "默认 QPS 限制：qwen-turbo 为 100，qwen-plus 为 60，qwen-max 为 30。需要更高并发可在控制台申请提升配额。错误码 429 表示超出速率限制。",
        "category": "api",
    },
    {
        "id": "doc_04",
        "title": "Embedding API 使用",
        "content": "使用 text-embedding-v3 模型生成 1024 维向量。调用方式：client.embeddings.create(model='text-embedding-v3', input=text)。可用于语义搜索、聚类、文本分类。",
        "category": "api",
    },
    {
        "id": "doc_05",
        "title": "OpenAI 兼容格式调用",
        "content": "使用 openai 库，base_url 设置为 https://dashscope.aliyuncs.com/compatible-mode/v1。代码：client = OpenAI(api_key=KEY, base_url=URL)，然后用 client.chat.completions.create() 调用。",
        "category": "api",
    },
    {
        "id": "doc_06",
        "title": "常见错误码处理",
        "content": "400 参数错误（检查输入格式）、401 API Key 无效（检查密钥配置）、429 速率限制（降低请求频率或申请配额）、500 服务端错误（稍后重试）。建议在代码中处理这些异常。",
        "category": "api",
    },
    {
        "id": "doc_07",
        "title": "Prompt Engineering 最佳实践",
        "content": "系统提示词定义角色和行为规范；用户提示词包含具体任务；复杂任务用 few-shot 示例；JSON 输出用 function calling 保证格式；长上下文放关键信息在开头和结尾（首因+近因效应）。",
        "category": "prompt",
    },
    {
        "id": "doc_08",
        "title": "Function Calling 使用",
        "content": "Function Calling 让 LLM 决定调用外部函数。定义函数 schema（name、description、parameters），在 chat.completions.create 中传入 tools 参数。LLM 返回 tool_calls，客户端执行后传回结果。",
        "category": "api",
    },
]


def get_embedding(text: str) -> list[float]:
    """调用 Embedding API。"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def build_index(documents: list[dict]) -> chromadb.Collection:
    """索引阶段：文档 → 切分 → embedding → 存 Vector DB。"""
    print("  [索引阶段] 构建知识库索引...")

    # 删除旧集合
    chroma_client = chromadb.PersistentClient(path="./chroma_data_step4")
    try:
        chroma_client.delete_collection("rag_docs")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="rag_docs",
        metadata={"description": "RAG 知识库"},
    )

    ids = [doc["id"] for doc in documents]
    texts = [doc["content"] for doc in documents]
    metadatas = [
        {"title": doc["title"], "category": doc["category"]}
        for doc in documents
    ]
    embeddings = [get_embedding(text) for text in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"  已索引 {len(documents)} 篇文档")
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[dict]:
    """检索阶段：query → embedding → 搜索 Vector DB。"""
    query_vec = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc_text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "content": doc_text,
            "distance": dist,
        })

    return retrieved


def generate_answer(query: str, retrieved: list[dict]) -> str:
    """生成阶段：用检索到的内容 + prompt 让 LLM 回答。"""
    # 拼接参考内容
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(
            f"[{i}] {r['title']} (距离={r['distance']:.4f})\n{r['content']}"
        )
    context = "\n\n".join(context_parts)

    # RAG prompt 设计
    prompt = f"""请基于以下参考资料回答用户问题。

参考资料：
{context}

要求：
1. 回答必须基于参考资料，不要编造信息
2. 如果参考资料中没有相关信息，请明确说"根据现有资料，我没有找到相关信息"
3. 引用参考资料时用 [1]、[2] 这样的标注

用户问题：{query}"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个技术助手，基于参考资料回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def main():
    print("Phase 3 Step 4: 最简 RAG Pipeline")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")
    print(f"Chat 模型: {CHAT_MODEL}")
    print()

    # ============================================================
    # 1. 构建索引
    # ============================================================

    print("  [1] 构建索引")
    print("  " + "-" * 50)
    collection = build_index(KNOWLEDGE_BASE)
    print()

    # ============================================================
    # 2. 正常问答
    # ============================================================

    print("  [2] 正常问答")
    print("  " + "-" * 50)

    normal_queries = [
        "qwen-plus 多少钱？",
        "API 调用频率限制怎么解决？",
        "百炼平台有哪些模型？",
    ]

    for q in normal_queries:
        print(f"\n  问题: 「{q}」")
        retrieved = retrieve(collection, q, top_k=2)
        print(f"  检索到 {len(retrieved)} 篇参考:")
        for r in retrieved:
            print(f"    - [{r['category']}] {r['title']} (距离={r['distance']:.4f})")

        answer = generate_answer(q, retrieved)
        print(f"\n  回答:\n  {answer}")
        print()

    # ============================================================
    # 3. 知识库中没有的问题
    # ============================================================

    print(f"  [3] 知识库里没有的问题")
    print("  " + "-" * 50)

    out_of_scope_queries = [
        "Python 的装饰器怎么写？",
        "GPT-4 和 qwen-max 哪个更好？",
        "Docker 怎么部署？",
    ]

    for q in out_of_scope_queries:
        print(f"\n  问题: 「{q}」")
        retrieved = retrieve(collection, q, top_k=2)
        print(f"  检索到 {len(retrieved)} 篇参考:")
        for r in retrieved:
            print(f"    - [{r['category']}] {r['title']} (距离={r['distance']:.4f})")

        answer = generate_answer(q, retrieved)
        print(f"\n  回答:\n  {answer}")
        print()

    # ============================================================
    # 4. 总结
    # ============================================================

    print(f"  [4] 总结")
    print("  " + "-" * 50)
    print("""
  RAG 三阶段流程:
    1. 索引阶段（一次性）: 文档 → 切分 → embedding → 存 Vector DB
    2. 检索阶段（每次查询）: 用户问题 → embedding → 搜索相似 chunk
    3. 生成阶段: 检索结果 + prompt → LLM 生成回答

  关键设计:
    - Prompt 中明确要求"基于参考资料回答"，减少幻觉
    - 没有相关信息时要求模型说"不知道"
    - 展示检索到的 chunk，让用户知道回答基于什么信息

  这个 pipeline 的不足（后续 Step 改进）:
    - 没有 chunk_overlap（检索可能漏信息）
    - 固定 top_k=3，不根据 query 调整
    - 没有评估机制，不知道回答质量
    - 只用了语义搜索，没有混合检索
""")


if __name__ == "__main__":
    main()
