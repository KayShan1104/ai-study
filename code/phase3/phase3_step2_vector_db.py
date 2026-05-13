"""
Phase 3 Step 2: Vector Database 基础

学习目标:
- 用 Chroma 存储文档的 embedding，体验语义搜索
- 对比关键词搜索 vs 语义搜索的差异
- 理解为什么"Python Web 框架"能搜到"Flask 是轻量级框架"

运行: python code/phase3/phase3_step2_vector_db.py
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

# ============================================================
# 20+ 条技术文档（模拟一个技术知识库）
# ============================================================

DOCUMENTS = [
    {"id": "doc_01", "content": "Python 的列表推导式可以简洁地创建列表，语法为 [expression for item in iterable]"},
    {"id": "doc_02", "content": "Java 是静态类型语言，代码需要先编译成字节码，然后在 JVM 上运行"},
    {"id": "doc_03", "content": "Flask 是一个轻量级 Python Web 框架，使用装饰器定义路由，适合小型项目"},
    {"id": "doc_04", "content": "FastAPI 是基于类型提示的现代 Python Web 框架，原生支持异步请求和自动 API 文档生成"},
    {"id": "doc_05", "content": "Django 是全栈 Python Web 框架，内置 ORM、admin 后台、用户认证等完整功能"},
    {"id": "doc_06", "content": "Git 是分布式版本控制系统，用于管理代码的历史版本和分支"},
    {"id": "doc_07", "content": "Docker 容器技术可以将应用及其依赖打包成镜像，实现跨环境一致运行"},
    {"id": "doc_08", "content": "RESTful API 设计原则使用 HTTP 方法（GET/POST/PUT/DELETE）操作资源"},
    {"id": "doc_09", "content": "单元测试使用 assert 验证代码的正确性，pytest 是 Python 最流行的测试框架"},
    {"id": "doc_10", "content": "Redis 是基于内存的键值数据库，常用于缓存、会话存储和消息队列"},
    {"id": "doc_11", "content": "PostgreSQL 是开源的关系型数据库，支持复杂查询、事务和 JSON 字段"},
    {"id": "doc_12", "content": "Gunicorn 是 Python WSGI HTTP 服务器，用于在生产环境部署 Flask/FastAPI 应用"},
    {"id": "doc_13", "content": "Nginx 是高性能 HTTP 服务器和反向代理，常用于负载均衡和静态文件服务"},
    {"id": "doc_14", "content": "JWT (JSON Web Token) 是一种无状态的身份验证方案，token 中包含用户信息和签名"},
    {"id": "doc_15", "content": "OAuth 2.0 是授权框架，允许第三方应用在不获取密码的情况下访问用户资源"},
    {"id": "doc_16", "content": "Celery 是 Python 的分布式任务队列，用于执行异步任务和定时任务"},
    {"id": "doc_17", "content": "SQLAlchemy 是 Python 的 ORM 库，将数据库表映射为 Python 对象"},
    {"id": "doc_18", "content": "asyncio 是 Python 的标准异步 I/O 库，用于编写并发代码（协程、事件循环）"},
    {"id": "doc_19", "content": "Pydantic 是基于类型提示的数据验证库，FastAPI 用它做请求参数校验"},
    {"id": "doc_20", "content": "GitHub Actions 是 CI/CD 工具，可以在代码推送时自动运行测试、构建和部署"},
    {"id": "doc_21", "content": "Kubernetes 是容器编排平台，用于管理大规模的容器集群、自动扩缩容"},
    {"id": "doc_22", "content": "GraphQL 是一种 API 查询语言，允许客户端指定需要的数据结构，避免过度获取"},
]


def get_embedding(text: str) -> list[float]:
    """调用 Embedding API 生成向量。"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def main():
    print("Phase 3 Step 2: Vector Database 基础")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")
    print()

    # ============================================================
    # 1. 创建 Chroma 集合并存入文档（自定义 embedding 函数）
    # ============================================================

    print("  [1] 创建 Chroma 集合并存入 22 篇文档")
    print("  " + "-" * 50)

    # PersistentClient 数据持久化到磁盘，Client 是内存中的
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_data_step1"))

    # 删除旧集合避免重复
    try:
        chroma_client.delete_collection("tech_docs")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="tech_docs",
        metadata={"description": "Python 技术知识库"},
    )

    # Chroma 支持自定义 embedding 函数，但我们直接传入文本，它会自动调用内置模型
    # 这里我们用 OpenAI 的 embedding，手动生成后存入
    print("  正在生成 embedding 并存入 Chroma...")

    ids = [doc["id"] for doc in DOCUMENTS]
    texts = [doc["content"] for doc in DOCUMENTS]
    embeddings = [get_embedding(text) for text in texts]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
    )

    print(f"  已存入 {len(DOCUMENTS)} 篇文档")

    # ============================================================
    # 2. 语义搜索查询
    # ============================================================

    QUERIES = [
        {"query": "Python 的 Web 框架有哪些", "desc": "语义搜索 — 没有直接提到 Flask/FastAPI/Django"},
        {"query": "如何部署 Python 应用", "desc": "语义搜索 — 涉及部署相关文档"},
        {"query": "怎么保证 API 安全认证", "desc": "语义搜索 — 认证和授权"},
        {"query": "异步编程怎么做", "desc": "语义搜索 — 异步相关"},
        {"query": "怎么自动测试和发布代码", "desc": "语义搜索 — CI/CD 和测试"},
    ]

    print(f"\n  [2] 语义搜索查询")
    print("  " + "-" * 50)

    for q in QUERIES:
        query_vec = get_embedding(q["query"])
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
        )

        print(f"\n  查询: 「{q['query']}」({q['desc']})")
        for i, (doc_id, doc_text, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
        )):
            # Chroma 返回的是距离（L2 distance），不是相似度
            # 距离越小表示越相似
            print(f"    [{i+1}] {doc_id} (距离={dist:.4f}): {doc_text}")

    # ============================================================
    # 3. 对比：关键词搜索 vs 语义搜索
    # ============================================================

    print(f"\n  [3] 关键词搜索 vs 语义搜索对比")
    print("  " + "-" * 50)

    test_query = "Python 的 Web 框架有哪些"
    query_vec = get_embedding(test_query)

    # 关键词搜索：检查文档中是否包含关键词
    print(f"\n  查询: 「{test_query}」")

    print(f"\n    关键词搜索（包含 'Web' 或 '框架' 的文档）:")
    keywords = ["Web", "框架", "Flask", "Django", "FastAPI"]
    keyword_results = [
        doc for doc in DOCUMENTS
        if any(kw in doc["content"] for kw in keywords)
    ]
    for doc in keyword_results:
        print(f"      [{doc['id']}] {doc['content']}")

    print(f"\n    语义搜索（top 3）:")
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
    )
    for i, (doc_id, doc_text, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    )):
        print(f"      [{i+1}] {doc_id} (距离={dist:.4f}): {doc_text}")

    # ============================================================
    # 4. 总结
    # ============================================================

    print(f"\n  [4] 总结")
    print("  " + "-" * 50)
    print("""
  关键词搜索 vs 语义搜索的核心差异:
    - 关键词搜索：字面匹配，查"Web 框架"只能找到包含这几个字的文档
    - 语义搜索：理解含义，即使文档里没有"Web 框架"四个字，
      只要描述的是 Web 框架（如"Flask 是轻量级 Python Web 框架"），也能找到

  为什么"Python Web 框架"能搜到 Flask/Django/FastAPI？
    - Embedding 模型在训练时见过大量"Flask"和"Web 框架"同时出现的文本
    - 它学到了这两者在语义空间中是相近的
    - 查询向量和文档向量在高维空间中距离近 → 被检索到

  Vector DB 的作用:
    - 存储海量向量 + 高效检索（HNSW 索引，不用遍历所有向量）
    - Chroma 是轻量级的，适合本地开发
    - 生产环境常用: Pinecone（托管）、Qdrant（Rust）、Milvus（大规模）、Weaviate
""")


if __name__ == "__main__":
    main()
