---
name: 第三阶段 - RAG 入门
description: 实现一个最简 RAG 系统，理解 Embedding 和 Vector DB
type: project
---

# 第三阶段：RAG（检索增强生成）入门

> 预计耗时：2-3周 | 开始日期：____ | 完成日期：____
> 前置：第一阶段（API 实操）、第二阶段（Prompt Engineering）
> 关联画像：ai_learning_profile.md

---

## 为什么先学 RAG

RAG 是目前企业中最广泛落地的 AI 模式。它的核心思想很简单：**先把用户问题相关的知识找出来，再让 LLM 基于这些知识回答**。这解决了 LLM 的两个核心问题：
1. 知识截止（模型训练完就不知道新东西了）
2. 幻觉（不知道答案就瞎编）

---

## Step 1：理解 Embedding

**学习目标**：理解 Embedding 是什么、能做什么、怎么用

**操作建议**：
1. 概念：Embedding 是把一段文本变成一组数字（向量），这组数字捕捉了文本的**语义**。语义相近的文本，它们的向量也相近
2. 用 OpenAI 或 Open-source 模型生成一些 embedding：
   ```python
   from openai import OpenAI
   client = OpenAI()

   response = client.embeddings.create(
       model="text-embedding-3-small",
       input="Python 是一种编程语言"
   )
   vector = response.data[0].embedding
   print(f"向量维度: {len(vector)}")  # 通常是 1536 维
   ```
3. 实验：计算几段文本的向量之间的相似度（余弦相似度）
   - "我爱编程" vs "我喜欢写代码" → 应该相似度高
   - "我爱编程" vs "今天天气很好" → 应该相似度低
4. 安装 numpy 和 sklearn 做余弦相似度计算：`pip install numpy scikit-learn`

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec1 = np.array([向量1]).reshape(1, -1)
vec2 = np.array([向量2]).reshape(1, -1)
similarity = cosine_similarity(vec1, vec2)[0][0]
print(f"相似度: {similarity:.4f}")
```

**Milestone 1**：
- [x] 成功生成 5 段不同文本的 embedding
- [x] 计算任意两两之间的相似度，结果符合直觉
- [x] 能口头解释：为什么语义相近的向量在空间中距离近？
- [x] 知道 embedding 的维度代表什么

---

## Step 2：Vector Database 基础

**学习目标**：用向量数据库存储和检索 embedding，体验语义搜索

**操作建议**：
1. 选择一个轻量级 Vector DB 入门（推荐 Chroma，纯 Python，不需要装数据库）：`pip install chromadb`
2. 把一些文档切分成小块（chunk），生成 embedding，存入 Chroma
3. 用查询语句搜索相似文档块

```python
import chromadb
client = chromadb.Client()

# 创建集合（自动 embedding，用内置模型）
collection = client.create_collection("my_docs")

# 添加文档
documents = [
    "Python 的列表推导式可以简洁地创建列表",
    "Java 是静态类型语言，需要编译",
    "Flask 是一个轻量级 Python Web 框架",
    "FastAPI 支持异步请求处理",
    "单元测试可以验证代码的正确性"
]
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 语义搜索
results = collection.query(
    query_texts=["Python 的 Web 框架有哪些"],
    n_results=2
)
print(results["documents"])
```

4. 对比：用关键词搜索（看是否包含"Python"这个词）vs 语义搜索（理解"Web 框架"的意思）

**Milestone 2**：
- [ ] 成功存入 20+ 条文档，查询结果语义相关
- [ ] 对比关键词搜索和语义搜索的差异
- [ ] 理解为什么"Python Web 框架"能搜到"Flask 是轻量级框架"（即使没有完全匹配的词）
- [ ] 了解 Chroma 之外还有哪些 Vector DB（Pinecone、Qdrant、Weaviate、Milvus），它们各自的特点

---

## Step 3：文档切分（Chunking）

**学习目标**：理解 chunking 策略对检索质量的影响

**操作建议**：
1. 为什么需要 chunking：整篇文章的 embedding 会丢失细节，太小又丢失上下文
2. 用 LangChain 的 text splitter 做实验：`pip install langchain-text-splitters`
   ```python
   from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

   # 按固定字符数切分
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50  # 重叠部分保持上下文
   )
   chunks = splitter.split_text(long_text)
   ```
3. 对比不同 chunk_size 的检索效果：100、300、500、1000
4. 了解 overlap 的作用：避免关键信息被切分在两个 chunk 之间

**Milestone 3**：
- [ ] 用一篇文章（比如你自己的项目 README）做 chunking 实验
- [ ] 对比 3 种不同 chunk_size 的检索结果
- [ ] 能解释为什么需要 chunk_overlap
- [ ] 了解语义 chunking（按段落/标题切分 vs 按字符数切分）

---

## Step 4：最简 RAG Pipeline

**学习目标**：串联 embedding + vector DB + LLM，实现完整的 RAG 流程

**操作建议**：
1. 实现完整流程：
   - **索引阶段**：文档 → 切分 → embedding → 存入 Vector DB
   - **查询阶段**：用户问题 → embedding → 检索相似 chunk → 拼接到 prompt → LLM 生成回答
2. 用 Python + Flask 做一个最简单的 Web 界面（或 CLI 也行）
3. prompt 设计：
   ```
   请基于以下参考资料回答用户问题。
   如果参考资料中没有相关信息，请明确说"我没有找到相关信息"。

   参考资料：
   {retrieved_chunks}

   用户问题：{user_question}
   ```
4. 用自己的项目文档或技术文档做知识库来源

**Milestone 4**：
- [ ] 端到端跑通：提问 → 检索 → 回答，回答基于检索内容
- [ ] 问一个知识库里没有的问题，Agent 应该承认不知道而不是瞎编
- [ ] 能打印出检索到了哪些 chunk（让用户看到"我参考了这些信息"）

---

## Step 5：RAG 质量评估

**学习目标**：为你的 RAG 系统建立评估指标——这也是往 Agent 测试方向的重要实践

**操作建议**：
1. 准备 15-20 个测试问题和"标准答案"
2. 设计评估维度：
   - **检索准确率（Context Precision）**：检索到的 chunk 是否真的包含答案相关信息？
   - **回答忠实度（Faithfulness）**：回答是否基于检索到的内容，而不是瞎编？
   - **回答相关性（Answer Relevance）**：回答是否真正回答了用户问题？
3. 人工先跑一遍评分，然后尝试用 LLM-as-judge 自动化评分
4. 记录评分结果，分析哪些 case 做得好、哪些差

```python
# 简单的评估框架
evaluation_results = []
for qa_pair in test_cases:
    question = qa_pair["question"]
    expected = qa_pair["expected_answer"]
    
    # 运行 RAG
    retrieved_chunks = search(question)
    answer = generate_answer(question, retrieved_chunks)
    
    # 用 LLM 做裁判
    judge_prompt = f"""
    用户问题：{question}
    参考答案：{expected}
    实际回答：{answer}
    
    请评分（1-5分）：
    - 回答是否准确？
    - 是否遗漏了关键信息？
    - 是否有幻觉（编造信息）？
    """
    score = call_llm(judge_prompt)
    
    evaluation_results.append({
        "question": question,
        "score": score,
        "retrieved_chunks_count": len(retrieved_chunks)
    })
```

**Milestone 5**：
- [ ] 有 15+ 个测试用例的评分记录
- [ ] 有 3 个维度的量化评分
- [ ] 至少一个 case 的分析报告（为什么好/为什么差）
- [ ] 尝试调整 chunk_size 或检索数量，观察评分变化

---

## Step 6：RAG 进阶技巧

**学习目标**：了解业界常用的 RAG 优化方法

**操作建议**：
1. 了解并尝试以下技术（至少选 2 种实现）：
   - **Hybrid Search**：关键词搜索 + 语义搜索结合
   - **Query Rewriting**：把用户问题改写为更利于检索的形式
   - **Re-ranking**：检索 10 个结果后用 re-ranker 模型精排，取 top 3
   - **Multi-hop Retrieval**：复杂问题分步检索
2. 对比优化前后的评估分数
3. 了解 RAGAS 评估框架（`pip install ragas`）

**Milestone 6**：
- [ ] 实现至少 1 种进阶技巧
- [ ] 优化后评估分数有提升（哪怕只提升 0.1 分）
- [ ] 了解 RAGAS 的基本用法

---

## 阶段验收

### 综合项目：项目文档问答助手

用你自己项目的文档（或其他技术文档）构建一个 RAG 问答系统：

1. 选择一份真实的技术文档（10+ 页）
2. 实现完整的 RAG pipeline（索引 + 检索 + 生成）
3. 用 Flask 提供 Web 接口或 CLI 交互
4. 有 20 个测试用例的评估报告
5. 展示检索到的 chunk（让用户知道回答基于什么信息）

**验收标准**：
- [ ] 端到端跑通，能回答问题
- [ ] 遇到文档中没有的内容时，明确说"不知道"
- [ ] 有评估报告和量化指标
- [ ] 代码结构清晰：索引、检索、生成三个阶段分离
- [ ] 能解释你的 chunking 策略选择和原因

---

## 常见陷阱

- **chunk 太大**：embedding 会丢失细节，检索到无关内容
- **chunk 太小**：丢失上下文，LLM 拿到碎片信息
- **不检查检索质量**：RAG 的核心是"检索"，检索错了后面全错
- **忽略 query rewriting**：用户的问题不一定直接匹配文档的措辞
- **把 RAG 当银弹**：RAG 解决的是"知识更新"问题，不是所有 LLM 问题
