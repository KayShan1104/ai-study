"""
Phase 3 阶段验收：项目文档问答助手

验收要求:
- 使用真实技术文档（10+页）作为知识库
- 完整的 RAG pipeline（索引 + 检索 + 生成）
- CLI 交互界面
- 20 个测试用例的评估报告
- 代码结构清晰：索引、检索、生成三个阶段分离

运行: python code/phase3/phase3_acceptance.py
"""

import os
import sys
import re
import json
import time
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
# 阶段 1：索引（Indexer）
# ============================================================

class Indexer:
    """索引阶段：文档读取 → 按标题切分 → 生成 embedding → 存入 Vector DB。"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "chroma_data_acceptance")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "phase3_acceptance"
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        return self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Phase3 验收知识库"},
        )

    def load_markdown_files(self, directory: str) -> list[dict]:
        """读取目录下的 Markdown 文件。"""
        docs = []
        for filename in os.listdir(directory):
            if filename.endswith(".md") and "Phase" in filename:
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append({"id": filename, "title": filename.replace(".md", ""), "content": content})
        return docs

    def chunk_by_headings(self, text: str, source_id: str) -> list[dict]:
        """按 Markdown 二级/三级标题切分。"""
        # 匹配 ## 或 ### 开头的行
        pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))

        chunks = []
        if not matches:
            # 没有标题，整篇作为一个 chunk
            chunks.append({"source": source_id, "heading": "全文", "text": text[:1000]})
            return chunks

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            chunks.append({"source": source_id, "heading": heading, "text": chunk_text})

        return chunks

    def build_index(self, docs: list[dict]) -> int:
        """批量索引文档。"""
        print("  [索引阶段] 正在处理文档...")
        all_ids, all_texts, all_metadatas = [], [], []

        for doc in docs:
            chunks = self.chunk_by_headings(doc["content"], doc["id"])
            for j, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_{j}"
                all_ids.append(chunk_id)
                all_texts.append(f"{chunk['heading']}\n{chunk['text']}")
                all_metadatas.append({"source": chunk["source"], "heading": chunk["heading"]})

        # 生成 embedding（分批避免 API 限流）
        embeddings = []
        for i in range(0, len(all_texts), 10):
            batch = all_texts[i:i+10]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([d.embedding for d in response.data])
            time.sleep(0.5)

        self.collection.add(
            documents=all_texts,
            embeddings=embeddings,
            ids=all_ids,
            metadatas=all_metadatas,
        )
        print(f"  ✅ 索引构建完成：{len(all_texts)} 个 chunk")
        return len(all_texts)


# ============================================================
# 阶段 2：检索（Retriever）
# ============================================================

class Retriever:
    """检索阶段：用户问题 → embedding → 搜索相似 chunk。"""

    def __init__(self, collection):
        self.collection = collection

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
        results = self.collection.query(
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
                "source": meta.get("source", "unknown"),
                "heading": meta.get("heading", "unknown"),
                "content": doc_text,
                "distance": dist,
            })
        return retrieved


# ============================================================
# 阶段 3：生成（Generator）
# ============================================================

class Generator:
    """生成阶段：检索结果 + prompt → LLM 生成回答。"""

    def generate(self, query: str, retrieved: list[dict]) -> str:
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] 来源: {r['source']} | {r['heading']}\n{r['content']}")
        context = "\n\n".join(context_parts)

        prompt = f"""你是一个技术文档助手。请基于以下参考资料回答用户问题。

参考资料：
{context}

要求：
1. 回答必须基于参考资料，严禁编造信息
2. 如果参考资料中没有相关信息，请明确回答："根据现有文档资料，我没有找到相关信息。"
3. 引用参考资料时使用 [1]、[2] 格式标注来源

用户问题：{query}"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "你是一个基于技术文档回答问题的助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content


# ============================================================
# CLI 交互与评估
# ============================================================

def run_cli(retriever, generator):
    """CLI 交互循环。"""
    print("\n  💬 进入交互模式 (输入 'quit' 退出, 'eval' 运行评估)")
    print("  " + "-" * 50)
    while True:
        query = input("\n  你: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break
        if query.lower() == "eval":
            return
        retrieved = retriever.search(query)
        print(f"  📚 检索到 {len(retrieved)} 篇参考:")
        for r in retrieved:
            print(f"    - [{r['source']}] {r['heading']} (距离={r['distance']:.4f})")
        answer = generator.generate(query, retrieved)
        print(f"\n  🤖 AI:\n  {answer}")


def run_evaluation(retriever, generator) -> list[dict]:
    """20 个测试用例的自动评估。"""
    test_cases = [
        # Phase 1 相关
        {"question": "OpenAI 兼容格式的 base_url 是什么？", "has_answer": True},
        {"question": "qwen-plus 和 qwen-turbo 有什么区别？", "has_answer": True},
        {"question": "API 调用失败返回 401 怎么解决？", "has_answer": True},
        {"question": "Function Calling 的基本流程是什么？", "has_answer": True},
        {"question": "如何限制 LLM 的生成 token 数量？", "has_answer": True},
        {"question": "temperature 参数影响什么？", "has_answer": True},
        {"question": "system prompt 和 user prompt 的区别是什么？", "has_answer": True},
        # Phase 2 相关
        {"question": "Few-shot prompting 是什么？", "has_answer": True},
        {"question": "Chain of Thought (CoT) 怎么用？", "has_answer": True},
        {"question": "JSON 输出格式怎么保证？", "has_answer": True},
        {"question": "Prompt 太长怎么处理？", "has_answer": True},
        {"question": "结构化输出 prompt 怎么设计？", "has_answer": True},
        {"question": "如何减少 LLM 的幻觉？", "has_answer": True},
        {"question": "代码审查 prompt 应该包含什么？", "has_answer": True},
        # 知识范围外的问题
        {"question": "React 的 useState 怎么用？", "has_answer": False},
        {"question": "Docker Compose 的配置文件怎么写？", "has_answer": False},
        {"question": "Kubernetes 的 Deployment 是什么？", "has_answer": False},
        {"question": "MySQL 8.0 的新特性有哪些？", "has_answer": False},
        {"question": "Vue3 和 Vue2 的区别？", "has_answer": False},
        {"question": "Go 语言的 goroutine 是什么？", "has_answer": False},
    ]

    print("\n  📊 运行评估（20 个测试用例）")
    print("  " + "-" * 50)

    results = []
    for i, tc in enumerate(test_cases, 1):
        retrieved = retriever.search(tc["question"], top_k=2)
        answer = generator.generate(tc["question"], retrieved)

        # 简单评估逻辑：检查是否包含"没有找到相关信息" 或 回答长度 > 50
        has_refusal = "没有找到相关信息" in answer
        has_content = len(answer) > 30 and not has_refusal

        # 判断是否符合预期
        if tc["has_answer"]:
            passed = has_content
            status = "✅" if passed else "❌"
        else:
            passed = has_refusal
            status = "✅" if passed else "⚠️"

        print(f"  [{i:2d}] {status} 「{tc['question'][:30]}...」")
        results.append({
            "question": tc["question"],
            "passed": passed,
            "retrieved": [f"{r['source']}#{r['heading']}" for r in retrieved],
            "answer_snippet": answer[:80],
        })

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n  📈 评估结果: {passed_count}/{len(results)} 通过 ({passed_count/len(results)*100:.0f}%)")
    return results


def main():
    print("="*60)
    print("  Phase 3 阶段验收：项目文档问答助手")
    print("="*60)
    print()

    # 1. 初始化并构建索引
    print("  [1] 加载文档并构建索引")
    print("  " + "-" * 50)
    obsidian_dir = os.path.join(os.path.dirname(__file__), "..", "..", "obsidian")
    if not os.path.exists(obsidian_dir):
        print(f"  ❌ 找不到 obsidian 目录: {obsidian_dir}")
        return

    indexer = Indexer()
    docs = indexer.load_markdown_files(obsidian_dir)
    if not docs:
        print("  ❌ 没有找到 Phase 笔记文件")
        return

    chunk_count = indexer.build_index(docs)
    print()

    # 2. 初始化检索和生成
    retriever = Retriever(indexer.collection)
    generator = Generator()

    # 3. 进入交互模式
    run_cli(retriever, generator)

    # 4. 运行评估（如果用户输入了 eval）
    # 注意：为了演示，这里直接运行评估
    print("\n  [2] 自动评估（20 个测试用例）")
    results = run_evaluation(retriever, generator)

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "acceptance_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 评估结果已保存至 {output_path}")
    print()


if __name__ == "__main__":
    main()
