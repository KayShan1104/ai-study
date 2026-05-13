"""
Phase 3 Step 3: 文档切分（Chunking）

学习目标:
- 理解 chunking 策略对检索质量的影响
- 对比不同 chunk_size 的检索效果
- 理解 chunk_overlap 的作用
- 了解语义切分 vs 字符数切分的差异

运行: python code/phase3/phase3_step3_chunking.py
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

EMBEDDING_MODEL = "text-embedding-v3"


def get_embedding(text: str) -> list[float]:
    """调用 Embedding API 生成向量。"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ============================================================
# 实验文档：阿里云百炼平台技术文档（节选）
# ============================================================

LONG_DOCUMENT = """
# 阿里云百炼平台使用指南

## 1. 什么是百炼平台

百炼（Dashscope）是阿里云提供的大模型服务平台，支持多种模型的 API 调用，包括通义千问系列（qwen-plus、qwen-max、qwen-turbo）以及其他第三方模型。平台提供了完整的 OpenAI 兼容接口，开发者可以通过标准的 OpenAI SDK 调用模型。

## 2. 模型选型

百炼平台提供多种模型规格：

- **qwen-turbo**：速度最快，成本最低，适合简单任务和大量并发
- **qwen-plus**：平衡性能和成本，推荐作为默认选择，适合大多数场景
- **qwen-max**：能力最强，适合复杂推理、代码生成等高质量要求场景
- **qwen-vl**：多模态模型，支持图像理解

选择模型时应根据任务复杂度权衡成本和效果。简单问答用 qwen-turbo 即可，复杂推理才需要 qwen-max。

## 3. API 调用方式

百炼平台支持两种调用方式：

### 3.1 OpenAI 兼容格式

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好"}],
)
```

### 3.2 Dashscope SDK

```python
import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

response = dashscope.Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好"}],
)
```

推荐使用 OpenAI 兼容格式，因为代码更通用，切换其他 OpenAI 兼容的模型时只需改 base_url。

## 4. Embedding API

百炼平台提供 text-embedding-v3 模型用于生成文本向量，维度为 1024。可用于语义搜索、聚类、文本分类等场景。

```python
response = client.embeddings.create(
    model="text-embedding-v3",
    input="这是一段示例文本",
)
embedding = response.data[0].embedding
```

## 5. 常见问题

### 5.1 速率限制

默认 QPS 限制因模型而异：qwen-turbo 为 100，qwen-plus 为 60，qwen-max 为 30。如果需要更高并发，可以在控制台申请提升配额。

### 5.2 Token 计费

计费按输入 + 输出的总 token 数计算。qwen-plus 的计费标准为：输入 0.004 元/千 token，输出 0.012 元/千 token。建议控制 prompt 长度，减少不必要的上下文。

### 5.3 错误处理

常见错误码：400 表示参数错误，401 表示 API Key 无效，429 表示超出速率限制，500 表示服务端错误。建议在代码中处理这些异常情况。
"""


# ============================================================
# Chunking 方法
# ============================================================

def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """按固定字符数切分，带 overlap。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "text": chunk.strip(),
            "start": start,
            "end": min(end, len(text)),
            "char_count": len(chunk),
        })
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(text: str) -> list[dict]:
    """按段落（段落分隔符）切分。"""
    # 按双换行分割，保留段落结构
    paragraphs = text.split("\n\n")
    chunks = []
    current_pos = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 找到段落在原文件中的起始位置
        idx = text.find(para, current_pos)
        if idx == -1:
            idx = current_pos
        chunks.append({
            "text": para,
            "start": idx,
            "end": idx + len(para),
            "char_count": len(para),
        })
        current_pos = idx + len(para)
    return chunks


def chunk_by_headings(text: str) -> list[dict]:
    """按标题（Markdown ##）切分，每个标题下的内容作为一个 chunk。"""
    lines = text.split("\n")
    chunks = []
    current_heading = ""
    current_content = []
    current_start = 0

    for i, line in enumerate(lines):
        if line.startswith("## "):
            # 保存之前的 chunk
            if current_content:
                full_text = current_heading + "\n" + "\n".join(current_content).strip()
                if full_text.strip():
                    chunks.append({
                        "text": full_text.strip(),
                        "start": current_start,
                        "end": i,
                        "char_count": len(full_text),
                        "heading": current_heading,
                    })
            current_heading = line.strip()
            current_content = []
            current_start = i
        else:
            current_content.append(line)

    # 最后一个 chunk
    if current_content:
        full_text = current_heading + "\n" + "\n".join(current_content).strip()
        if full_text.strip():
            chunks.append({
                "text": full_text.strip(),
                "start": current_start,
                "end": len(lines),
                "char_count": len(full_text),
                "heading": current_heading,
            })

    return chunks


def main():
    print("Phase 3 Step 3: 文档切分（Chunking）")
    print(f"文档总字符数: {len(LONG_DOCUMENT)}")
    print()

    # ============================================================
    # 1. 不同 chunk_size 的字符数切分对比
    # ============================================================

    print("  [1] 不同 chunk_size 对比（固定字符数切分）")
    print("  " + "-" * 50)

    chunk_configs = [
        {"chunk_size": 100, "overlap": 0, "desc": "小 chunk，无 overlap"},
        {"chunk_size": 100, "overlap": 20, "desc": "小 chunk，20% overlap"},
        {"chunk_size": 300, "overlap": 30, "desc": "中等 chunk，10% overlap"},
        {"chunk_size": 500, "overlap": 50, "desc": "大 chunk，10% overlap"},
    ]

    for cfg in chunk_configs:
        chunks = chunk_by_characters(LONG_DOCUMENT, cfg["chunk_size"], cfg["overlap"])
        sizes = [c["char_count"] for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        print(f"\n    chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']} ({cfg['desc']})")
        print(f"    切分结果: {len(chunks)} 个 chunk")
        print(f"    大小范围: {min(sizes)}-{max(sizes)} 字符")
        print(f"    平均大小: {avg_size:.0f} 字符")
        print(f"    前 3 个 chunk 预览:")
        for i, c in enumerate(chunks[:3]):
            preview = c["text"][:60].replace("\n", " ")
            print(f"      [{i+1}] \"{preview}...\"")

    # ============================================================
    # 2. 不同切分策略对比
    # ============================================================

    print(f"\n  [2] 不同切分策略对比")
    print("  " + "-" * 50)

    strategies = [
        ("固定字符数 (500/50)", lambda: chunk_by_characters(LONG_DOCUMENT, 500, 50)),
        ("按段落切分", lambda: chunk_by_paragraph(LONG_DOCUMENT)),
        ("按标题切分", lambda: chunk_by_headings(LONG_DOCUMENT)),
    ]

    for name, func in strategies:
        chunks = func()
        sizes = [c["char_count"] for c in chunks]
        print(f"\n    【{name}】")
        print(f"    切分结果: {len(chunks)} 个 chunk")
        print(f"    大小范围: {min(sizes)}-{max(sizes)} 字符")
        print(f"    前 3 个 chunk 标题/内容:")
        for i, c in enumerate(chunks[:3]):
            preview = c["text"][:80].replace("\n", " ")
            heading = c.get("heading", "")
            if heading:
                print(f"      [{i+1}] [{heading}] \"{preview}...\"")
            else:
                print(f"      [{i+1}] \"{preview}...\"")

    # ============================================================
    # 3. 检索质量对比：不同 chunk_size 对搜索的影响
    # ============================================================

    print(f"\n  [3] 检索质量对比 — 不同 chunk_size 对搜索的影响")
    print("  " + "-" * 50)

    test_queries = [
        "百炼平台有哪些模型可以选择？",
        "qwen-plus 的计费标准是什么？",
        "如何处理 API 速率限制错误？",
    ]

    test_configs = [
        {"name": "字符数 100/0", "chunks": chunk_by_characters(LONG_DOCUMENT, 100, 0)},
        {"name": "字符数 300/30", "chunks": chunk_by_characters(LONG_DOCUMENT, 300, 30)},
        {"name": "按段落", "chunks": chunk_by_paragraph(LONG_DOCUMENT)},
        {"name": "按标题", "chunks": chunk_by_headings(LONG_DOCUMENT)},
    ]

    for query in test_queries:
        print(f"\n    查询: 「{query}」")
        query_vec = get_embedding(query)

        for cfg in test_configs:
            # 计算每个 chunk 与 query 的相似度
            scored = []
            for c in cfg["chunks"]:
                c_vec = get_embedding(c["text"])
                sim = cosine_similarity(query_vec, c_vec)
                scored.append((sim, c))
            scored.sort(key=lambda x: x[0], reverse=True)

            top_sim, top_chunk = scored[0]
            preview = top_chunk["text"][:80].replace("\n", " ")
            print(f"      [{cfg['name']}] top1 相似度={top_sim:.4f}: \"{preview}...\"")

    # ============================================================
    # 4. overlap 效果验证
    # ============================================================

    print(f"\n  [4] Overlap 效果验证")
    print("  " + "-" * 50)

    # 构造一个边界案例：关键信息正好被切分点分开
    test_text = "百炼平台提供 qwen-plus 模型用于文本生成。" + "A" * 50 + "qwen-plus 的计费标准为：输入 0.004 元/千 token。"

    chunks_no_overlap = chunk_by_characters(test_text, chunk_size=80, overlap=0)
    chunks_with_overlap = chunk_by_characters(test_text, chunk_size=80, overlap=20)

    query = "qwen-plus 计费多少钱"
    query_vec = get_embedding(query)

    print(f"\n    测试文本: \"{test_text}\"")
    print(f"    查询: 「{query}」")

    print(f"\n    无 overlap (0):")
    for i, c in enumerate(chunks_no_overlap):
        c_vec = get_embedding(c["text"])
        sim = cosine_similarity(query_vec, c_vec)
        preview = c["text"][:60].replace("\n", " ")
        print(f"      chunk[{i+1}] 相似度={sim:.4f}: \"{preview}...\"")

    print(f"\n    有 overlap (20):")
    for i, c in enumerate(chunks_with_overlap):
        c_vec = get_embedding(c["text"])
        sim = cosine_similarity(query_vec, c_vec)
        preview = c["text"][:60].replace("\n", " ")
        print(f"      chunk[{i+1}] 相似度={sim:.4f}: \"{preview}...\"")

    # ============================================================
    # 5. 总结
    # ============================================================

    print(f"\n  [5] 总结")
    print("  " + "-" * 50)
    print("""
  Chunking 的核心权衡:
    - chunk 太大 → embedding 捕捉的信息太杂，检索精度下降
    - chunk 太小 → 丢失上下文，LLM 拿到碎片信息
    - overlap → 防止关键信息被切分点割裂，但会增加存储成本

  切分策略选择:
    - 固定字符数：通用，但可能把一句话切成两半
    - 按段落：保持语义完整性，但段落长短不一
    - 按标题：语义最完整，但 chunk 大小差异大
    - 最佳实践：先用按标题切分，如果某节太长再递归按段落/字符数切

  生产环境常用配置:
    - chunk_size: 300-800 字符（中文 200-500 字）
    - overlap: 10-15% 的 chunk_size
    - 结构化文档（Markdown/HTML）优先按标题/段落切分
""")


if __name__ == "__main__":
    main()
