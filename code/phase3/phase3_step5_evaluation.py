"""
Phase 3 Step 5: RAG 质量评估

学习目标:
- 为 RAG 系统建立评估指标
- 理解 Context Precision（检索准确率）、Faithfulness（忠实度）、Answer Relevance（回答相关性）
- 用 LLM-as-judge 自动化评分
- 分析哪些 case 做得好、哪些差

运行: python code/phase3/phase3_step5_evaluation.py
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
# 知识库（和 Step 4 相同）
# ============================================================

KNOWLEDGE_BASE = [
    {"id": "doc_01", "title": "百炼平台模型选型", "content": "百炼平台提供多种模型：qwen-turbo（速度最快，成本最低，适合简单任务）、qwen-plus（平衡性能和成本，推荐默认选择）、qwen-max（能力最强，适合复杂推理和代码生成）、qwen-vl（多模态，支持图像理解）。", "category": "model"},
    {"id": "doc_02", "title": "qwen-plus 计费标准", "content": "计费按输入+输出总 token 数。qwen-plus：输入 0.004 元/千 token，输出 0.012 元/千 token。建议控制 prompt 长度，减少不必要的上下文。", "category": "pricing"},
    {"id": "doc_03", "title": "API 速率限制", "content": "默认 QPS 限制：qwen-turbo 为 100，qwen-plus 为 60，qwen-max 为 30。需要更高并发可在控制台申请提升配额。错误码 429 表示超出速率限制。", "category": "api"},
    {"id": "doc_04", "title": "Embedding API 使用", "content": "使用 text-embedding-v3 模型生成 1024 维向量。调用方式：client.embeddings.create(model='text-embedding-v3', input=text)。可用于语义搜索、聚类、文本分类。", "category": "api"},
    {"id": "doc_05", "title": "OpenAI 兼容格式调用", "content": "使用 openai 库，base_url 设置为 https://dashscope.aliyuncs.com/compatible-mode/v1。代码：client = OpenAI(api_key=KEY, base_url=URL)，然后用 client.chat.completions.create() 调用。", "category": "api"},
    {"id": "doc_06", "title": "常见错误码处理", "content": "400 参数错误（检查输入格式）、401 API Key 无效（检查密钥配置）、429 速率限制（降低请求频率或申请配额）、500 服务端错误（稍后重试）。建议在代码中处理这些异常。", "category": "api"},
    {"id": "doc_07", "title": "Prompt Engineering 最佳实践", "content": "系统提示词定义角色和行为规范；用户提示词包含具体任务；复杂任务用 few-shot 示例；JSON 输出用 function calling 保证格式；长上下文放关键信息在开头和结尾（首因+近因效应）。", "category": "prompt"},
    {"id": "doc_08", "title": "Function Calling 使用", "content": "Function Calling 让 LLM 决定调用外部函数。定义函数 schema（name、description、parameters），在 chat.completions.create 中传入 tools 参数。LLM 返回 tool_calls，客户端执行后传回结果。", "category": "api"},
]


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def build_index(documents: list[dict]) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="./chroma_data_step5")
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


def retrieve(collection, query: str, top_k: int = 3) -> list[dict]:
    query_vec = get_embedding(query)
    results = collection.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
    return [{"title": m.get("title",""), "category": m.get("category",""), "content": d, "distance": dist}
            for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]


def generate_answer(query: str, retrieved: list[dict]) -> str:
    context_parts = [f"[{i}] {r['title']}\n{r['content']}" for i, r in enumerate(retrieved, 1)]
    context = "\n\n".join(context_parts)
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


# ============================================================
# LLM-as-Judge 评估
# ============================================================

def judge_context_precision(query: str, retrieved: list[dict]) -> tuple[float, str]:
    """评估检索到的 chunk 是否真的包含答案相关信息。"""
    context_titles = ", ".join([r["title"] for r in retrieved])
    context_contents = "\n".join([f"- [{r['title']}] {r['content']}" for r in retrieved])

    judge_prompt = f"""你是一个 RAG 系统评估专家。请评估检索到的参考文档是否包含用户问题所需的答案信息。

用户问题：{query}
检索到的文档：
{context_contents}

请给出 1-5 分：
1分：所有检索到的文档都与问题无关
2分：只有极少文档相关，大部分无关
3分：约一半文档相关，但信息不完整
4分：大部分文档相关，覆盖了问题要点
5分：所有检索到的文档都与问题高度相关，信息完整

只返回 JSON：{{"score": 1-5的数字, "reason": "简短原因"}}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("score", 0), result.get("reason", "")


def judge_faithfulness(query: str, answer: str, retrieved: list[dict]) -> tuple[float, str]:
    """评估回答是否基于检索到的内容，而不是瞎编。"""
    context_contents = "\n".join([f"- [{r['title']}] {r['content']}" for r in retrieved])

    judge_prompt = f"""你是一个 RAG 系统评估专家。请评估回答是否忠实于参考资料（没有编造信息）。

用户问题：{query}
参考资料：
{context_contents}
实际回答：
{answer}

请给出 1-5 分：
1分：回答完全脱离了参考资料，全是编造的
2分：回答大部分脱离了参考资料
3分：回答部分基于参考资料，但包含了一些编造信息
4分：回答基本基于参考资料，只有少量未经验证的补充
5分：回答完全基于参考资料，没有编造任何信息

只返回 JSON：{{"score": 1-5的数字, "reason": "简短原因"}}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("score", 0), result.get("reason", "")


def judge_answer_relevance(query: str, answer: str) -> tuple[float, str]:
    """评估回答是否真正回答了用户问题。"""
    judge_prompt = f"""你是一个 RAG 系统评估专家。请评估回答是否真正回答了用户问题。

用户问题：{query}
实际回答：
{answer}

请给出 1-5 分：
1分：回答完全没有回答问题
2分：回答只说了皮毛，没有实质内容
3分：回答部分回答了问题，但有重要遗漏
4分：回答基本回答了问题，只有少量不足
5分：回答完整、准确地回答了用户问题

只返回 JSON：{{"score": 1-5的数字, "reason": "简短原因"}}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("score", 0), result.get("reason", "")


# ============================================================
# 测试用例集（15+ 个）
# ============================================================

TEST_CASES = [
    # 知识库有答案的问题
    {"question": "qwen-plus 多少钱？", "expected_key_points": ["输入0.004元/千token", "输出0.012元/千token"]},
    {"question": "百炼平台有哪些模型？", "expected_key_points": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-vl"]},
    {"question": "API 调用频率限制是多少？", "expected_key_points": ["qwen-turbo 100", "qwen-plus 60", "qwen-max 30"]},
    {"question": "触发速率限制怎么办？", "expected_key_points": ["降低请求频率", "申请配额提升"]},
    {"question": "429 错误码什么意思？", "expected_key_points": ["超出速率限制"]},
    {"question": "text-embedding-v3 有什么用？", "expected_key_points": ["语义搜索", "聚类", "文本分类"]},
    {"question": "Embedding 向量多少维？", "expected_key_points": ["1024维"]},
    {"question": "怎么写 Function Calling？", "expected_key_points": ["定义函数schema", "传入tools参数", "处理tool_calls"]},
    {"question": "OpenAI 兼容格式的 base_url 是什么？", "expected_key_points": ["https://dashscope.aliyuncs.com/compatible-mode/v1"]},
    {"question": "401 错误怎么处理？", "expected_key_points": ["API Key无效", "检查密钥配置"]},
    {"question": "qwen-max 适合什么场景？", "expected_key_points": ["复杂推理", "代码生成"]},
    {"question": "Prompt Engineering 有什么最佳实践？", "expected_key_points": ["系统提示词定义角色", "few-shot示例", "关键信息放开头和结尾"]},
    {"question": "qwen-vl 支持什么功能？", "expected_key_points": ["多模态", "图像理解"]},
    # 知识库没有答案的问题
    {"question": "Python 的装饰器怎么写？", "expected_key_points": []},
    {"question": "GPT-4 和 qwen-max 哪个更好？", "expected_key_points": []},
    {"question": "Docker 怎么部署？", "expected_key_points": []},
]


def main():
    print("Phase 3 Step 5: RAG 质量评估")
    print(f"测试用例数: {len(TEST_CASES)}")
    print()

    # 1. 构建索引
    print("  [1] 构建索引")
    print("  " + "-" * 50)
    collection = build_index(KNOWLEDGE_BASE)
    print(f"  已索引 {len(KNOWLEDGE_BASE)} 篇文档\n")

    # 2. 运行评估
    print("  [2] 运行评估（LLM-as-Judge）")
    print("  " + "-" * 50)

    evaluation_results = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n  [{i}/{len(TEST_CASES)}] 「{tc['question']}」")

        retrieved = retrieve(collection, tc["question"], top_k=2)
        answer = generate_answer(tc["question"], retrieved)

        # LLM-as-Judge 三维度评分
        ctx_score, ctx_reason = judge_context_precision(tc["question"], retrieved)
        faith_score, faith_reason = judge_faithfulness(tc["question"], answer, retrieved)
        rel_score, rel_reason = judge_answer_relevance(tc["question"], answer)

        result = {
            "question": tc["question"],
            "retrieved_count": len(retrieved),
            "retrieved_titles": [r["title"] for r in retrieved],
            "answer": answer[:200],
            "context_precision": ctx_score,
            "context_precision_reason": ctx_reason,
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "answer_relevance": rel_score,
            "answer_relevance_reason": rel_reason,
        }
        evaluation_results.append(result)

        print(f"    检索: {[r['title'] for r in retrieved]}")
        print(f"    Context Precision: {ctx_score}/5 — {ctx_reason}")
        print(f"    Faithfulness: {faith_score}/5 — {faith_reason}")
        print(f"    Answer Relevance: {rel_score}/5 — {rel_reason}")

    # 3. 汇总统计
    print(f"\n  [3] 评估汇总")
    print("  " + "-" * 50)

    ctx_scores = [r["context_precision"] for r in evaluation_results]
    faith_scores = [r["faithfulness"] for r in evaluation_results]
    rel_scores = [r["answer_relevance"] for r in evaluation_results]

    print(f"  Context Precision（检索准确率）: {sum(ctx_scores)/len(ctx_scores):.2f}/5.00")
    print(f"  Faithfulness（回答忠实度）: {sum(faith_scores)/len(faith_scores):.2f}/5.00")
    print(f"  Answer Relevance（回答相关性）: {sum(rel_scores)/len(rel_scores):.2f}/5.00")
    print(f"  综合平均分: {(sum(ctx_scores)+sum(faith_scores)+sum(rel_scores))/(3*len(ctx_scores)):.2f}/5.00")

    # 4. 分析好/差 case
    print(f"\n  [4] 表现最好的 case（综合评分 ≥ 4.5）")
    print("  " + "-" * 50)
    for r in evaluation_results:
        avg = (r["context_precision"] + r["faithfulness"] + r["answer_relevance"]) / 3
        if avg >= 4.5:
            print(f"    ✓ 「{r['question']}」 — 综合 {avg:.1f}")
            print(f"      检索: {r['retrieved_titles']}")

    print(f"\n  [5] 表现最差的 case（综合评分 ≤ 3.0）")
    print("  " + "-" * 50)
    for r in evaluation_results:
        avg = (r["context_precision"] + r["faithfulness"] + r["answer_relevance"]) / 3
        if avg <= 3.0:
            print(f"    ✗ 「{r['question']}」 — 综合 {avg:.1f}")
            print(f"      检索: {r['retrieved_titles']}")
            print(f"      Context: {r['context_precision']}/5 — {r['context_precision_reason']}")
            print(f"      Faithful: {r['faithfulness']}/5 — {r['faithfulness_reason']}")
            print(f"      Relevant: {r['answer_relevance']}/5 — {r['answer_relevance_reason']}")

    # 保存结果
    output_path = "code/phase3/evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    print(f"\n  评估结果已保存至 {output_path}")


if __name__ == "__main__":
    main()
