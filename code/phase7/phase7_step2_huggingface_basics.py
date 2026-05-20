"""
Phase 7 Step 2: HuggingFace Transformers 入门

目标：
1. 用 pipeline 做开箱即用的文本推理
2. 理解 Tokenizer 的工作原理（input_ids / attention_mask）
3. 用 AutoModel 提取句子的向量表示（[CLS] embedding）
4. 对比不同文本的语义相似度
"""

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============================================================
# Part 1: Pipeline — 开箱即用的推理能力
# ============================================================

print("=" * 60)
print("Part 1: Pipeline — 开箱即用")
print("=" * 60)

# --- 1.1 情感分析 ---
# pipeline 封装了 tokenizer + model + 后处理逻辑
# 第一次运行会自动下载模型（~250MB），后续会从缓存加载
print("\n1.1 情感分析 (sentiment-analysis)")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

test_sentences = [
    "I love building AI systems, it's so rewarding!",
    "The shipping was delayed and the package arrived damaged.",
    "Our customs clearance request has been pending for 3 days.",
]

for sent in test_sentences:
    result = classifier(sent)[0]
    print(f"  '{sent[:50]}...'")
    print(f"    → {result['label']} (confidence: {result['score']:.4f})")

# --- 1.2 特征提取 pipeline ---
print("\n1.2 特征提取 (feature-extraction)")
extractor = pipeline(
    "feature-extraction",
    model="distilbert-base-uncased",
    tokenizer="distilbert-base-uncased",
)

# 返回每个 token 的向量，我们取 [0] 位置（对应 [CLS] token）作为整句表示
features = extractor("Hello, world!")
cls_vector = features[0][0]  # [CLS] token 的 768 维向量
print(f"  [CLS] 向量维度: {len(cls_vector)}")
print(f"  前 5 维: {[f'{v:.4f}' for v in cls_vector[:5]]}")

# ============================================================
# Part 2: Tokenizer 深入 — 文本如何变成数字
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Tokenizer 详解")
print("=" * 60)

# --- 2.1 基本 Tokenization ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "Hello, world!",
    "I'm building a logistics classifier.",
    "My container is stuck at customs.",
]

print("\n2.1 Tokenization 结果:")
for text in texts:
    tokens = tokenizer(text)
    print(f"  原文: '{text}'")
    print(f"  input_ids:    {tokens['input_ids']}")
    print(f"  attention_mask: {tokens['attention_mask']}")
    # 把 ID 还原回 token 字符串
    decoded = tokenizer.convert_ids_to_tokens(tokens['input_ids'])
    print(f"  tokens:       {decoded}")
    print()

# --- 2.2 理解 special tokens ---
print("2.2 Special Tokens:")
print(f"  [CLS] ID = {tokenizer.cls_token_id}  (出现在序列开头，用于分类任务)")
print(f"  [SEP] ID = {tokenizer.sep_token_id}  (出现在序列末尾，分隔多段输入)")
print(f"  [PAD] ID = {tokenizer.pad_token_id}  (padding 占位符)")
print(f"  [UNK] ID = {tokenizer.unk_token_id}  (词表中没有的词，变成 [UNK])")

# --- 2.3 多段输入（问答场景）---
print("\n2.3 多段输入（区分 question 和 context）:")
qa_text = "What is the ETA?", "The vessel arrives at Shanghai port on May 25th."
qa_tokens = tokenizer(qa_text[0], qa_text[1])
qa_decoded = tokenizer.convert_ids_to_tokens(qa_tokens['input_ids'])
print(f"  Question: {qa_text[0]}")
print(f"  Context:  {qa_text[1]}")
print(f"  Tokens:   {qa_decoded}")
print(f"  token_type_ids: {qa_tokens['token_type_ids']}")
print(f"    → 0 表示属于 question，1 表示属于 context")

# --- 2.4 批量编码 + Padding ---
print("\n2.4 批量编码 + 自动 Padding:")
batch = [
    "Short text",
    "This is a much longer sentence that needs padding to match the length",
]
batch_tokens = tokenizer(batch, padding=True, return_tensors="pt")
print(f"  input_ids shape: {batch_tokens['input_ids'].shape}")
print(f"  attention_mask:  {batch_tokens['attention_mask']}")
# 短句补了 [PAD]（ID=0），attention_mask 中 0 表示"这是 padding，忽略它"

# ============================================================
# Part 3: AutoModel — 提取语义向量
# ============================================================

print("\n" + "=" * 60)
print("Part 3: AutoModel — 提取语义向量 (Sentence Embedding)")
print("=" * 60)

model = AutoModel.from_pretrained("bert-base-uncased")

# 准备文本
sentences = [
    "Shipping container delayed at port",     # 物流延误
    "My cargo is stuck in customs",            # 清关问题
    "I love programming in Python",            # 编程（无关）
    "The vessel has been delayed for a week",  # 物流延误（换种说法）
]

# Tokenize + 推理
print("\n3.1 提取 [CLS] 向量:")
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state 形状: (batch_size, seq_len, hidden_dim)
# 取每个序列的第一个 token（[CLS]）的向量作为整句表示
cls_embeddings = outputs.last_hidden_state[:, 0, :]
print(f"  向量形状: {cls_embeddings.shape}")  # (4, 768) → 4 句话，每句一个 768 维向量

# --- 3.2 语义相似度矩阵 ---
print("\n3.2 语义相似度矩阵（余弦相似度）:")
similarities = cosine_similarity(cls_embeddings.numpy())

labels = [
    "1: 物流延误",
    "2: 清关问题",
    "3: 编程（无关）",
    "4: 物流延误（换说法）",
]

# 打印上三角矩阵
print(f"  {'':>25} | {'1: 延误':>8} {'2: 清关':>8} {'3: 编程':>8} {'4: 延误2':>8}")
print(f"  {'-'*25}-+-" + "-"*40)
for i, label in enumerate(labels):
    row = f"  {label:>25} |"
    for j in range(4):
        if j >= i:
            row += f" {similarities[i][j]:.4f}"
        else:
            row += f" {'':>8}"
    print(row)

print("\n  观察:")
print(f"    - '物流延误' vs '物流延误(换说法)': {similarities[0][3]:.4f} (高，语义相近)")
print(f"    - '物流延误' vs '清关问题':       {similarities[0][1]:.4f} (中等，都属物流)")
print(f"    - '物流延误' vs '编程':           {similarities[0][2]:.4f} (低，无关)")

# ============================================================
# Part 4: 用 AutoModelForSequenceClassification 做分类推理
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 预训练分类模型推理")
print("=" * 60)

# 加载一个已经训练好的情感分类模型
print("\n4.1 手动加载分类模型:")
clf_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
clf_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

test_texts = [
    "Amazing product, highly recommended!",
    "Terrible experience, never ordering again.",
    "The package arrived on time and in good condition.",
]

for text in test_texts:
    inputs = clf_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = clf_model(**inputs)
    # logits → softmax → 概率
    probs = torch.softmax(outputs.logits, dim=-1)
    label = "POSITIVE" if probs[0][1] > probs[0][0] else "NEGATIVE"
    confidence = max(probs[0]).item()
    print(f"  '{text}'")
    print(f"    → {label} (confidence: {confidence:.4f})")
    print(f"    → NEGATIVE: {probs[0][0].item():.4f}, POSITIVE: {probs[0][1].item():.4f}")

# ============================================================
# Part 5: 关键概念回顾
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 关键概念回顾")
print("=" * 60)
print("""
HuggingFace Transformers 核心组件:

  Tokenizer (文本 → 数字):
    - input_ids:     文本变成词表中的数字 ID 序列
    - attention_mask: 区分真实内容 (1) 和 padding (0)
    - token_type_ids: 多段输入时区分属于哪一段（问答场景）
    - [CLS] token:   序列开头的特殊 token，其向量用于分类任务
    - [SEP] token:   分隔符，标记序列或段的结束
    - [PAD] token:   padding 占位，attention_mask 会让模型忽略它

  Model (数字 → 向量/预测):
    - AutoModel:          基础模型，输出每个 token 的隐藏状态 (hidden states)
    - AutoModelForSequenceClassification: 带分类头的模型，直接输出 logits
    - pipeline:           封装了 tokenizer + model + 后处理的"一键推理"

  为什么 [CLS] 能代表整句?
    BERT 训练时有一个 "Next Sentence Prediction" 任务，[CLS] token 在多层
    self-attention 中聚合了所有其他 token 的信息，所以它的最终隐藏状态可以
    作为整句的语义表示。这也是为什么分类任务通常在 [CLS] 上加一个 Linear 层。
""")
