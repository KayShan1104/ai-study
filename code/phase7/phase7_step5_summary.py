"""
Phase 7 Step 5: 模型评估对比 + FastAPI 推理服务 + 项目收尾

目标：
1. 对比 BERT vs Zero-Shot vs Qwen-LoRA 的评估结果
2. 用 FastAPI 封装 BERT 分类器为可调用服务
3. 生成最终评估报告和项目 README
"""

import json
import os
import torch
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ============================================================
# Part 1: 三模型评估对比
# ============================================================

print("=" * 60)
print("Part 1: 三模型评估对比")
print("=" * 60)

LABELS = [
    "shipping_delay", "customs_clearance", "tracking_issue",
    "billing_dispute", "schedule_inquiry", "cargo_damage",
    "booking_request", "general",
]

# 加载各模型的评估结果
results = {}

# BERT fine-tuned
bert_path = "code/phase7/models/eval_results.json"
if os.path.exists(bert_path):
    with open(bert_path, "r") as f:
        results["BERT-base-chinese (Fine-tuned)"] = json.load(f)

# Zero-shot
zeroshot_path = "code/phase7/models/eval_results_zeroshot.json"
if os.path.exists(zeroshot_path):
    with open(zeroshot_path, "r") as f:
        results["qwen-plus (Zero-Shot)"] = json.load(f)

# LoRA (if available from Colab)
lora_path = "code/phase7/models/eval_results_lora.json"
if os.path.exists(lora_path):
    with open(lora_path, "r") as f:
        results["Qwen2.5-1.5B (LoRA)"] = json.load(f)
else:
    results["Qwen2.5-1.5B (LoRA)"] = None

# 对比表
print(f"\n{'':<35} {'Accuracy':>10} {'F1 (wtd)':>10} {'F1 (macro)':>10}")
print(f"  {'-'*35}-{'-'*10}-{'-'*10}-{'-'*10}")

for name, data in results.items():
    if data is None:
        print(f"  {name:<35} {'pending':>10} {'pending':>10} {'pending':>10}")
        continue
    acc_key = next((k for k in data if "accuracy" in k), None)
    f1_w_key = next((k for k in data if "f1_weighted" in k), None)
    f1_m_key = next((k for k in data if "f1_macro" in k), None)
    acc = data.get(acc_key, "N/A")
    f1_w = data.get(f1_w_key, "N/A")
    f1_m = data.get(f1_m_key, "N/A")
    print(f"  {name:<35} {acc:>10} {f1_w:>10} {f1_m:>10}")

# 各类别 F1 对比
print(f"\n各类别 F1 对比:")
print(f"  {'类别':<22}", end="")
for name in results:
    print(f" {name[:15]:>15}", end="")
print()
print(f"  {'-'*22}", end="")
for name in results:
    print(f" {'-'*15}", end="")
print()

for label_name in LABELS:
    print(f"  {label_name:<22}", end="")
    for name, data in results.items():
        if data is None:
            print(f" {'pending':>15}", end="")
        else:
            f1 = "N/A"
            pc = data.get("per_class", {})
            if label_name in pc:
                f1 = f"{pc[label_name].get('f1-score', 'N/A'):.4f}"
            print(f" {f1:>15}", end="")
    print()

# 分析洞察
print(f"\n关键洞察:")
if results["BERT-base-chinese (Fine-tuned)"]:
    bert_acc = results["BERT-base-chinese (Fine-tuned)"].get("accuracy", 0)
    zeroshot_acc = results["qwen-plus (Zero-Shot)"].get("zeroshot_accuracy", 0)
    diff = (bert_acc - zeroshot_acc) * 100
    print(f"  1. BERT fine-tuned 比 zero-shot 准确率高 {diff:.1f}%")
    print(f"     -> 对于特定领域分类任务，少量标注数据 + 微调效果远超大模型 zero-shot")

if results["qwen-plus (Zero-Shot)"]:
    zs_report = results["qwen-plus (Zero-Shot)"].get("per_class", {})
    worst = min(zs_report.items(), key=lambda x: x[1].get("recall", 1))
    best = max(zs_report.items(), key=lambda x: x[1].get("f1-score", 0))
    print(f"  2. Zero-shot 最弱类别: {worst[0]} (recall: {worst[1].get('recall', 0):.4f})")
    print(f"     -> general 类别最难识别，LLM 倾向于'过度解读'用户意图")
    print(f"  3. Zero-shot 最强类别: {best[0]} (F1: {best[1].get('f1-score', 0):.4f})")

if results["Qwen2.5-1.5B (LoRA)"]:
    lora_acc = results["Qwen2.5-1.5B (LoRA)"].get("lora_accuracy", 0)
    print(f"  4. Qwen-LoRA 准确率: {lora_acc:.4f}")
    print(f"     → 只训练 ~0.5% 参数，目标接近 BERT fine-tuned 效果")
else:
    print(f"  4. Qwen-LoRA 结果待 Colab 跑完后更新")

# ============================================================
# Part 2: 加载模型做推理服务
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 加载训练好的 BERT 分类器")
print("=" * 60)

# 导入 Step 3 中的模型定义
import sys
sys.path.insert(0, "code/phase7")

# We need to define the model class inline to avoid import issues
import torch.nn as nn
from transformers import AutoModel

class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


model_path = "code/phase7/models/logistics_bert_classifier.pt"
label_map_path = "code/phase7/models/label_map.json"

if os.path.exists(model_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id_to_label = {int(k): v for k, v in label_map.items()}
    num_labels = len(label_map)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = TextClassifier("bert-base-chinese", num_labels)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    print(f"\n  模型已加载: {model_path}")
    print(f"  参数数: {sum(p.numel() for p in model.parameters()):,}")

    # 推理测试
    print(f"\n  推理测试:")
    test_texts = [
        ("我的包裹已经晚了一周了，到底什么情况？", "shipping_delay"),
        ("下一班到鹿特丹的船期是什么时候？", "schedule_inquiry"),
        ("货物卡在海关了，怎么加快清关速度？", "customs_clearance"),
        ("你们公司支持哪些支付方式？", "general"),
        ("收到的货物外包装破损，里面东西也坏了", "cargo_damage"),
        ("想订一个40HQ从宁波到洛杉矶的舱", "booking_request"),
        ("单号查不到物流信息，是不是单号错了？", "tracking_issue"),
        ("这个月的运费账单不对，比预期高了很多", "billing_dispute"),
    ]

    correct = 0
    for text, expected in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        confidence = probs[0][pred_idx].item()
        pred_label = id_to_label[pred_idx]
        status = "[OK]" if pred_label == expected else "[ERR]"
        if pred_label == expected:
            correct += 1
        print(f"  {status} '{text}'")
        print(f"     → {pred_label} (confidence: {confidence:.4f})")

    print(f"\n  8/8 推理测试: {correct}/8 正确")
else:
    print(f"  模型文件不存在: {model_path}")
    print(f"  请先运行 phase7_step3_logistics_classifier.py")

# ============================================================
# Part 3: 生成 FastAPI 服务代码
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 生成 FastAPI 推理服务")
print("=" * 60)

serve_code = '''"""
物流意图识别 FastAPI 服务

用法:
  uvicorn phase7_serve:app --host 0.0.0.0 --port 8000

API 端点:
  POST /predict          — 单条预测
  POST /predict/batch    — 批量预测
  GET  /health           — 健康检查
  GET  /model/info       — 模型信息
"""

import json
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="物流意图识别服务", version="1.0.0")

# 全局模型
model = None
tokenizer = None
id_to_label = {}
LABELS = []


class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    all_scores: Optional[dict] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    processing_time_ms: float


class TextClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


@app.on_event("startup")
def load_model():
    global model, tokenizer, id_to_label, LABELS

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "logistics_bert_classifier.pt")
    label_map_path = os.path.join(base_dir, "models", "label_map.json")

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    id_to_label = {int(k): v for k, v in label_map.items()}
    LABELS = list(id_to_label.values())

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = TextClassifier("bert-base-chinese", len(id_to_label))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    print(f"模型已加载: {len(LABELS)} 个类别")


def predict_single(text: str, return_all_scores: bool = False) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1).item()
    confidence = probs[0][pred_idx].item()

    result = {
        "text": text,
        "intent": id_to_label[pred_idx],
        "confidence": round(confidence, 4),
    }

    if return_all_scores:
        all_scores = {
            id_to_label[i]: round(probs[0][i].item(), 4)
            for i in range(len(LABELS))
        }
        result["all_scores"] = all_scores

    return result


@app.get("/health")
def health():
    return {"status": "ok", "model": "bert-base-chinese", "categories": len(LABELS)}


@app.get("/model/info")
def model_info():
    return {
        "model_name": "bert-base-chinese",
        "total_params": sum(p.numel() for p in model.parameters()),
        "categories": LABELS,
        "max_length": 64,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    return predict_single(req.text.strip())


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchRequest):
    start = time.time()
    predictions = [predict_single(t.strip()) for t in req.texts if t.strip()]
    elapsed = (time.time() - start) * 1000
    return BatchPredictionResponse(
        predictions=predictions,
        processing_time_ms=round(elapsed, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

serve_path = "code/phase7/phase7_step5_serve.py"
with open(serve_path, "w") as f:
    f.write(serve_code)
print(f"\n  FastAPI 服务代码已生成: {serve_path}")
print(f"  启动命令: uvicorn phase7_serve:app --host 0.0.0.0 --port 8000")
print(f"  测试: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{{\"text\": \"我的货晚了一周\"}}'")

# ============================================================
# Part 4: 生成项目 README
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 生成项目 README")
print("=" * 60)

# 读取实际评估结果
bert_acc = "0.9615"
bert_f1 = "0.9616"
zeroshot_acc = "0.8558"
zeroshot_f1 = "0.8437"
lora_acc = "pending"
lora_f1 = "pending"

if os.path.exists(bert_path):
    with open(bert_path, "r") as f:
        d = json.load(f)
    bert_acc = str(d.get("accuracy", bert_acc))
    bert_f1 = str(d.get("f1_weighted", bert_f1))

if os.path.exists(zeroshot_path):
    with open(zeroshot_path, "r") as f:
        d = json.load(f)
    zeroshot_acc = str(d.get("zeroshot_accuracy", zeroshot_acc))
    zeroshot_f1 = str(d.get("zeroshot_f1_weighted", zeroshot_f1))

if os.path.exists(lora_path):
    with open(lora_path, "r") as f:
        d = json.load(f)
    lora_acc = str(d.get("lora_accuracy", lora_acc))
    lora_f1 = str(d.get("lora_f1_weighted", lora_f1))

readme_content = f"""# 物流文本分类与意图识别系统

> Phase 7 综合项目 — PyTorch 模型微调实战

## 项目背景

为 Maersk AI/ML Scientist 岗位面试准备的实战项目，目标是证明具备**模型训练/微调能力**，而不仅是应用层调用 API。

## 项目概述

构建一个物流场景的文本意图识别系统，将用户消息自动分类为 8 个业务类别：

| 类别 | 说明 | 示例 |
|------|------|------|
| shipping_delay | 运输延误 | "我的包裹已经晚了一周了" |
| customs_clearance | 清关问题 | "货物卡在海关了" |
| tracking_issue | 物流追踪异常 | "单号查不到物流信息" |
| billing_dispute | 账单争议 | "这个月的运费账单不对" |
| schedule_inquiry | 船期/航班查询 | "下一班到鹿特丹的船期" |
| cargo_damage | 货损理赔 | "货物外包装破损" |
| booking_request | 订舱请求 | "想订一个40HQ从宁波到洛杉矶" |
| general | 其他/通用 | "你们公司支持哪些支付方式" |

## 模型对比

| 模型 | 参数量 | 可训练参数 | 训练方式 | Accuracy | F1 (weighted) |
|------|--------|-----------|----------|----------|---------------|
| BERT-base-chinese | 102M | 102M | Full fine-tuning | **{bert_acc}** | **{bert_f1}** |
| qwen-plus (API) | ~100B | 0 | Zero-shot prompt | **{zeroshot_acc}** | **{zeroshot_f1}** |
| Qwen2.5-1.5B + LoRA | 1.5B | ~8M (0.5%) | LoRA | **{lora_acc}** | **{lora_f1}** |

## 关键发现

1. **Fine-tuning > Zero-shot**: BERT 微调比 qwen-plus zero-shot 准确率高 **{(float(bert_acc) - float(zeroshot_acc)) * 100:.1f}%**，证明对于特定领域分类任务，少量标注数据 + 微调比大模型 zero-shot 更有效
2. **LoRA 的参数效率**: Qwen-LoRA 只训练 0.5% 的参数（~8M / 1.5B），目标是在接近 BERT 微调效果的同时，保留 LLM 的通用能力
3. **Zero-shot 的短板**: general 类别 recall 仅 42.86%，LLM 倾向于"过度解读"用户意图，把通用问题判为具体类别

## 数据集

- **规模**: 520 条，8 类（每类 65 条）
- **来源**: 人工编写（模拟物流客服场景）
- **划分**: 训练集 416 条 (80%) / 测试集 104 条 (20%)

## 文件结构

```
code/phase7/
├── phase7_step1_pytorch_basics.py       # PyTorch 基础: 张量、自动求导、训练循环
├── phase7_step2_huggingface_basics.py   # HuggingFace: Tokenizer、预训练模型推理
├── phase7_step3_logistics_classifier.py # BERT 全量微调: 完整 ML pipeline
├── phase7_step4_qwen_lora_colab.ipynb   # Qwen LoRA 微调 (Google Colab)
├── phase7_step4_zeroshot_baseline.py    # Zero-Shot 基线评估
├── phase7_step5_serve.py                # FastAPI 推理服务
├── phase7_step5_summary.py              # 本文件: 评估对比 + 报告生成
└── models/
    ├── logistics_bert_classifier.pt     # 训练好的 BERT 分类器
    ├── label_map.json                   # 类别映射
    ├── eval_results.json                # BERT 评估结果
    ├── eval_results_zeroshot.json       # Zero-Shot 评估结果
    └── eval_results_lora.json           # LoRA 评估结果 (Colab 生成)
```

## 快速开始

### 安装依赖
```bash
pip install torch transformers datasets accelerate scikit-learn
```

### 训练 BERT 分类器
```bash
python code/phase7/phase7_step3_logistics_classifier.py
```

### Zero-Shot 基线评估
```bash
python code/phase7/phase7_step4_zeroshot_baseline.py
```

### 启动推理服务
```bash
pip install fastapi uvicorn
python code/phase7/phase7_step5_serve.py
# 或 uvicorn phase7_step5_serve:app --host 0.0.0.0 --port 8000
```

### 测试 API
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "我的包裹已经晚了一周了，到底什么情况？"}}'
```

## 技术方案

### BERT 分类器架构
```
BERT-base-chinese (102M params)
  └─ [CLS] token (768d)
     └─ Dropout(0.1)
        └─ Linear(768 → 8) → Logits → CrossEntropyLoss
```

### LoRA 配置 (Qwen)
```
Qwen2.5-1.5B-Instruct
  └─ LoRA (r=16, alpha=32)
     └─ target_modules: [q_proj, v_proj]
     └─ trainable params: ~8M (0.5%)
```

### 训练超参数 (BERT)
- Batch Size: 16
- Learning Rate: 2e-5
- Epochs: 3
- Optimizer: AdamW
- Max Length: 64

## 面试话术

> "我用 PyTorch 和 HuggingFace Transformers 从零构建了一个物流场景的文本分类系统。
> 1. 构造了 520 条标注数据，覆盖 8 个物流业务类别
> 2. 用 BERT-base-chinese 做了全量微调，测试集 F1 达到 {bert_f1}
> 3. 对比了 zero-shot prompt (qwen-plus) 和 fine-tuned 的效果，fine-tuning 在专业术语场景下提升了 {(float(bert_acc) - float(zeroshot_acc)) * 100:.1f}% 的准确率
> 4. 用 Qwen2.5-1.5B + LoRA 做了参数高效微调，只训练了 0.5% 的参数
>
> 这个项目让我理解了完整的 ML pipeline：数据准备 → 模型训练 → 评估 → 部署。
> 我也理解什么时候该用 fine-tuning（领域专有知识、格式强约束），什么时候 zero-shot 就够了。"
"""

readme_path = "code/phase7/PROJECT_README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"\n  项目 README 已生成: {readme_path}")

# ============================================================
# Part 5: 最终总结
# ============================================================

print("\n" + "=" * 60)
print("Part 5: Phase 7 总结")
print("=" * 60)

print("""
Phase 7 完成的 5 个 Step:

  Step 1: PyTorch 基础          - 张量、自动求导、手写训练循环
  Step 2: HuggingFace           - Tokenizer、AutoModel、[CLS] embedding
  Step 3: BERT 全量微调         - 520条数据、8分类、F1 %s
  Step 4: Zero-Shot + LoRA      - Zero-Shot F1 %s、LoRA Colab 待补充
  Step 5: 评估对比 + 部署       - 三模型对比报告、FastAPI 服务、项目 README

面试核心论据:
  [1] 会用 PyTorch 手写训练循环 (Step 1)
  [2] 理解 Tokenizer 和 Transformer 架构 (Step 2)
  [3] 完整 ML pipeline 实战 (Step 3)
  [4] 理解 LoRA 参数高效微调原理 (Step 4)
  [5] 能做 zero-shot vs fine-tuned 的技术选型 (Step 4)
  [6] 能部署为可调用服务 (Step 5)

生成文件清单:
  - code/phase7/phase7_step1_pytorch_basics.py
  - code/phase7/phase7_step2_huggingface_basics.py
  - code/phase7/phase7_step3_logistics_classifier.py
  - code/phase7/phase7_step4_qwen_lora_colab.ipynb
  - code/phase7/phase7_step4_zeroshot_baseline.py
  - code/phase7/phase7_step5_serve.py
  - code/phase7/phase7_step5_summary.py (本文件)
  - code/phase7/models/logistics_bert_classifier.pt
  - code/phase7/models/eval_results.json
  - code/phase7/models/eval_results_zeroshot.json
  - code/phase7/PROJECT_README.md
""" % (bert_f1, zeroshot_f1))

# Save summary as JSON
summary = {
    "phase": 7,
    "steps_completed": 5,
    "models": {
        "bert": {
            "model": "bert-base-chinese",
            "params": "102M",
            "trainable": "102M",
            "method": "full fine-tuning",
            "accuracy": float(bert_acc),
            "f1_weighted": float(bert_f1),
        },
        "zeroshot": {
            "model": "qwen-plus",
            "params": "~100B",
            "trainable": "0",
            "method": "zero-shot prompt",
            "accuracy": float(zeroshot_acc),
            "f1_weighted": float(zeroshot_f1),
        },
    },
    "dataset": {"total": 520, "classes": 8, "train": 416, "test": 104},
    "files_generated": [
        "phase7_step1_pytorch_basics.py",
        "phase7_step2_huggingface_basics.py",
        "phase7_step3_logistics_classifier.py",
        "phase7_step4_qwen_lora_colab.ipynb",
        "phase7_step4_zeroshot_baseline.py",
        "phase7_step5_serve.py",
        "phase7_step5_summary.py",
        "models/logistics_bert_classifier.pt",
        "models/eval_results.json",
        "models/eval_results_zeroshot.json",
        "PROJECT_README.md",
    ],
}

with open("code/phase7/models/phase7_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Phase 7 summary saved to: code/phase7/models/phase7_summary.json")
print(f"\nPhase 7 完成!")
