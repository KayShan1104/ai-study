"""
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
