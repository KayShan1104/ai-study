# 物流文本分类与意图识别系统

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
| BERT-base-chinese | 102M | 102M | Full fine-tuning | **0.9615** | **0.9616** |
| qwen-plus (API) | ~100B | 0 | Zero-shot prompt | **0.8558** | **0.8437** |
| Qwen2.5-1.5B + LoRA | 1.5B | ~8M (0.5%) | LoRA | **pending** | **pending** |

## 关键发现

1. **Fine-tuning > Zero-shot**: BERT 微调比 qwen-plus zero-shot 准确率高 **10.6%**，证明对于特定领域分类任务，少量标注数据 + 微调比大模型 zero-shot 更有效
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
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "我的包裹已经晚了一周了，到底什么情况？"}'
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
> 2. 用 BERT-base-chinese 做了全量微调，测试集 F1 达到 0.9616
> 3. 对比了 zero-shot prompt (qwen-plus) 和 fine-tuned 的效果，fine-tuning 在专业术语场景下提升了 10.6% 的准确率
> 4. 用 Qwen2.5-1.5B + LoRA 做了参数高效微调，只训练了 0.5% 的参数
>
> 这个项目让我理解了完整的 ML pipeline：数据准备 → 模型训练 → 评估 → 部署。
> 我也理解什么时候该用 fine-tuning（领域专有知识、格式强约束），什么时候 zero-shot 就够了。"
