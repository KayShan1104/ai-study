# 第七阶段：PyTorch 模型微调实战 — 物流文本分类与意图识别

> 预计耗时：2-3 周 | 开始日期：____ | 完成日期：____
> 前置：Phase 1（API 实操）、Phase 2（Prompt Engineering）、Phase 3（RAG）
> 关联画像：ai_learning_profile.md
> 项目目标：补强 PyTorch + 模型训练能力，产出物流场景 fine-tuning 项目，面向 Maersk AI/ML Scientist 岗位

---

## 为什么做这个项目

你的技术栈在 LLM 应用层（RAG、Agent、Function Calling）已经比较完整，但**模型训练/微调**是空白。"AI/ML Scientist"这类岗位的核心要求之一是：不仅会调用 API，还要能训练和调整模型本身。

选择**物流文本分类 + 意图识别**作为项目主题，原因有三：

1. **面试叙事**：可以直接对 Maersk 说"我做了一个物流场景的 fine-tuning 项目"，弥补行业背景短板
2. **技术覆盖**：同时覆盖 PyTorch 基础、HuggingFace 生态、数据准备、训练流程、模型评估
3. **可延伸性**：这个项目的 pipeline 可以复用到其他 NLP 场景，不局限于物流

---

## Step 1：PyTorch 基础 — 从张量到训练循环

**学习目标**：掌握 PyTorch 的核心概念，能手写一个完整的训练循环

**背景**：你 14 年 Python 经验，numpy 应该不陌生。PyTorch 的核心就是一个"支持自动求导的 numpy"。

**操作建议**：

1. 安装 PyTorch：`pip install torch`（CPU 版本即可，后面用不到 GPU）
2. 理解 Tensor：跟 numpy array 对比，多一个 `.requires_grad` 和 `.backward()`

```python
import torch

# Tensor vs numpy
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward(torch.ones_like(x))  # 自动求导
print(x.grad)  # dy/dx = 2x + 3 = [5, 7, 9]
```

3. 手写一个最简单的线性回归训练循环（二分类）：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模拟数据：2 个特征，二分类
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100,)).float()

# 模型
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 训练循环
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

4. 理解关键概念：
   - `nn.Module`：所有模型的基类，类似你写 Flask 的 `app`
   - `forward()`：模型的前向传播逻辑
   - `loss.backward()`：反向传播，计算梯度
   - `optimizer.step()`：用梯度更新参数
   - `zero_grad()`：每次训练前清空梯度（不然梯度会累积）

**Milestone 1**：
- [x] 能解释 `requires_grad`、`backward()`、`zero_grad()`、`step()` 各自的作用
- [x] 手写线性回归训练循环能跑通，loss 持续下降
- [x] 知道 BCELoss vs BCEWithLogitsLoss 的区别（后者更稳定，内部做了 sigmoid）

---

## Step 2：HuggingFace Transformers 入门 — 用预训练模型

**学习目标**：学会用 HuggingFace 加载预训练模型做推理，理解 Tokenizer 和 Model 的关系

**操作建议**：

1. 安装：`pip install transformers datasets accelerate`
2. 用预训练模型做文本分类推理：

```python
from transformers import pipeline

# 用预训练模型做情感分析（体验一下开箱即用）
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love building AI systems")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

3. 理解 Tokenizer（关键概念）：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello, world!", return_tensors="pt")
print(tokens)
# {'input_ids': [[101, 7592, 1010, 2088, 102]], 'token_type_ids': ..., 'attention_mask': ...}
```

   - `input_ids`：文本变成数字 ID 序列
   - `attention_mask`：哪些 token 是真实内容（1=真实，0=padding）
   - `token_type_ids`：多段输入时区分属于哪一段（如问答场景）

4. 用 AutoModel 加载模型做特征提取：

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Shipping container delayed"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # (1, seq_len, hidden_dim)
# 取 [CLS] token 的向量作为整句表示
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(cls_embedding.shape)  # (1, 768)
```

5. 了解 HuggingFace 模型库：去 https://huggingface.co/models 搜索 "bert chinese"、"qwen"，看模型的 description 和 files

**Milestone 2**：
- [x] 能用 pipeline 做情感分析推理
- [x] 理解 tokenizer 的输出结构（input_ids / attention_mask）
- [x] 能用 AutoModel 提取句子的向量表示
- [x] 知道 [CLS] token 的作用（用于分类任务的聚合表示）

---

## Step 3：物流文本分类 — 从零训练一个分类器

**学习目标**：用 BERT/RoBERTa 作为 backbone，在自定义数据集上训练一个文本分类模型

**这一步是整个 Phase 的核心**。你会构建一个完整的 ML pipeline：数据准备 → 模型构建 → 训练 → 评估 → 保存。

**操作建议**：

1. 准备数据集 — 构造一个物流场景的文本分类数据集（建议 500-1000 条）：

```python
# 分类体系（参考物流常见场景）
LABELS = {
    "shipping_delay": "运输延误相关",      # "我的包裹已经晚了一周了"
    "customs_clearance": "清关问题",       # "货物卡在海关，怎么加快清关"
    "tracking_issue": "物流追踪异常",     # "单号查不到物流信息"
    "billing_dispute": "账单争议",        # "这个月的运费账单不对"
    "schedule_inquiry": "船期/航班查询",  # "下一班到鹿特丹的船期是什么时候"
    "cargo_damage": "货损理赔",           # "收到的货物外包装破损"
    "booking_request": "订舱请求",        # "想订一个 40HQ 从宁波到洛杉矶的舱"
    "general": "其他/通用",               # "你们公司支持哪些支付方式"
}
```

   数据来源建议：
   - 用 LLM 批量生成（Prompt Engineering 的实战应用）
   - 参考公开数据集（如 LCQMC 中文客服数据集）
   - 自己写每类 50-100 条，LLM 扩写

2. 构建 Dataset：

```python
from datasets import Dataset
from transformers import AutoTokenizer

# 假设你有数据：texts, labels
dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

3. 构建分类模型：

```python
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
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits
```

4. 训练循环（完整版）：

```python
model = TextClassifier("bert-base-chinese", num_labels=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(3):
    model.train()
    for batch in train_loader:  # DataLoader
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["label"]).sum().item()
    accuracy = correct / len(val_dataset)
    print(f"Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}")
```

5. 保存模型：

```python
torch.save(model.state_dict(), "models/logistics_classifier.pt")
```

**Milestone 3**：
- [x] 数据集 500+ 条，覆盖 6-8 个类别，train/test split
- [x] 模型训练能跑通，验证集准确率 > 80%
- [x] 能打印混淆矩阵，分析哪些类别容易混淆
- [x] 保存训练好的模型，能对新文本做推理

---

## Step 4：LLM Fine-tuning — 用 Qwen 做意图识别

**学习目标**：微调一个 LLM（Qwen 系列），理解 LoRA/QLoRA 等参数高效微调方法

**背景**：Step 3 是"传统" NLP 分类（BERT + 分类头）。Step 4 升级到 LLM fine-tuning，这是面试时更有说服力的项目。

**操作建议**：

1. 准备意图识别数据集（跟 Step 3 类似，但格式改为对话格式）：

```python
# 数据集格式：instruction + input → output
INTENT_DATA = [
    {
        "instruction": "识别以下用户消息的意图类别",
        "input": "我的货柜在洋山港等了三天还没装上船",
        "output": "shipping_delay"
    },
    {
        "instruction": "识别以下用户消息的意图类别",
        "input": "能帮我查一下 MAERSK SEATLAND 的 ETA 吗",
        "output": "schedule_inquiry"
    },
    # ... 200+ 条
]
```

2. 安装 fine-tuning 工具：`pip install peft bitsandbytes trl`
   - `peft`：Parameter-Efficient Fine-Tuning（LoRA）
   - `bitsandbytes`：量化（4bit 加载大模型，降低显存需求）
   - `trl`：HuggingFace 的训练器封装（SFTTrainer）

3. 用 LoRA 微调（完整流程）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 加载模型（4bit 量化，CPU 也能跑但慢，推荐有 GPU 或用云端）
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # LoRA rank，越大表达能力越强但参数越多
    lora_alpha=32,           # 缩放系数
    target_modules=["q_proj", "v_proj"],  # 只微调 attention 层
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 看有多少参数被训练（通常 < 1%）

# 训练
training_args = TrainingArguments(
    output_dir="./qwen-intent-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # 混合精度训练
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
)
trainer.train()
trainer.save_model("./qwen-intent-lora")
```

4. 推理测试：

```python
from peft import PeftModel

# 加载 base model + LoRA 权重
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "./qwen-intent-lora")

prompt = "识别以下用户消息的意图类别\n用户消息：我的货柜在港口被扣了\n意图："
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

5. 如果没有 GPU，替代方案：
   - 用 Google Colab 免费 GPU（T4）
   - 用阿里云 PAI 等平台（你可能已经有账号）
   - 先用 Step 3 的 BERT 分类器作为面试展示，Step 4 作为"进行中"的加分项

**Milestone 4**：
- [x] 理解 LoRA 的原理（低秩分解，只训练一小部分参数）
- [x] 能解释为什么 fine-tuning 不需要更新全部参数（节省显存、防止灾难性遗忘）
- [x] LoRA 微调能跑通，对新输入能正确识别意图（Colab notebook 已就绪）
- [x] 对比 base model 和 fine-tuned model 的输出差异

---

## Step 5：模型评估、部署与项目收尾

**学习目标**：建立完整的评估体系，把模型打包成可演示的项目

**操作建议**：

1. 建立评估指标（跟 Phase 6 的评估体系呼应）：

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# 评估分类模型
y_true = [...]  # 真实标签
y_pred = [...]  # 预测标签

print(classification_report(y_true, y_pred, target_names=LABELS.keys()))
print(confusion_matrix(y_true, y_pred))

# 关键指标：Accuracy, Precision, Recall, F1 (macro & weighted)
```

2. 对比不同模型的评估结果（建一个表格）：

| 模型 | 参数量 | 训练方式 | Accuracy | F1 (macro) | 推理延迟 |
|------|--------|----------|----------|------------|----------|
| BERT-base-chinese | 102M | Full fine-tuning | ? | ? | ? |
| Qwen2.5-1.5B | 1.5B | LoRA (r=16) | ? | ? | ? |
| Qwen-plus (API) | — | Zero-shot prompt | ? | ? | ? |

   - 最后一行用 Phase 1 的 API 调用能力做 zero-shot 对比，体现你对"微调 vs 零样本"的理解

3. 用 FastAPI 包装成服务（呼应你已有的 FastAPI 经验）：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    label, confidence = model.predict(req.text)
    return {"intent": label, "confidence": confidence}
```

4. 准备项目 README（作为 Portfolio 项目）：
   - 项目背景：为什么做物流文本分类
   - 技术方案：BERT vs LLM fine-tuning 的 tradeoff
   - 评估结果：各模型的指标对比
   - 可复现性：`pip install -r requirements.txt` + `python train.py`

**Milestone 5**：
- [x] 评估报告：至少 3 个模型的指标对比
- [x] FastAPI 服务能接收请求并返回分类结果
- [x] 项目 README 完整，包含技术方案、评估结果、使用方法
- [x] 能口头解释：为什么选 LoRA 而不是全量微调？什么场景下 zero-shot 就够了？

---

## 阶段验收

### 综合项目：物流文本分类与意图识别系统

**产出物**：

1. **训练好的模型**：
   - BERT-base-chinese 分类器（Step 3）
   - Qwen2.5-1.5B LoRA fine-tuned 权重（Step 4，有 GPU 的话）

2. **评估报告**（`eval_report.md`）：
   - 数据集统计（类别分布、train/test split）
   - 各模型的 Accuracy、F1、混淆矩阵
   - zero-shot vs fine-tuned 对比分析

3. **可运行的代码**：
   - `data/generate_dataset.py` — 数据集生成脚本
   - `train_bert.py` — BERT 分类器训练
   - `train_lora.py` — Qwen LoRA 微调
   - `evaluate.py` — 评估脚本
   - `serve.py` — FastAPI 推理服务

4. **项目 README**（可作为 GitHub 项目展示）

**验收标准**：
- [x] 数据集 500+ 条，覆盖 ≥ 6 个物流场景类别
- [x] BERT 分类器在测试集上 F1 > 0.85
- [x] 能清晰解释训练循环的每一步（数据流 → 前向 → loss → 反向 → 优化）
- [x] 能解释 LoRA 的原理和优势（参数高效、避免灾难性遗忘）
- [x] 有完整的评估报告和模型对比
- [x] 能用面试语言描述项目："我用 PyTorch + HuggingFace 训练了一个物流文本分类模型，对比了 BERT 全量微调和 Qwen LoRA 微调的效果..."

---

## 常见陷阱

- **只调参不理解**：训练循环的每一步都要能解释"为什么"，不要只是 copy-paste
- **数据集太小**：BERT 至少需要几百条数据才能学到东西，太少会过拟合
- **不划分 train/test**：必须留出测试集，否则无法判断模型是"真的学会了"还是"背下来了"
- **忽略 baseline**：一定要有一个 baseline（比如 zero-shot LLM 或简单规则），否则无法证明微调的价值
- **没有 GPU 就放弃**：BERT-base-chinese（102M 参数）CPU 也能训，只是慢一些。没有 GPU 就只做 Step 3，Step 4 作为后续计划
- **学习率太大**：fine-tuning 的学习率通常比从头训练小 10-100 倍（1e-5 ~ 5e-5），太大模型会"崩溃"

---

## 面试话术准备

完成这个项目后，你在面试中可以这样说：

> "在模型训练方面，我用 PyTorch 和 HuggingFace Transformers 从零构建了一个物流场景的文本分类系统。具体来说：
> 1. 构造了 800 条标注数据，覆盖 8 个物流业务类别
> 2. 用 BERT-base-chinese 做了全量微调，测试集 F1 达到 0.89
> 3. 用 Qwen2.5-1.5B + LoRA 做了参数高效微调，只训练了 0.5% 的参数，F1 达到 0.91
> 4. 对比了 zero-shot prompt 和 fine-tuned 的效果，fine-tuning 在专业术语场景下提升了 25% 的准确率
>
> 这个项目让我理解了完整的 ML pipeline：数据准备 → 模型训练 → 评估 → 部署。我也理解什么时候该用 fine-tuning（领域专有知识、格式强约束），什么时候 zero-shot 就够了。"
