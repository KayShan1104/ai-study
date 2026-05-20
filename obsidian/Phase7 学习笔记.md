# Phase 7 学习笔记

> 开始日期：2026-05-20
> 目标：掌握 PyTorch + HuggingFace 模型微调能力，产出物流场景 fine-tuning 项目

---

## Phase 7 计划概览

详见 [[phase7_pytorch_finetuning|第七阶段：PyTorch 模型微调实战]]

**项目目标**：
- 补强 PyTorch + 模型训练能力（面向 Maersk AI/ML Scientist 岗位）
- 产出物流文本分类 + 意图识别项目
- 对比 BERT 全量微调 vs Qwen LoRA 微调 vs zero-shot prompt 的效果

**5 个 Step**：
1. PyTorch 基础 — 张量、自动求导、训练循环
2. HuggingFace Transformers — Tokenizer、预训练模型推理
3. 物流文本分类 — BERT + 自定义数据集 + 完整训练 pipeline
4. LLM Fine-tuning — Qwen + LoRA 参数高效微调
5. 模型评估、部署与项目收尾

---

## Step 1: PyTorch 基础 — 张量与训练循环

### 运行结果解读

代码文件：`code/phase7/phase7_step1_pytorch_basics.py`

**自动求导（Autograd）**：
- `y = x^2 + 3x`，理论导数 `dy/dx = 2x + 3` → `[5, 7, 9]`，PyTorch 算出来完全一致
- **梯度累积**演示了不 `zero_grad()` 的后果：第二次 `backward()` 后梯度从 `[2,4,6]` 变成 `[7,11,15]`，两次梯度叠加了
- `zero_grad()` 清空后再 `backward()`，梯度恢复为干净的 `[3,3,3]`（对应 `y = 3x` 的导数）

**训练循环结果**：
- 初始 loss `0.8233` → 最终 loss `0.1786`（下降 ~78%）
- 初始准确率 `41%`（接近随机猜）→ 最终准确率 `92%`
- 推理验证：`[-2,-2]` → 类别 0（概率 0.0000），`[2,2]` → 类别 1（概率 0.9989），`(0,0)` 边界点 → 类别 0（概率 0.4678，模型不确定）

---

### Q&A：训练循环核心概念

#### Q1: `optimizer.zero_grad()` 为什么必须在 `loss.backward()` 之前？放到 `step()` 之后行不行？

**A**: 本质上两种顺序都能工作，关键是保证**每次 `backward()` 之前梯度是干净的**。

训练循环的标准顺序是 `zero_grad → forward → backward → step`。如果把 `zero_grad` 放到 `step` 之后，变成 `forward → backward → step → zero_grad`，下一次循环的 `backward` 拿到的梯度仍然是干净的（因为上一轮末尾清零了），所以也能跑。

但 `step → zero_grad` 的写法有一个隐患：如果在训练循环中间插入了其他需要梯度的操作（比如梯度裁剪 `clip_grad_norm_`），它需要在 `backward` 之后、`step` 之前执行，此时梯度不能被提前清空。因此 PyTorch 社区约定俗成的写法是把 `zero_grad` 放在最前面，形成固定的心智模型。

**实际影响**：如果你忘记写 `zero_grad()`，梯度会不断累积。这在某些场景下是有意使用的技巧（比如显存不够时，用梯度累积模拟更大的 batch size），但正常训练时必须清零。

#### Q2: `BCEWithLogitsLoss` vs `BCELoss` 的区别？为什么推荐用前者？

**A**: 核心区别在于**是否内置 sigmoid**。

| 对比项 | `BCELoss` | `BCEWithLogitsLoss` |
|--------|-----------|---------------------|
| 输入要求 | 必须先在模型输出上手动 `sigmoid()` | 直接接收原始 logits（未压缩的实数值） |
| 内部实现 | `-(y * log(x) + (1-y) * log(1-x))` | 内部合并了 sigmoid + BCE 的计算 |
| 数值稳定性 | 容易溢出 | **更稳定** |

`BCEWithLogitsLoss` 更稳定的原因是它使用了 **log-sum-exp trick**：

当 logits 很大时，`sigmoid(x)` 会趋近于 0 或 1，然后 `log(sigmoid(x))` 就会出现 `log(0)` 的问题，产生 `-inf`。`BCEWithLogitsLoss` 把 sigmoid 和 BCE 合并成一个公式：

```
loss = max(x, 0) - x * y + log(1 + exp(-|x|))
```

这个公式在任何 x 值下都不会溢出，所以数值更稳定。

**实践建议**：只要最后一层输出的是 logits（没有经过 sigmoid/softmax），就用 `BCEWithLogitsLoss`（二分类）或 `CrossEntropyLoss`（多分类）。这是 PyTorch 官方推荐的做法。

#### Q3: 训练到 100 轮时准确率 92%，继续训练到 1000 轮会怎样？

**A**: 取决于数据集的复杂度和模型容量，可能出现三种情况：

1. **继续提升**（如果还没收敛）：loss 继续下降，准确率提升到 95%+。从结果看，60-100 轮之间 loss 还在明显下降（0.1892 → 0.1786），说明还没完全收敛。

2. **过拟合（Overfitting）**：训练集准确率继续上升到 99%+，但**测试集准确率下降**。模型"记住了"训练数据的噪声和特定样本，失去了泛化能力。这是最常见的问题。

3. **收敛 plateau**：loss 降到某个值后不再变化，准确率稳定在某个值附近。

**怎么判断是否过拟合**：

```
训练集准确率 ↑ 同时 测试集准确率 ↓ → 过拟合
训练集准确率 ↑ 同时 测试集准确率 ↑ → 还没收敛，可以继续训练
```

**防止过拟合的常见手段**：
- **早停（Early Stopping）**：监控验证集 loss，连续 N 轮不下降就停止训练
- **Dropout**：随机"关掉"一部分神经元，强迫模型不依赖特定特征
- **L2 正则化**：在 loss 中加上参数权重的惩罚项（Adam 优化器内置了 weight decay）
- **数据增强**：增加训练数据的多样性

---

## Step 2: HuggingFace Transformers — Tokenizer 与预训练模型推理

### 运行结果解读

代码文件：`code/phase7/phase7_step2_huggingface_basics.py`

**Pipeline 推理**：
- `"I love building AI systems"` → POSITIVE (0.9999)
- `"The shipping was delayed and the package arrived damaged"` → NEGATIVE (0.9997)
- `"Our customs clearance request has been pending for 3 days"` → NEGATIVE (0.9756)
- `pipeline` 封装了 tokenizer + model + 后处理，一行代码就能做推理

**Tokenizer 结果**：
- `"classifier"` 被 WordPiece 拆成 `['class', '##ifier']`（子词切分）
- `"I'm"` 被拆成 `['i', "'", 'm']`（标点也单独 tokenize）
- `token_type_ids` 在问答场景中标记 `0`=question、`1`=context
- 批量编码时自动 Padding，短句补 `[PAD]`（ID=0），`attention_mask` 标记 0 让模型忽略

**语义相似度矩阵**：

| 对比 | 相似度 | 解读 |
|------|--------|------|
| 物流延误 vs 物流延误(换说法) | **0.8746** | 最高，语义相近 |
| 清关问题 vs 物流延误(换说法) | **0.9208** | 很高（都属物流负面场景） |
| 物流延误 vs 清关问题 | **0.8444** | 中等偏高 |
| 物流延误 vs 编程(无关) | **0.8149** | 最低但仍偏高 |

**注意**：所有相似度都 > 0.81，说明 BERT 的 `[CLS]` embedding 不是为语义相似度优化的。生产环境做相似度搜索通常用 Sentence-BERT (SBERT) 或 SimCSE 等专门模型。

---

### Q&A：HuggingFace 核心概念

#### Q1: Tokenizer 的 input_ids、attention_mask、token_type_ids 分别是什么？

**A**: 三个都是 tokenizer 输出的整数序列，作用不同：

- **input_ids**：文本通过词表（vocab）映射成的数字 ID 序列。例如 `"hello"` → `7592`。模型只认识数字，不认识文字。BERT 的词表大小约 30,522。

- **attention_mask**：跟 input_ids 等长的 0/1 序列。`1` 表示这个位置是真实内容，`0` 表示是 padding。模型在做 self-attention 时会忽略 `attention_mask=0` 的位置，防止 padding 影响语义表示。

- **token_type_ids**：用于区分多段输入。最典型的场景是问答（question answering），question 部分标记为 `0`，context 部分标记为 `1`。模型通过这层信息知道哪些 token 属于问题、哪些属于参考文本。单段输入时可以忽略（全是 0）。

#### Q2: `[CLS]` token 为什么能代表整句的语义？

**A**: 关键在于 BERT 的预训练机制和 self-attention 的结构。

BERT 有两个预训练任务：
1. **Masked Language Model (MLM)**：随机遮住一些 token 让模型预测
2. **Next Sentence Prediction (NSP)**：判断两句话是否是连续的

`[CLS]` 出现在每个序列的开头。在 BERT 的每一层 self-attention 中，`[CLS]` 的表示都融合了所有其他 token 的信息（因为 attention 机制让每个 token 都能看到其他 token）。经过 12 层（BERT-base）的逐层聚合后，`[CLS]` 的最终隐藏状态（last_hidden_state）就包含了整句的语义信息。

在 NSP 任务中，模型需要在 `[CLS]` 上加一个分类头来判断两句话的关系。这证明了 `[CLS]` 已经"理解"了整句话的含义。所以在下游分类任务中，通常取 `[CLS]` 的向量加一个 Linear 层做分类。

**但注意**：`[CLS]` 适合分类任务（加分类头 fine-tuning），但不直接适合语义相似度任务。因为 BERT 没有显式地训练"两个句子的向量要靠近"这种目标。这就是为什么 Sentence-BERT 会对 BERT 做额外的对比学习训练。

#### Q3: `AutoModel` vs `AutoModelForSequenceClassification` vs `pipeline` 有什么区别？什么时候用哪个？

**A**: 三个是不同层次的抽象：

| 层级 | 工具 | 输出 | 使用场景 |
|------|------|------|----------|
| 最低层 | `AutoModel` | 每个 token 的 hidden state | 需要自定义模型结构时（如加自定义分类头、做特征提取） |
| 中层 | `AutoModelForSequenceClassification` | 分类 logits | 已有预训练分类模型，直接做推理或 fine-tuning |
| 最高层 | `pipeline` | 格式化后的结果 | 快速原型验证、不需要控制细节 |

**选择建议**：
- 做实验/验证想法 → 用 `pipeline`，最快
- 做 fine-tuning → 用 `AutoModelForSequenceClassification`（HuggingFace Trainer 需要）
- 做自定义架构（比如多模态、图神经网络）→ 用 `AutoModel` 拿底层表示

#### Q4: 为什么加载模型时有 `UNEXPECTED keys` 警告？

**A**: 这是因为下载的预训练权重和当前加载的模型架构不完全匹配。

比如用 `AutoTokenizer` 加载 `bert-base-uncased` 的 tokenizer，但用 `AutoModel`（不是 `AutoModelForPreTraining`）加载模型。预训练权重里包含了 `cls.predictions.*`（MLM 头）和 `cls.seq_relationship.*`（NSP 头）的参数，但 `AutoModel` 只需要 encoder 部分，不需要这些分类头。所以这些 key 是 "UNEXPECTED"（意外的、用不上的）。

这是**正常的**，不代表有问题。只要你不是在做精确的权重恢复（比如 resume training），可以忽略这个警告。

---

## Step 3: 物流文本分类 — BERT 全量微调

### 运行结果解读

代码文件：`code/phase7/phase7_step3_logistics_classifier.py`

**数据集**：520 条，8 类（每类 65 条），train/test split = 416/104

**训练过程**：

| Epoch | Train Loss | Train Acc | Val Acc | 解读 |
|-------|-----------|-----------|---------|------|
| 1 | 1.3243 | 54.57% | **86.54%** | 第一轮验证集就达到 86%，BERT 的预训练知识迁移效果明显 |
| 2 | 0.3070 | 92.79% | **95.19%** | 快速收敛 |
| 3 | 0.0928 | 98.32% | **96.15%** | 训练集接近完美，验证集略有提升 |

**最终评估**：
- 测试集准确率：**96.15%**（104 条中 100 条正确）
- F1 (weighted)：**0.9616**
- F1 (macro)：**0.9612**

**混淆矩阵分析**（4 条错误）：

| 真实 → 预测 | 数量 | 分析 |
|-------------|------|------|
| shipping_delay → tracking_issue | 1 | 延误和追踪异常容易混淆（都涉及"物流没更新"） |
| cargo_damage → shipping_delay | 1 | 货损描述中可能包含"没收到"等延误相关措辞 |
| cargo_damage → customs_clearance | 1 | 可能样本中提到了"海关查验后损坏" |
| general → tracking_issue | 1 | 通用问题中可能包含"怎么查物流"之类的表述 |

**推理演示**（5/5 全部正确）：
- "我的包裹已经晚了一周了" → shipping_delay (0.9782)
- "下一班到鹿特丹的船期" → schedule_inquiry (0.9812)
- "货物卡在海关了" → customs_clearance (0.9955)
- "你们公司支持哪些支付方式" → general (0.9923)
- "货物外包装破损" → cargo_damage (0.9954)

---

### Q&A：BERT 微调实战

#### Q1: 为什么第一轮训练准确率只有 54% 但验证集就有 86%？

**A**: 这是正常的，原因是 train loss 是**每个 batch 的平均**，而准确率是**整个 epoch 结束后的全局统计**。

第一轮训练时，模型参数还是预训练的初始值（随机初始化了分类头），所以前几个 batch 的 loss 很高。随着训练进行，分类头快速学习，后面 batch 的准确率逐渐提升。但 train loss 反映的是整个 epoch 的平均，所以看起来低。而验证集的准确率是在模型经过一轮更新后评估的，此时分类头已经学到了初步的分类能力。

这体现了 **迁移学习** 的价值：BERT 在预训练阶段已经学会了理解中文语义，我们只需要训练一个 Linear 分类头来映射 `[CLS]` 向量到类别标签。

#### Q2: 训练集准确率 98% vs 验证集 96%，有过拟合吗？

**A**: 差距只有 2%，**不算过拟合**。这是正常的泛化差距。

判断过拟合的标准：
- 训练集准确率 >> 验证集准确率（差距 > 5-10%）→ 过拟合
- 训练集和验证集准确率都高，差距小 → 正常收敛
- 验证集准确率下降，训练集继续上升 → 开始过拟合

当前情况是 98% vs 96%，差距很小，模型收敛良好。如果继续训练到 10+ epoch，训练集可能到 100% 但验证集开始下降，那时候就是过拟合了。

#### Q3: 为什么只有 520 条数据就能达到 96% 的准确率？

**A**: 三个原因：

1. **预训练知识迁移**：BERT 在大规模中文语料上预训练过，已经掌握了中文的语法和语义。我们不是"从零训练"，而是"微调"一个已经很聪明的模型。

2. **类别区分度高**：8 个类别的文本特征差异明显。"船期查询"类包含大量地名和时间词，"订舱请求"包含柜型和航线描述，"账单争议"包含费用相关词汇。这些特征对 BERT 来说很容易区分。

3. **数据集质量**：种子数据是人工编写的，每类内的表述风格一致，类间差异大。如果是真实场景的用户数据，类别边界会更模糊，准确率会低一些。

**面试时要注意**：96% 是基于人工编写的种子数据，真实业务数据上可能会降到 85-90%。需要补充数据增强（同义改写、噪声注入）来提升鲁棒性。

#### Q4: `CrossEntropyLoss` 和 `BCEWithLogitsLoss` 的区别？为什么多分类用前者？

**A**: 两者的数学定义不同：

- **BCEWithLogitsLoss**：二元交叉熵，适用于**二分类**。每个样本只属于一个类别（是/否），输出一个 logit，经过 sigmoid 后跟标签比较。

- **CrossEntropyLoss**：多元交叉熵，适用于**多分类（互斥）**。每个样本只属于一个类别，但类别数 > 2。输出 N 个 logits（N = 类别数），经过 softmax 后变成概率分布，再跟 one-hot 标签比较。

数学关系：`CrossEntropyLoss = LogSoftmax + NLLLoss`。它内部会自动做 softmax，所以模型最后一层**不需要**加 `nn.Softmax()`，直接输出 logits 即可。

如果是**多标签分类**（一个样本可以同时属于多个类别），则需要用 `BCEWithLogitsLoss`，每个类别独立输出一个 logit。

#### Q5: 训练时为什么要用 `model.train()` 和 `model.eval()`？

**A**: 这两个方法控制模型中某些层的**训练/推理行为差异**：

- `model.train()`：开启训练模式。`Dropout` 层会随机丢弃一部分神经元（防止过拟合），`BatchNorm` 层会用当前 batch 的统计量更新 running mean/var。

- `model.eval()`：开启推理模式。`Dropout` 层**停止随机丢弃**（所有神经元都参与推理），`BatchNorm` 层使用训练阶段积累的 running mean/var。

如果在推理时忘记切换到 `eval()`，Dropout 会随机丢弃神经元，导致**同一输入每次输出不同**——这在生产环境中是不可接受的。所以推理时必须用 `model.eval()` + `torch.no_grad()`（同时禁用梯度计算节省内存）。

---

## Step 4: Qwen LoRA 微调 + Zero-Shot 基线对比

### 本地运行结果：Zero-Shot Baseline

代码文件：`code/phase7/phase7_step4_zeroshot_baseline.py`
Colab Notebook：`code/phase7/phase7_step4_qwen_lora_colab.ipynb`

**Zero-Shot (qwen-plus) 评估结果**：
- 测试集准确率：**85.58%**（104 条中 89 条正确）
- F1 (weighted)：**0.8437**
- 比 BERT 微调低了 ~10 个百分点

**各类别表现**：

| 类别 | Precision | Recall | F1 | 分析 |
|------|-----------|--------|-----|------|
| shipping_delay | 1.0000 | **0.6000** | 0.7500 | Recall 很低，延误相关表述被误判为 tracking_issue |
| customs_clearance | 0.9231 | 1.0000 | 0.9600 | 好 |
| tracking_issue | 0.8000 | 0.8571 | 0.8276 | 尚可 |
| billing_dispute | 0.7647 | **1.0000** | 0.8667 | Precision 偏低，其他类别被误判为 billing |
| schedule_inquiry | 1.0000 | 0.9231 | 0.9600 | 好 |
| cargo_damage | 0.8889 | 1.0000 | 0.9412 | 好 |
| booking_request | 0.7059 | 1.0000 | 0.8276 | Precision 偏低，general 被误判 |
| general | 1.0000 | **0.4286** | 0.6000 | Recall 极低，大部分 general 被误判为其他类别 |

**关键发现**：

1. **Zero-shot 不如 fine-tuned**：qwen-plus（~100B 参数）zero-shot 准确率 85.58%，而 BERT-base-chinese（102M 参数）fine-tuned 达到 96.15%。说明对于**特定领域的分类任务**，少量标注数据 + 微调比大模型 zero-shot 更有效。

2. **general 类别是最大短板**：Zero-shot 下 general 的 recall 只有 42.86%，意味着大部分通用问题都被误判为其他具体类别。这是因为 LLM 倾向于"过度解读"用户意图——即使是一般性询问，它也会尝试找到一个具体的类别。

3. **shipping_delay recall 低**：60% 的延误消息被误判为 tracking_issue。这合理，因为"物流没更新"既是延误的表现，也是追踪问题，边界模糊。

### 三模型对比

| 模型 | 参数量 | 可训练参数 | 训练方式 | Accuracy | F1 (weighted) |
|------|--------|-----------|----------|----------|---------------|
| Zero-Shot (qwen-plus) | ~100B | 0 | Prompt only | **0.8558** | **0.8437** |
| BERT-base-chinese | 102M | 102M | Full fine-tuning | **0.9615** | **0.9616** |
| Qwen-LoRA (Colab) | 1.5B | ~8M (0.5%) | LoRA | pending | pending |

**面试观点**：
- Zero-shot 适合**快速原型验证**，不需要训练成本，但精度有限
- Fine-tuning 在**特定领域任务**上远超 zero-shot（+10% 准确率）
- LoRA 的目标：在接近 BERT 微调的精度下，只训练 0.5% 的参数（省显存、防灾难性遗忘）

---

### Q&A：LoRA 与微调

#### Q1: 什么是 LoRA？为什么只训练 q_proj 和 v_proj？

**A**: LoRA（Low-Rank Adaptation）的核心思想是：**模型的微调不需要更新全部参数，只需要在关键层添加低秩矩阵即可。**

Transformer 的 self-attention 中，Q/K/V 矩阵负责把输入投影到不同的子空间。研究发现，微调时主要需要调整的是 **Q（query）和 V（value）** 的投影方向：
- Q 决定了"模型关注什么"（attention pattern）
- V 决定了"模型输出什么信息"

K（key）和 O（output）矩阵相对稳定，不需要大幅调整。所以 LoRA 通常只在 `q_proj` 和 `v_proj` 上加低秩适配器。

LoRA 的数学原理：原本 Q = W @ x，LoRA 改为 Q = (W + ΔW) @ x，其中 ΔW = A @ B，A 是 (d_model × r) 矩阵，B 是 (r × d_model) 矩阵。r 就是 LoRA rank（我们设 r=16），远小于 d_model（768 或 1536）。

**效果**：原本需要更新 1.5B 参数，用 LoRA (r=16) 只需要更新约 8M 参数（0.5%）。

#### Q2: LoRA 相比全量微调有什么优势？

**A**: 三个核心优势：

1. **参数高效**：只训练 0.5%-1% 的参数，显存需求大幅降低。1.5B 模型全量微调需要 ~30GB 显存，LoRA 只需 ~8GB（T4 够用）。

2. **避免灾难性遗忘（Catastrophic Forgetting）**：预训练模型的绝大部分参数保持不变，不会丢失通用语言能力。LoRA 只是"叠加"了一个小的适配器。

3. **多任务共享**：同一个 base model 可以加载不同的 LoRA 权重，实现多任务切换。比如一个物流意图 LoRA + 一个客服回复 LoRA，共享同一个 base model。

#### Q3: 为什么 zero-shot qwen-plus 反而不如 fine-tuned BERT？

**A**: 这涉及到**任务匹配度**和**模型知识迁移**的问题：

- qwen-plus 虽然有更强的通用能力，但 zero-shot 下它只能依靠 pre-training 阶段学到的"常识"。物流场景的 8 个业务类别（如 shipping_delay vs tracking_issue）有细微的语义边界，zero-shot 很难精确区分。

- BERT 经过 fine-tuning 后，分类头已经**专门学习**了这 8 个类别的决策边界。训练数据中包含了"什么是延误、什么是追踪问题"的具体示例，模型直接学习了这些模式。

**类比**：qwen-plus 像一个什么都知道的博士生，但没学过你们公司的业务分类；BERT 像一个只学了 520 道例题的本科生，但考的全是例题范围内的题。对于特定考试，后者反而考得更好。

**实践指导**：
- 任务明确、有标注数据 → fine-tuning 效果更好
- 任务探索阶段、没有标注数据 → zero-shot 做 baseline
- 最佳方案：zero-shot 筛掉明显能解决的 case，fine-tuned 模型处理边界 case

---

## Step 5: 评估对比 + FastAPI 部署 + 项目收尾

### 运行结果

代码文件：`code/phase7/phase7_step5_summary.py`、`code/phase7/phase7_step5_serve.py`
项目文档：`code/phase7/PROJECT_README.md`

**三模型评估对比**：

| 模型 | 参数量 | 可训练 | 训练方式 | Accuracy | F1 (weighted) |
|------|--------|--------|----------|----------|---------------|
| BERT-base-chinese | 102M | 102M | Full fine-tuning | **0.9615** | **0.9616** |
| qwen-plus (API) | ~100B | 0 | Zero-shot | **0.8558** | **0.8437** |
| Qwen2.5-1.5B + LoRA | 1.5B | ~8M (0.5%) | LoRA | pending | pending |

**各类别 F1 对比**：

| 类别 | BERT | Zero-Shot | 差异 |
|------|------|-----------|------|
| shipping_delay | 0.9000 | 0.7500 | +15.0% |
| customs_clearance | 0.9600 | 0.9600 | 0 |
| tracking_issue | 0.9333 | 0.8276 | +10.6% |
| billing_dispute | 1.0000 | 0.8667 | +13.3% |
| schedule_inquiry | 1.0000 | 0.9600 | +4.0% |
| cargo_damage | 0.9333 | 0.9412 | -0.8% |
| booking_request | 1.0000 | 0.8276 | +17.2% |
| general | 0.9630 | 0.6000 | +36.3% |

**关键发现**：
1. BERT fine-tuning 比 zero-shot 准确率高 **10.6%** — 对于特定领域分类任务，少量标注数据 + 微调远超大模型 zero-shot
2. `general` 类别差异最大（+36.3%）— zero-shot 下 LLM 倾向于"过度解读"，把通用问题判为具体类别
3. `cargo_damage` 是唯一 zero-shot 略优于 fine-tuned 的类别（差异 -0.8%）— 可能因为"破损"等关键词在 LLM 预训练语料中语义已经很清晰
4. 8/8 推理演示全部正确，置信度 0.97-0.99

**FastAPI 服务**：
- 4 个端点：`POST /predict`、`POST /predict/batch`、`GET /health`、`GET /model/info`
- 支持单条和批量预测，返回意图类别 + 置信度 + 可选的全类别分数

---

### Phase 7 总结

**完成的 5 个 Step**：

| Step | 内容 | 核心产出 |
|------|------|----------|
| 1 | PyTorch 基础 | 手写训练循环、理解 autograd |
| 2 | HuggingFace | Tokenizer、AutoModel、[CLS] embedding |
| 3 | BERT 全量微调 | 520条数据、8分类、F1 0.9616 |
| 4 | Zero-Shot + LoRA | Zero-Shot F1 0.8437、LoRA Colab notebook |
| 5 | 评估对比 + 部署 | 三模型对比报告、FastAPI 服务 |

**面试核心论据**：
1. 会用 PyTorch 手写训练循环（从 Tensor 到 backward 到 optimizer.step）
2. 理解 Tokenizer 和 Transformer 架构（input_ids、attention_mask、[CLS]）
3. 完整 ML pipeline 实战（数据 → tokenize → 训练 → 验证 → 评估 → 保存）
4. 理解 LoRA 参数高效微调原理（低秩分解、q_proj/v_proj）
5. 能做 zero-shot vs fine-tuned 的技术选型（有数据就微调，没数据就 zero-shot）
6. 能部署为可调用服务（FastAPI）
