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
