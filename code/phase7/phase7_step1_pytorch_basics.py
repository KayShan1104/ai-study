"""
Phase 7 Step 1: PyTorch 基础 — 从张量到训练循环

目标：
1. 理解 Tensor 和自动求导（autograd）
2. 手写一个完整的训练循环（线性回归二分类）
3. 理解 nn.Module、loss、optimizer 的协作关系
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Part 1: Tensor 基础 — 可以理解为 "支持自动求导的 numpy"
# ============================================================

print("=" * 60)
print("Part 1: Tensor 与自动求导 (autograd)")
print("=" * 60)

# --- 1.1 基础运算 ---
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"\n1.1 基础运算")
print(f"  x + y = {x + y}")
print(f"  x * y = {x * y}")
print(f"  x.dot(y) = {x.dot(y)}")  # 点积 = 1*4 + 2*5 + 3*6 = 32

# --- 1.2 自动求导 ---
# requires_grad=True 告诉 PyTorch: "这个张量需要追踪梯度"
x_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# y = x^2 + 3x
y_func = x_grad ** 2 + 3 * x_grad
# dy/dx = 2x + 3
# x=1 → dy/dx=5, x=2 → dy/dx=7, x=3 → dy/dx=9

# backward() 需要传入一个跟 y_func 形状相同的张量（梯度权重）
# 因为 y_func 是标量求和后的结果，这里传入 ones 表示每个元素权重相等
y_func.sum().backward()

print(f"\n1.2 自动求导")
print(f"  函数: y = x^2 + 3x")
print(f"  理论导数: dy/dx = 2x + 3 → [5, 7, 9]")
print(f"  PyTorch 计算: dy/dx = {x_grad.grad}")
# tensor([5., 7., 9.])  ✓  跟理论一致

# --- 1.3 理解 zero_grad ---
# 如果不 zero_grad，梯度会累积
y_func2 = x_grad ** 2  # 新的计算
y_func2.sum().backward()
print(f"\n1.3 梯度累积（没有 zero_grad）")
print(f"  第二次 backward 后，梯度变成了: {x_grad.grad}")
# 现在是 [5+2, 7+4, 9+6] = [7, 11, 15]，两次梯度叠加了

# 创建一个新的 tensor 演示 zero_grad 的作用
x_clean = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y1 = x_clean ** 2
y1.sum().backward()
print(f"\n  第一次 backward: {x_clean.grad}")
x_clean.grad.zero_()  # 清空梯度
y2 = x_clean * 3
y2.sum().backward()
print(f"  zero_grad 后第二次 backward: {x_clean.grad}")
# 这次只有 dy/dx=3 的梯度，不会跟之前叠加

# ============================================================
# Part 2: nn.Module — 模型的容器
# ============================================================

print("\n" + "=" * 60)
print("Part 2: nn.Module — 构建神经网络")
print("=" * 60)

# --- 2.1 Sequential 模型 ---
model = nn.Sequential(
    nn.Linear(2, 16),     # 输入 2 维，输出 16 维
    nn.ReLU(),            # 激活函数
    nn.Linear(16, 1)      # 输出 1 维（二分类的 logit）
)

print(f"\n2.1 模型结构:")
print(model)
# 参数统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数数: {total_params}")  # (2*16+16) + (16*1+1) = 48 + 17 = 65

# --- 2.2 自定义 Module（推荐方式，更灵活）---
class SimpleClassifier(nn.Module):
    """一个简单的前馈神经网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model2 = SimpleClassifier(input_dim=2, hidden_dim=16, output_dim=1)
print(f"\n2.2 自定义模型:")
print(model2)

# --- 2.3 前向传播 ---
dummy_input = torch.tensor([[1.0, -0.5], [0.3, 0.7]])
output = model2(dummy_input)
print(f"\n2.3 前向传播示例:")
print(f"  输入: {dummy_input.tolist()}")
print(f"  输出 (logits): {output.squeeze().tolist()}")
# 这里的输出是随机的（因为参数随机初始化），就是模型"猜"的答案

# ============================================================
# Part 3: 完整的训练循环
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 完整训练循环 — 二分类")
print("=" * 60)

# --- 3.1 生成模拟数据 ---
# 生成两类点：类别 0 在左下角，类别 1 在右上角
torch.manual_seed(42)

n_samples = 100
X_class0 = torch.randn(n_samples // 2, 2) - torch.tensor([1.0, 1.0])
X_class1 = torch.randn(n_samples // 2, 2) + torch.tensor([1.0, 1.0])
X = torch.cat([X_class0, X_class1], dim=0)
y = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)])

print(f"\n3.1 数据集: {X.shape[0]} 条样本，每个样本 2 个特征")
print(f"  类别 0: {int((y == 0).sum())} 条，类别 1: {int((y == 1).sum())} 条")

# --- 3.2 训练配置 ---
model = SimpleClassifier(input_dim=2, hidden_dim=32, output_dim=1)
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵（内部包含 sigmoid）
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\n3.2 训练配置:")
print(f"  Loss 函数: BCEWithLogitsLoss")
print(f"  Optimizer: Adam (lr=0.01)")
print(f"  训练轮数: 100 epochs")

# --- 3.3 训练循环 ---
losses = []
accuracies = []

for epoch in range(100):
    # Step A: 前向传播
    logits = model(X).squeeze()
    loss = criterion(logits, y)

    # Step B: 反向传播
    optimizer.zero_grad()  # ← 别忘了清空梯度！
    loss.backward()
    optimizer.step()  # 用梯度更新参数

    # 记录
    losses.append(loss.item())
    preds = (logits > 0).float()  # logit > 0 等价于 sigmoid > 0.5
    acc = (preds == y).sum().item() / len(y)
    accuracies.append(acc)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

# --- 3.4 训练结果可视化 ---
print(f"\n3.3 训练趋势:")
print(f"  初始 loss: {losses[0]:.4f} → 最终 loss: {losses[-1]:.4f}")
print(f"  初始 acc:  {accuracies[0]:.4f} → 最终 acc:  {accuracies[-1]:.4f}")

# --- 3.5 推理验证 ---
print(f"\n3.5 推理验证:")
model.eval()  # 切换到评估模式（关闭 dropout 等训练专属行为）
with torch.no_grad():  # 推理不需要梯度，节省内存
    test_inputs = torch.tensor([
        [-2.0, -2.0],  # 明显是类别 0
        [2.0, 2.0],    # 明显是类别 1
        [0.0, 0.0],    # 边界情况
    ])
    test_logits = model(test_inputs).squeeze()
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs > 0.5).long()

    for i, (inp, prob, pred) in enumerate(zip(test_inputs, test_probs, test_preds)):
        print(f"  输入 {inp.tolist()} → 概率: {prob.item():.4f} → 预测: 类别 {pred.item()}")

# ============================================================
# Part 4: 关键概念回顾
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 关键概念回顾")
print("=" * 60)
print("""
训练循环的 4 步（必须按顺序）:
  1. optimizer.zero_grad()   — 清空上次梯度，否则会累积
  2. logits = model(X)       — 前向传播，模型"猜"答案
  3. loss = criterion(logits, y)  — 计算"猜"的有多差
  4. loss.backward()         — 反向传播，算出每个参数的梯度
  5. optimizer.step()        — 用梯度更新参数，让下次猜得更好

重要概念:
  - Tensor: 支持自动求导的多维数组（= numpy array + 梯度追踪）
  - requires_grad: 是否需要计算梯度（模型参数默认 True，输入数据默认 False）
  - nn.Module: 所有模型的基类，封装了参数管理和 forward/backward
  - BCEWithLogitsLoss: 二分类 loss，内部自动做 sigmoid + BCE，数值更稳定
  - model.train() / model.eval(): 切换训练/评估模式（影响 dropout、batchnorm 等）
  - torch.no_grad(): 推理时禁用梯度计算，省内存省时间
""")
