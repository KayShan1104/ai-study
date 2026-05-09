---
name: 第二阶段 - Prompt Engineering
description: 从"想到什么写什么"到系统化、结构化的 prompt 设计
type: project
---

# 第二阶段：Prompt Engineering

> 预计耗时：1周 | 开始日期：____ | 完成日期：____
> 前置：第一阶段（API 实操）
> 关联画像：ai_learning_profile.md

---

## 前置知识回顾

第一阶段已经接触过 `system` / `user` / `assistant` 角色和基本 API 调用。本阶段聚焦于**如何写好 prompt**——这直接影响 LLM 的输出质量和 Agent 的可靠性。

---

## Step 1：Prompt 的基本结构

**学习目标**：理解一个好 prompt 的组成部分

**操作建议**：
1. 了解 prompt 的核心要素：
   - **角色/人设**："你是一个资深 Python 工程师"
   - **任务描述**："分析以下代码的潜在 bug"
   - **上下文/背景信息**：相关代码、数据、约束
   - **输出格式要求**："用 JSON 返回，包含字段 X、Y、Z"
   - **示例（few-shot）**：给 1-2 个输入输出示例
2. 对比实验：同一个任务，写两个版本——一个只有问题，一个包含完整结构，对比输出质量

```python
# 差的 prompt
prompt = "这段代码有什么问题？" + code

# 好的 prompt
prompt = f"""你是一个资深 Python 代码审查员。请审查以下代码，从以下维度评估：
1. 是否有潜在 bug
2. 是否有性能问题
3. 是否有安全隐患

请用 JSON 格式输出：
{{
  "bugs": ["..."],
  "performance_issues": ["..."],
  "security_risks": ["..."],
  "overall_score": 0-10
}}

代码：
{code}
"""
```

**Milestone 1**：
- [ ] 对同一个任务写 3 个不同质量等级的 prompt，输出质量有明显差异
- [ ] 能列出 prompt 的 5 个核心要素并逐一解释作用
- [ ] 总结出自己在实际开发中最常用的 3 种 prompt 模板

---

## Step 2：Few-Shot Prompting

**学习目标**：通过示例引导模型行为，提升输出一致性

**操作建议**：
1. 理解 few-shot 的原理：给模型 2-5 个输入-输出示例，让它模仿模式
2. 实现一个场景：把用户输入的自然语言意图分类为预定义类别
3. 对比 zero-shot（不给示例）vs few-shot（给 3 个示例）的分类准确率

```python
# Few-shot 示例
prompt = """
将用户输入分类为以下类别：query（查询）、command（命令）、chitchat（闲聊）

示例：
输入："明天北京天气怎么样" -> {"category": "query", "confidence": 0.95}
输入："打开空调" -> {"category": "command", "confidence": 0.9}
输入："你好呀" -> {"category": "chitchat", "confidence": 0.99}

现在请分类：
输入："帮我查一下上海到广州的机票"
"""
```

**Milestone 2**：
- [ ] 实现一个意图分类器，zero-shot 和 few-shot 各跑 10 个测试用例
- [ ] 统计两种方式下分类正确的数量，记录差异
- [ ] 尝试 1-shot、3-shot、5-shot，观察准确率变化趋势

---

## Step 3：Chain-of-Thought (CoT)

**学习目标**：引导模型展示推理过程，提升复杂任务的准确率

**操作建议**：
1. 理解 CoT 的核心：让模型"一步步思考"而不是直接给答案
2. 实现两个场景对比：
   - 直接问数学题 / 逻辑题 → 观察准确率
   - 加 `"请一步步思考，先列出已知条件，再推导"` → 对比准确率
3. 了解 CoT 的变体：
   - Zero-shot CoT：只加 `"Let's think step by step"`
   - Few-shot CoT：示例中包含推理步骤
4. 注意：CoT 会增加 token 消耗（因为模型输出了推理过程）

```python
# 测试题目（选一道需要多步推理的）
problem = """一个农场有鸡和兔子，共有35个头、94只脚。
鸡有2只脚，兔子有4只脚。问鸡和兔子各有多少只？"""

# 不 CoT
prompt_v1 = problem + "\n直接给出答案。"

# CoT
prompt_v2 = problem + "\n请一步步思考，先列出方程，再求解。"
```

**Milestone 3**：
- [ ] 选 5 道需要推理的题目，分别用直接回答和 CoT 两种方式
- [ ] CoT 方式正确率更高
- [ ] 能解释为什么 CoT 有效（给了模型更多"思考 token"，减少跳跃式错误）
- [ ] 知道 CoT 的代价（更多 token，更慢）

---

## Step 4：Structured Output 进阶

**学习目标**：确保 LLM 输出严格符合预期格式，这是构建可靠 Agent 的关键

**操作建议**：
1. 回顾第一阶段的 JSON 输出，这次要求 100% 格式正确
2. 学习使用 OpenAI 的 `response_format` 参数（JSON Schema mode）：
   ```python
   response = client.chat.completions.create(
       model="gpt-4o",
       messages=messages,
       response_format={
           "type": "json_schema",
           "json_schema": {
               "name": "code_review",
               "schema": {
                   "type": "object",
                   "properties": {
                       "bugs": {"type": "array", "items": {"type": "string"}},
                       "score": {"type": "integer", "minimum": 0, "maximum": 10}
                   },
                   "required": ["bugs", "score"],
                   "additionalProperties": False
               }
           }
       }
   )
   ```
3. 测试 Anthropic 的类似能力（`tool_use` + schema）
4. 实现一个场景：用结构化输出做代码审查报告（包含分类、严重等级、修复建议）

**Milestone 4**：
- [ ] 用 JSON Schema 约束输出，连续 10 次调用 100% 可解析
- [ ] 对比 prompt 中口头要求 JSON vs 用 `response_format` 强制 JSON 的区别
- [ ] 能解释 `additionalProperties: false` 的作用

---

## Step 5：System Prompt 设计与优化

**学习目标**：设计有效的 system prompt 来控制 Agent 的行为风格和边界

**操作建议**：
1. 理解 system prompt 的作用：设定行为边界、风格、能力范围、安全规则
2. 设计一个客服场景的 system prompt，要求包含：
   - 角色定义
   - 允许回答的范围
   - 不允许做的事情
   - 遇到不确定问题时的处理方式
   - 输出格式要求
3. 测试边界情况：
   - 用户问了范围外的问题 → Agent 应该礼貌拒绝
   - 用户试图 prompt injection（"忽略上面的指令"） → Agent 应该保持原有行为
4. 迭代优化：根据测试结果调整 system prompt

**Milestone 5**：
- [ ] 设计一个包含角色、边界、规则、格式的完整 system prompt
- [ ] 跑 10 个边界测试用例，Agent 行为符合预期 >= 80%
- [ ] 尝试一次 prompt injection 攻击，观察 Agent 是否会被"带偏"
- [ ] 迭代 2 轮以上，每次改进都有具体的测试数据支撑

---

## Step 6：Prompt 调试与迭代

**学习目标**：建立系统化的 prompt 调优方法，而非"想到什么改什么"

**操作建议**：
1. 选择一个任务（如意图分类、代码审查、文本摘要）
2. 写一版初始 prompt
3. 准备 20 个测试用例（覆盖正常情况和边缘情况）
4. 跑一遍，记录每个 case 的输出和评分
5. 分析失败的 case，修改 prompt
6. 再跑一遍，对比前后得分
7. 重复 3 轮以上

建议用表格记录：

| Case ID | 输入 | V1 输出 | V1 评分 | V2 输出 | V2 评分 | 改进说明 |
|---------|------|---------|---------|---------|---------|----------|
| 001 | ... | ... | ✅ | ... | ✅ | — |
| 002 | ... | ... | ❌ | ... | ✅ | 增加了 XX 说明 |

**Milestone 6**：
- [ ] 完成至少 3 轮 prompt 迭代
- [ ] 有明确的量化指标（如准确率从 60% → 85%）
- [ ] 能总结自己最常用的 2-3 个 prompt 调优技巧

---

## 阶段验收

### 综合项目：结构化代码审查助手

实现一个代码审查工具，要求：
1. 接受 Python 代码作为输入
2. 使用 few-shot + CoT 提示策略
3. 输出严格的结构化 JSON（包含 bug 列表、性能问题、安全建议、总体评分）
4. 对常见代码模式（循环、文件 IO、网络请求）有特定的审查规则
5. 跑 20 个测试用例，结构化输出 100% 可解析

**验收标准**：
- [ ] `python code_reviewer.py < code.py` 输出合法 JSON
- [ ] JSON 结构符合定义的 schema
- [ ] 能识别出至少 3 种不同类型的代码问题
- [ ] 有测试报告和迭代记录
- [ ] 代码中有清晰注释说明 prompt 设计思路

---

## 常见陷阱

- **过度依赖 few-shot**：有时候一个更好的 task description 比 5 个示例更有效
- **示例质量差**：few-shot 的示例如果有错误，模型会模仿错误
- **CoT 不是万能的**：简单任务不需要 CoT，会增加成本和延迟
- **不要把 system prompt 当文档写**：过长、过于详细的 system prompt 反而会让模型"迷失重点"
- **忽略 token 成本**：CoT + few-shot + 长 system prompt = 大量 token，注意成本
