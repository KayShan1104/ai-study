# Phase 1 学习笔记

> **日期**: 2026-05-09
> **前置知识**: LLM API 基础调用、多轮对话、Structured Output
> **项目目录**: `D:\workspace\study\`
> **关联文档**: `plan/phase1_api_practice.md`
> **后续阶段**: [[Phase4 学习笔记]] — Agent 框架

---

## Step 1：第一次 API 调用

### temperature 参数详解

**范围**: 0 到 2

| 值 | 含义 |
|---|------|
| 0 | 完全确定，每次回答相同 |
| 2 | 最大随机性，最具创造性 |

**常用范围**（0 到 1.5）:

| 范围 | 场景 |
|---|------|
| 0-0.3 | 事实性内容 |
| 0.7-1.0 | 平衡对话 |
| 1.0-1.5 | 创意内容 |
| 1.5-2.0 | 高度随机（较少使用） |

**实际建议**:

| 场景 | 推荐值 |
|---|------|
| 事实性问答 | temperature = 0（最准确） |
| 日常对话 | temperature = 0.7（平衡） |
| 创意写作 | temperature = 1.0+（有创意） |
| 代码生成 | temperature = 0-0.3（准确性优先） |

### system / user / assistant 三个角色

| 角色 | 作用 | 特点 | 示例 |
|---|---|---|---|
| system | 设定 AI 的身份、性格、规则 | 只在对话开始时设置一次，AI 会记住这个设定 | "你是一个专业的 Python 程序员" |
| user | 代表用户的输入和问题 | 每次用户说的话都是 user 角色 | "Python 中列表和元组有什么区别？" |
| assistant | 代表 AI 的回复 | API 返回的 AI 回答都是 assistant 角色 | 在多轮对话中需要把 AI 之前的回复也传回去，维持上下文 |

### 实际产品中的 system 角色定义

以 Claude Code CLI 为例，用户可以通过以下方式给 AI 加"人设"：

| 方式 | 注入位置 | 强度 | 适合场景 |
|---|---|---|---|
| `--append-system-prompt` | 真正的 system prompt | 最强 | 脚本/自动化，每次启动传参 |
| CLAUDE.md 文件 | 用户消息（紧挨 system） | 强 | 人设、项目规范，最常用 |
| `.claude/rules/` | 条件触发的用户消息 | 强（命中时） | 按文件类型分流规则 |
| settings.json | 配置层 | 中等 | 模型、语言、权限，不定义人设 |
| Auto Memory | 启动时的补充消息 | 补充 | 自动积累用户偏好 |

**理解要点**：在 API 层面 `system` 是消息列表里的一条记录；在实际产品中，它是各种持久化配置、prompt 文件和设置项的集合。

---

## Step 2：多轮对话与上下文管理

### 为什么 LLM 需要每次传完整消息历史

> 因为 LLM 是无状态的，每次对话都需要提供完整的上下文信息。

### 模型上下文窗口

| 模型 | 窗口大小 |
|---|---|
| qwen3.6-plus | 32768 tokens |

---

## Structured Output（结构化输出）

### try/except 处理模型格式错误

对应 Milestone 3 要求："能处理模型偶尔输出格式错误的情况"。

**位置**: `phase1_step3_structured_output.py` 的 `json_with_error_handling()` 函数（第 118-168 行）

```python
try:
    parsed_data = json.loads(raw_output)
    print(f"✅ 解析成功: {parsed_data}")
    return parsed_data
except json.JSONDecodeError as e:
    print(f"❌ JSON解析失败: {e}")
    # 重试机制：把错误输出追加到消息历史，让模型重新输出
    if attempt < 2:
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({"role": "user", "content": "输出不是有效JSON，请重新输出，只要JSON格式。"})
```

这段做了两件事：
1. **捕获** `json.JSONDecodeError`，防止程序崩溃
2. **重试**——把错误输出追加到消息历史，让模型重新输出

其余 5 处（第 42、112、205、259、345 行）是简单的 try/except，只打印错误不做重试。

**核心策略**: 用 try/except 兜底 + 重试机制（最多 3 次）提高成功率。

---

## Function Calling

### 核心概念

### Function Calling 的本质

Function Calling 是 LLM **决定调用外部工具**的机制。模型在整个过程中只是一个"决策者"，它只负责决定调什么工具、传什么参数。**执行工具是客户端代码的事**。

> 模型不会执行工具。模型只"决定"调用什么工具，不"执行"工具。执行是你客户端代码的事。

### 类比理解

| 角色 | 类比 | 实际对应 |
|------|------|----------|
| 指挥官 | 说"去查北京的天气" | LLM 模型 |
| 士兵 | 真正去查天气，查完回来报告 | 你的客户端代码 |
| 菜单 | 告诉厨师有哪些菜、配料是什么 | Tool 定义（JSON Schema） |

---

## 完整流程

```
用户提问 → 模型分析意图 → 决定调用工具 → 客户端执行 → 结果回传 → 模型生成回复
```

### 详细说明

```
1. 定义工具：你把 tools（JSON Schema）发给模型，告诉它有哪些工具可用
2. 模型决策：模型分析用户问题，返回 "我想调 get_weather(city='北京')"
3. ★ 执行工具：你的客户端代码收到决定，调用真正的 get_weather 函数
4. 回传结果：你的代码把执行结果作为新消息发回给模型
5. 模型回复：模型基于你传回的结果，生成最终文字回复给用户
```

### 一句话总结

> **你负责跑腿，模型负责决策，循环直到模型觉得够了为止。**

---

## Tool 定义的作用

Tool 定义本质上是一份**JSON Schema 说明书**，告诉模型：

- **name**: 工具名称（模型用来识别和调用）
- **description**: 工具功能描述（模型做语义匹配的主要依据）
- **parameters**: 参数定义（类型、必填项、描述）

**它不包含任何真正的逻辑代码**，只是让模型知道"有什么工具、需要什么参数"。

---

## 模型如何决定何时调用工具

当你在 API 调用中传入了 `tools` 参数时，模型会：

1. 拿用户问题和每个 tool 的 `description` + `parameters` 做**语义匹配**
2. 如果匹配到某个工具 → 返回 `tool_calls`
3. 如果不匹配任何工具 → 直接文字回复

> 模型不是在做关键词匹配，而是做语义理解。

---

## 多轮调用（连续 Function Calling）

不是固定 2 次对话，而是**循环直到模型不需要再调工具**：

| 用户问题 | 轮次 | 每轮发生了什么 |
|----------|------|----------------|
| "你好" | 1 轮 | 模型直接文字回复 |
| "北京天气怎么样？" | 2 轮 | 查天气 → 回复 |
| "北京温度×上海温度÷3" | 3 轮 | 查两个城市天气 → 计算 → 回复 |

### 消息历史示例（3 轮场景）

```json
[
  {"role": "user",    "content": "北京温度×上海温度÷3"},
  {"role": "assistant", "tool_calls": [get_weather("北京"), get_weather("上海")]},
  {"role": "tool",     "content": "北京 25°C"},
  {"role": "tool",     "content": "上海 22°C"},
  {"role": "assistant", "tool_calls": [calculate("25*22/3")]},
  {"role": "tool",     "content": "183.33"}
]
```

模型每次调用都看到完整的消息历史，所以它知道之前的工具结果是怎么来的。

---

## 命名规范与工具选择

### name 和 description 的权重

| 字段 | 权重 | 作用 |
|------|------|------|
| `description` | **主要** | 用自然语言描述功能，LLM 做语义匹配 |
| `name` | **辅助** | LLM 会从中提取语义线索 |

### 多个相似工具怎么办？

如果多个工具 description 完全相同，模型会靠 name 的语义做选择（不是完全随机），但效果可能变慢、变差。

**最佳实践**：
- 工具名要表意
- 描述要有区分度
- 不要让多个工具的 description 重复

### function_map 的对接

`tools` 里的 `name` 字段必须和 `function_map` 的 key 一致。它们是两个世界之间的"握手协议"：

```python
# tools 定义（发给 LLM）
tools = [{"type": "function", "function": {"name": "get_weather", ...}}]

# function_map（本地代码）
function_map = {
    "get_weather": get_weather  # key 必须和 tools 里的 name 一致
}
```

---

## 关键代码模式

### while 循环处理多轮调用

```python
messages = [{"role": "user", "content": user_input}]

while True:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
        temperature=0
    )
    assistant_msg = response.choices[0].message

    if assistant_msg.tool_calls:
        messages.append(assistant_msg)
        for tc in assistant_msg.tool_calls:
            func = function_map[tc.function.name]
            result = func(**json.loads(tc.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
    else:
        # 模型直接文字回复，跳出循环
        print(assistant_msg.content)
        break
```

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| "模型执行了工具" | ❌ 模型只决定调用，执行是你的客户端代码 |
| "固定 2 轮对话" | ❌ 轮次取决于问题复杂度，可能是 1~N 轮 |
| "模型第二次只看到工具结果" | ❌ 模型看到完整的消息历史 |
| "description 一样也没关系" | ⚠️ 能用但效果打折扣，模型会靠 name 做偏好选择 |

---

## 多工具路由（Step 5）

### 实验：4 工具路由正确率测试

**工具列表**：`get_weather`、`calculate`、`search_product`、`convert_currency`

**10 个测试用例结果**：

| # | 问题 | 期望工具 | 实际工具 | 结果 |
|---|------|----------|----------|------|
| 1 | 北京天气怎么样？ | get_weather | get_weather | ✅ |
| 2 | 100美元等于多少人民币？ | convert_currency | convert_currency | ✅ |
| 3 | iPhone 现在卖多少钱？ | search_product | search_product | ✅ |
| 4 | 256 除以 8 等于多少？ | calculate | calculate | ✅ |
| 5 | 1000欧元能换多少日元？ | convert_currency | convert_currency | ✅ |
| 6 | 北京和上海温度差多少？ | get_weather×2 + calculate | get_weather×2 | ❌ |
| 7 | 你好请介绍自己 | 不调 | 不调 | ✅ |
| 8 | 今天星期几？ | 不调 | 不调 | ✅ |
| 9 | 写一首关于春天的诗 | 不调 | 不调 | ✅ |
| 10 | 北京到上海高铁票多少钱？ | search_product | search_product | ✅ |

**正确率**: 9/10 = 90%（Milestone 要求 >= 80%）

**唯一错误 case**: "北京和上海温度差多少" —— 模型查了两个天气后**自己心算减法**，没有调 `calculate`。和之前 Step 4 实验一致，模型认为简单加减法不需要调工具。这不是 bug，而是模型的自主判断。

**关键发现**:
- 模型对"不需要工具"的判断很准确（场景 7、8、9 全对）
- `description` 区分度很重要，比如 `convert_currency` 强调"不涉及商品价格查询"，避免和 `search_product` 混淆
- 模糊意图场景（"高铁票多少钱"）模型能正确处理，调了 `search_product`

---

## Step 6：Token 计量与成本意识

### Token 指标含义

| 指标 | 含义 | 对应什么 |
|---|---|---|
| **prompt_tokens** | 你发给模型的 token 数 | messages + tools 定义 |
| **completion_tokens** | 模型返回的 token 数 | AI 的文字回复 |
| **total_tokens** | 两者之和 | 本次调用总消耗 |

> `total_tokens = prompt_tokens + completion_tokens`

### qwen-plus 价格（2025-2026 官方定价）

[来源](https://help.aliyun.com/zh/model-studio/model-pricing)

| 方向 | 单价 |
|---|---|
| 输入（prompt） | ¥0.0008 / 千Tokens（¥0.8/百万） |
| 输出（completion） | ¥0.002 / 千Tokens（¥2.0/百万） |

**输出比输入贵 2.5 倍**，这是行业通例。

### 真实数据费用计算

| 场景 | prompt | completion | 单次费用 |
|---|---|---|---|
| 北京天气 | 318 | 20 | ¥0.000294 |
| 美元换算 | 329 | 21 | ¥0.000305 |
| 高铁票查询 | 318 | 20 | ¥0.000294 |

计算公式：`输入费用 = prompt_tokens / 1000 × 0.0008`，`输出费用 = completion_tokens / 1000 × 0.002`

### 规模化成本估算

| 场景 | 计算 | 月费用 |
|---|---|---|
| 1000 用户，每人每天 20 轮 | 1000 × 20 × 30 × ¥0.000294 | **¥176.4/月** |
| 10 万用户，同上 | 100 × 上面的费用 | **¥17,640/月** |
| 10 万用户 + 每次平均 prompt=5000 | 输入费用大幅上升 | **¥800+/天** |

### 节省 Token 策略

| 策略 | 效果 |
|---|---|
| 精简 system prompt | 减少 prompt_tokens（占大头） |
| 控制 max_tokens | 限制 output 上限 |
| 用便宜的模型做初筛 | 如 qwen-turbo（输入 ¥0.0003/千） |
| 上下文缓存 | 重复的历史消息缓存后打折 |

---

## 阶段验收：CLI 智能助手

### 项目概述

实现一个命令行助手，支持多轮对话、工具路由、Token 计量。代码在 `assistant.py`。

### 架构设计

```
assistant.py
├── 工具定义区
│   ├── 工具函数（get_weather, calculate, search_product, convert_currency）
│   ├── TOOLS（JSON Schema，发给 LLM）
│   └── FUNCTION_MAP（本地派发表）
├── Assistant 类
│   ├── __init__: 初始化 client + system prompt + 消息历史
│   ├── _execute_tools(): 执行工具调用
│   ├── _print_usage(): 打印 token 消耗和费用
│   └── chat(): 核心 while 循环，处理多轮 API 调用
└── CLI 入口
    ├── 交互循环（输入 → chat → 输出）
    ├── 特殊命令（quit, clear）
    └── 会话总结
```

### 验收结果

| 验收项 | 状态 |
|---|---|
| 运行 `python assistant.py` 启动不报错 | ✅ |
| 需要工具的问题，正确调用并回复 | ✅（天气、计算、搜索、汇率） |
| 普通问题，直接回答不调工具 | ✅（打招呼、写诗） |
| 连续多轮对话，维持上下文 | ✅（3 轮测试通过） |
| 工具定义与执行逻辑分离 | ✅（TOOLS/FUNCTION_MAP vs Assistant 类） |
| 没有硬编码 API key | ✅（使用 .env） |
| 每次调用打印 token 消耗和费用 | ✅ |

---

## 待深入：开发者控制工具路由（Phase 4 占位）

> **问题**: 所有工具选择都交给 LLM 自己做，如果是我本地的敏感/内部工具，能否由开发者自己控制路由逻辑？

**Phase 1 当前状态**: 全交给 LLM 通过 `tools` 参数做路由决策（软控制，靠 description 引导）。

**Phase 4 将学习**:
- [[Phase4 学习笔记#手写 ReAct Agent]] — 不用框架，自己用代码控制 Thought → Action → Observation 循环
- [[Phase4 学习笔记#Agent 的错误处理与健壮性]] — 处理 LLM 选错工具、返回不存在工具名等异常情况
- 三种控制模式对比：Prompt 引导 vs 代码硬路由 vs 混合模式

---

## 参考代码

- `phase1_step4_function_calling.py` — Function Calling 基础实现（5 个测试场景）
- `phase1_step5_multi_tool_routing.py` — 多工具路由测试（4 工具，10 用例，90% 正确率）
