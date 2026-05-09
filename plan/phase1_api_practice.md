---
name: 第一阶段 - API 实操
description: 打破 LLM API 零经验状态，掌握基本的消息格式、对话管理和 Function Calling
type: project
---

# 第一阶段：LLM API 实操

> 预计耗时：1-2周 | 开始日期：____ | 完成日期：____
> 关联画像：ai_learning_profile.md

---

## 前置准备

### Step 0：环境搭建

**学习目标**：获取 API key，配置 Python 开发环境

**操作建议**：
1. 注册 Claude API（Anthropic Console）或 OpenAI API（platform.openai.com）
   - 推荐先选 OpenAI，文档和生态更成熟
   - 如果注册有障碍，用 Claude API 也可以
2. 创建虚拟环境：`python -m venv venv`
3. 安装 SDK：`pip install openai` 或 `pip install anthropic`
4. 将 API key 存入环境变量（不要硬编码到代码中）
   - Windows PowerShell：`$env:OPENAI_API_KEY="your-api-key-here"`
   - 或写入 `.env` 文件，用 `python-dotenv` 加载

**Milestone 0**：
- [x] 成功注册账号，获取 API key
- [x] 虚拟环境创建成功，SDK 安装无报错
- [x] `python -c "import openai; print(openai.__version__)"` 能正常输出

---

## Step 1：第一次 API 调用

**学习目标**：用最简单的方式调通一次 LLM，理解 request-response 流程

**操作建议**：
1. 写一个最简单的脚本，发送一条消息并打印回复
2. 理解三个基本角色：`system`、`user`、`assistant`
3. 理解关键参数：`model`、`temperature`、`max_tokens`

**Milestone 1**：
- [x] 脚本成功运行，收到非空回复
- [x] 修改 `temperature` 为 0 和 1.5 各跑一次，能观察到输出风格差异
- [x] 修改 `system` prompt，观察对输出风格的影响
- [x] 能口头解释 `system` / `user` / `assistant` 三个角色的区别

---

## Step 2：多轮对话与上下文管理

**学习目标**：理解 LLM 本身是无状态的，多轮对话需要客户端维护消息历史

**操作建议**：
1. 实现一个简单的交互式命令行聊天程序
2. 每轮对话后，把用户消息和模型回复都追加到 `messages` 列表中
3. 尝试在第三轮对话中引用第一轮的内容，观察模型是否能"记住"
4. 了解上下文窗口限制（如 GPT-4o 是 128K tokens），理解"上下文满了会怎样"

```python
messages = [
    {"role": "system", "content": "你是助手。"}
]

while True:
    user_input = input("你: ")
    if user_input == "quit":
        break
    messages.append({"role": "user", "content": user_input})
    # 调用 API，把完整 messages 传进去
    # 把回复 append 到 messages
    # 打印回复
```

**Milestone 2**：
- [x] 命令行程序能连续对话 10 轮以上不中断
- [x] 在第 8 轮提到第 1 轮说过的内容，模型能正确引用
- [x] 能口头解释：为什么 LLM 需要每次传完整消息历史？
- [x] 能说出你所用模型的上下文窗口大小

---

## Step 3：Structured Output（结构化输出）

**学习目标**：让 LLM 输出 JSON 等机器可读格式，这是后续 Function Calling 和 Agent 的基础

**操作建议**：
1. 用 prompt 引导模型输出 JSON：`"请用 JSON 格式回答，包含 name, age, skills 三个字段"`
2. 尝试让模型输出不同结构的 JSON（列表、嵌套对象）
3. 用 `json.loads()` 解析模型输出，验证是否是合法 JSON
4. 学习 OpenAI 的 `response_format` 参数（JSON Schema mode）或 Anthropic 的对应能力

```python
# 测试场景：把一段自然语言描述转成结构化数据
messages = [
    {"role": "user", "content": """
    提取以下信息为 JSON：张三，28岁，擅长Python和Java，有5年经验。
    格式要求：{"name": "", "age": 0, "skills": [], "experience_years": 0}
    只输出 JSON，不要其他文字。
    """}
]
```

**Milestone 3**：
- [x] 连续 5 次调用，JSON 解析成功率 >= 80%
- [x] 能处理模型偶尔输出格式错误的情况（加 try/except）
- [x] 了解 JSON Schema 约束输出的基本用法

---

## Step 4：Function Calling / Tool Use

**学习目标**：理解 LLM 如何通过 Function Calling 调用外部工具——这是 Agent 的核心机制

**操作建议**：
1. 学习 Function Calling 的标准流程：
   - 定义工具的 JSON Schema（工具名、描述、参数）
   - 在 API 调用中传入 `tools` 参数
   - 模型返回 `tool_calls`（注意：模型不执行工具，只**决定调用哪个工具、传什么参数**）
   - 客户端执行工具，把结果作为新消息传回模型
   - 模型基于工具结果生成最终回复
2. 实现 2 个简单工具：
   - `get_weather(city)`：返回模拟天气数据（不需要真调天气 API）
   - `calculate(expression)`：用 Python `eval()` 做简单计算
3. 完整跑通一次：用户问"北京明天天气怎样？"→ 模型决定调 get_weather → 你执行 → 模型基于结果回复

```python
# 工具定义示例（OpenAI）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]
```

**Milestone 4**：
- [x] 用户问天气 → 模型正确返回 `tool_calls`，包含 `get_weather` 和 `city="北京"`
- [x] 客户端执行工具，把结果传回，模型基于结果生成最终回复
- [x] 用户问一个不需要工具的问题（如"你好"）→ 模型不调 tool 直接回复
- [x] 能口头解释 Function Calling 的完整流程（定义 → 调用 → 执行 → 回传 → 回复）

---

## Step 5：多工具路由与条件判断

**学习目标**：让模型在多个工具中做选择，并根据工具结果做逻辑判断

**操作建议**：
1. 在 Step 4 基础上增加工具：
   - `search_product(keyword)`：返回模拟商品搜索结果
   - `convert_currency(amount, from, to)`：返回模拟汇率换算
2. 测试模型能否根据用户意图选择正确的工具：
   - "北京到上海的高铁票多少钱？" → convert_currency 不合适，应该是 search_product 或没有匹配工具
   - "100美元等于多少人民币？" → convert_currency
3. 观察模型在模糊意图下的行为，尝试用更好的 tool description 来改善

**Milestone 5**：
- [x] 定义 4+ 个工具，模型选择正确率 >= 80%（跑 10 个测试用例）
- [x] 当用户意图不匹配任何工具时，模型能直接回复而非强行调用
- [x] 记录每次选择错误的 case，分析是 tool description 问题还是模型能力问题

---

## Step 6：Token 计量与成本意识

**学习目标**：理解 token 概念、计费方式，建立成本意识

**操作建议**：
1. 查看 API response 中的 `usage` 字段：`prompt_tokens`、`completion_tokens`、`total_tokens`
2. 了解不同模型的单价（查官方 pricing 页面）
3. 估算：如果你有一个 1000 用户的 AI 产品，每人每天聊 20 轮，月成本多少？
4. 了解节省 token 的策略：精简 system prompt、控制 max_tokens、用便宜的模型做初筛

**Milestone 6**：
- [x] 能解释 token 和 word 的大致换算关系（英文约 1 token ≈ 0.75 词）
- [x] 能算出一次调用的费用
- [x] 知道你所用模型的价格（每 M tokens 多少钱）

---

## 阶段验收

完成以上所有 Milestone 后，做一个综合小项目作为最终验收：

### 综合项目：CLI 智能助手

实现一个命令行助手，要求：
1. 支持多轮对话（维护上下文）
2. 内置 3+ 个工具（至少包含一个计算类、一个查询类）
3. 模型能正确选择工具或决定不调工具
4. 工具结果正确传回并用于最终回复
5. 输出格式为 JSON 时能被正确解析
6. 打印每次调用的 token 消耗

**验收标准**：
- [ ] 运行 `python assistant.py` 启动后不报错
- [ ] 问需要工具的问题，能正确调用并给出合理回复
- [ ] 问普通问题，能直接回答
- [ ] 连续对话 15 轮，能记住早期关键信息
- [ ] 代码结构清晰，工具定义与执行逻辑分离
- [ ] 没有硬编码 API key

---

## 常见陷阱

- **不要在 Step 1 就追求完美**：先跑通，再优化
- **API key 泄露**：绝对不要提交到 git，用 `.env` + `.gitignore`
- **Function Calling 的误区**：模型只"决定"调用什么工具，不"执行"工具。执行是你客户端代码的事
- **温度参数**：调试 Function Calling 时用 `temperature=0`，减少随机性
