# Phase 2 学习笔记

> **日期**: 2026-05-11
> **前置知识**: Phase 1（API 实操、Structured Output、Function Calling）
> **关联文档**: `plan/phase2_prompt_engineering.md`

---

## Step 1：Prompt 的基本结构

### 对比实验：3 个质量等级的 prompt

测试任务：同一段有 SQL 注入漏洞的代码，用 3 个不同 prompt 让模型审查。

**Level 1 — 只有问题** (`"这段代码有什么问题？" + code`)
- Prompt 要素：仅任务描述 + 上下文
- 输出：Markdown 长文，结构松散
- Token 消耗：prompt=68, completion=1033, total=1101
- 费用：¥0.00212
- 可程序化：不可解析

**Level 2 — 角色 + 任务** (`"你是有经验的 Python 工程师。请分析..."`)
- Prompt 要素：角色 + 任务 + 上下文
- 输出：Markdown 长文，比 Level 1 更有针对性但仍然无结构约束
- Token 消耗：prompt=82, completion=1433, total=1515
- 费用：¥0.00293
- 可程序化：不可解析

**Level 3 — 完整结构** (角色 + 任务 + 上下文 + 输出格式 + few-shot 示例)
- Prompt 要素：5 个要素齐全
- 输出：结构化 JSON，一次解析成功
- Token 消耗：prompt=352, completion=174, total=526
- 费用：¥0.00063
- 可程序化：JSON 可解析

**关键发现**：
- Level 3 虽然 prompt 长了 3 倍，但总体 token 消耗反而最低（526 vs 1101/1515），因为输出被格式约束得很短
- Level 1/2 的输出无法被下游程序直接消费
- 有 few-shot 示例时，JSON 格式一次成功

---

### Prompt 的 5 个核心要素拆解

| 要素 | 作用 | 示例 |
|------|------|------|
| 角色/人设 | 设定 AI 的身份和行为边界 | "你是一个资深 Python 工程师" |
| 任务描述 | 明确要做什么 | "分析以下代码的潜在 bug" |
| 上下文/背景 | 提供必要信息 | 代码片段、数据、约束条件 |
| 输出格式要求 | 控制输出结构 | "用 JSON 返回，包含字段 X、Y、Z" |
| 示例 (few-shot) | 让模型模仿模式，提升一致性 | 给 1-2 个输入输出样例 |

---

### Prompt 要素的存放位置讨论

**核心原则：全局放"不变的部分"，运行时组"变化的部分"。**

| Prompt 要素 | 放哪 | 原因 |
|-------------|------|------|
| 角色/人设 | 全局配置 | 固定不变，所有请求都生效 |
| 行为规则/边界 | 全局配置 | 同上 |
| 输出格式要求 | 看情况 | 全局统一的放配置，按场景变的放代码 |
| 任务描述 + 上下文 | 运行时组装 | 每次请求不同 |
| 示例 (few-shot) | 模板/配置 | 按任务选，不是一成不变 |

**不同项目类型的全局 prompt 存放方式**：

| 项目类型 | 全局 prompt 放哪 | 运行时 prompt 在哪 |
|----------|-----------------|-------------------|
| Claude Code / AI CLI | CLAUDE.md / settings.json | 用户输入 + 代码上下文 |
| AI Agent 应用 | config.yaml / .env 里的 SYSTEM_PROMPT | 根据用户输入动态组装 task + context |
| Web 应用 | 数据库/配置文件存 system prompt | 后端代码里拼完整 prompt |
| API 服务 | 环境变量或配置中心 | Handler 里组装 messages |

**以 Claude Code 项目为例**：
- CLAUDE.md 里放的是角色定义 + 行为规则
- 每次请求时，Claude Code 会：`system prompt (底层) + CLAUDE.md 内容（注入为紧挨 system 的用户消息）+ 用户的实际请求 = 完整的 prompt`
- CLAUDE.md 只是完整 prompt 中"角色 + 规则"那一层

**few-shot 不一定适合全局**：比如代码审查，审 Python 和审前端可能需要不同的示例，按场景动态选更合适。

---

### System vs User 中的角色定义对比实验

测试了 system 消息和 user 消息中角色定义的区别、冲突时的优先级。

**实验 1：角色放 system vs user**

| 场景 | 配置 | 结果 |
|------|------|------|
| A | 角色在 system，user 只有问题 | 结构化点评，直接列问题 |
| B | 角色在 user（同一消息里） | 也列了问题，但更啰嗦 |
| C | system 为空，user 只有问题 | 更中立，先夸再说问题 |

结论：角色放 system vs user 都能生效，但 system 的作用域是对所有后续消息，user 里的只在那条消息有效。

**实验 2：system 和 user 角色冲突**

| 场景 | system 指令 | user 指令 | 结果 |
|------|-------------|-----------|------|
| D | 友善助手 | 毒舌批评家 | 听了 user 的（毒舌），但开头有软化 |
| E | 毒舌批评家 | 友善鼓励 | 完全听了 user 的（友善） |
| F | 友善 | 友善 | 正常友善 |

**结论：在 qwen-plus 上，user 消息优先级高于 system。当前指令覆盖背景设定。**

**实验 3：system 规则能否被 user 覆盖**

| 场景 | system 规则 | user 要求 | 结果 |
|------|-------------|-----------|------|
| G | 严禁输出代码 | 请写一个素数函数 | system 赢了：只给文字描述，没给代码 |
| H | 只能用中文 | 请用英文回答 | user 赢了：直接用英文回答 |
| I（对照） | 无约束 | 请用英文回答 | 正常英文 |

结论：system 约束不是绝对的，模型会根据 user 消息的强度做权衡。强规则（禁止输出代码）比弱规则（用中文）更可能被遵守。

**核心总结**：
- `system` 作用是"设定基调"，不是"强制命令"
- 底层 system 和 user 都是消息列表里的 token，模型整体理解
- 冲突时通常 user 优先级更高（当前指令覆盖背景设定）
- **system prompt 是配置不是运行时逻辑**：初始化时加载一次，后面每次请求复用，不用每次都拼
- 不同复杂度对应不同存储方式：硬编码字符串 → 配置文件 → 模板文件 → 数据库（多用户多角色 SaaS）

---

### System Prompt 的工程化管理（补充）

代码在 `phase2_step1c_prompt_templates.py`。

**方式 1 — 硬编码字符串**
```python
SYSTEM_PROMPT = "你是一个资深游戏玩家..."
# 每次调用直接复用
```
适合快速 demo，改人设要改代码。

**方式 2 — 从文件加载**
```python
system_prompt = load_prompt_file("system_game_guide.txt")
```
改人设只改 `.txt` 文件，不动代码。适合固定人设的小工具。

**方式 3 — 模板变量 + Agent 类**
```python
class GameAgent:
    def __init__(self, game_name, character_profile):
        self.system_prompt = template.format(
            game_name=game_name, role=character_profile
        )
```
初始化一次，后续复用。支持多游戏、多角色。可扩展 context 注入（游戏状态、背包信息等）。

**实际效果对比**（同一问题"新手开局选什么职业"）：

| Agent | 回答风格 |
|-------|----------|
| 艾尔登法环 — 硬核攻略作者 | 推荐武士/流浪骑士，列出了具体武器、战灰、后期路线 |
| 星露谷物语 — 休闲种田向导 | 推荐农夫/牧场主，语气轻松，提到前7天规划 |

模板变量直接决定了 Agent 的知识领域和语言风格。

---

### Tool 定义也是 Prompt 的一种形式

`tools` 参数从 API 层面看是独立的参数，但从模型视角：

```
实际发送 = system 消息 + tools 定义的 JSON + user 消息
```

SDK 把 `tools` 序列化成 token 注入到 prompt 中，本质上是 SDK 帮你封装好的**结构化 prompt 模板**。

| 手动拼 prompt | 用 tools 参数 |
|---------------|--------------|
| 自己写"你有以下工具：get_weather(city)..." | SDK 自动拼接 JSON |
| 自己解析模型的 JSON 回复 | SDK 自动解析 tool_calls |
| 格式不标准时模型可能不理解 | 标准格式，模型训练时见过 |

**结论**：tool definition 和 prompt design 是同一件事的两种格式——一个结构化，一个自由文本。

---

### Milestone 1 完成进度

| 要求 | 状态 |
|------|------|
| 对同一个任务写 3 个不同质量等级的 prompt，输出质量有明显差异 | ✅ 完成（SQL 注入代码审查，Level 1/2/3 对比） |
| 能列出 prompt 的 5 个核心要素并逐一解释作用 | ✅ 完成（角色定义 Role、任务描述 Task Description、输出格式 Output Format、上下文背景 Context、示例 Few-shot） |
| 总结出自己在实际开发中最常用的 3 种 prompt 模板 | ✅ 完成（结构化输出、工具路由、多轮对话） |

---

### 常用 Prompt 模板总结

基于 Phase 1 实践，最常用的 3 种 prompt 模板：

**模板 1：结构化输出（Structured Output）** — `phase1_step3` / `phase1_step6`
```python
messages = [
    {"role": "system", "content": "你是一个数据提取助手。"},
    {"role": "user", "content": f"""
请从以下文本中提取信息，用 JSON 格式输出：
{{"name": "", "age": 0, "skills": []}}
只输出 JSON，不要其他文字。
文本：{raw_text}
"""}
]
```
适用场景：信息提取、数据转换、报告生成。

**模板 2：工具路由（Tool Routing）** — `phase1_step4` / `phase1_step5` / `assistant.py`
```python
messages = [
    {"role": "system", "content": "你是一个智能助手，可以根据用户需求选择合适的工具。"},
    {"role": "user", "content": user_input},
]
# tools 定义另传，包含 name / description / parameters
```
适用场景：CLI 助手、智能客服、任何需要模型决定调哪个工具的场景。

**模板 3：多轮对话（Multi-turn Conversation）** — `phase1_step2` / `assistant.py`
```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  # 人设 + 规则
]
# 每轮追加：
messages.append({"role": "user", "content": user_input})
messages.append({"role": "assistant", "content": reply})
```
适用场景：聊天机器人、客服、Code Review 迭代对话。

---

