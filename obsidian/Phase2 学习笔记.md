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

## Step 2：Few-Shot Prompting（少样本提示）

### 实验：Zero-Shot vs Few-Shot 意图分类准确率对比

代码在 `phase2_step2_few_shot.py`。

任务：将用户输入分类为 query / command / chitchat / creative。

#### 第一轮：10 个简单 case

| 方式 | 正确率 |
|------|--------|
| Zero-Shot | 100% (10/10) |
| 1-Shot | 100% (10/10) |
| 3-Shot | 100% (10/10) |
| 5-Shot | 100% (10/10) |

结论：任务太简单，qwen-plus 本身就能区分，few-shot 没有提升空间。

#### 第二轮：20 个 case（前 10 个简单，后 10 个模糊/边缘）

| 方式 | 正确率 |
|------|--------|
| Zero-Shot | **95% (19/20)** |
| 1-Shot | 90% (18/20) |
| 3-Shot | 85% (17/20) |
| 5-Shot | 85% (17/20) |

**Few-shot 反而比 zero-shot 差**。

#### 3 个共同错误 case 分析

| 输入 | 期望 | Zero-Shot 结果 | Few-Shot 结果 | 原因 |
|------|------|----------------|---------------|------|
| 用 Python 写一个快速排序 | command | creative | creative | 类别边界模糊（写代码算 command 还是 creative？） |
| 你觉得我应该学 Go 还是 Rust | chitchat | chitchat ✅ | query ❌ | zero-shot 判断正确，few-shot 反而带偏 |
| 帮我分析一下这个日志 | command | command ✅ | query ❌ | zero-shot 判断正确，few-shot 反而带偏 |

#### 关键发现

- **过度依赖 few-shot**：有时候一个更好的 task description 比 5 个示例更有效
- Few-shot 的示例如果不够覆盖边缘场景，反而会"带偏"模型
- 示例越多，占用 context 空间也越多，增加 token 成本
- 这与 [[Phase2 学习笔记#Prompt 的基本结构|Prompt 的核心要素]] 中提到的"示例 (few-shot)"是提升一致性的手段，但不是万能药——任务本身简单时不需要

### Few-Shot 的实际使用策略

**实际工作中的做法**：

```
zero-shot 跑 3-5 个 case
  ├── 全对 → 够了，不加 few-shot（省 token）
  └── 有错 → 加 few-shot 再跑
        ├── 改善了 → 保留 few-shot
        └── 更差了 → 回退，优化 task description
```

**不同场景下 few-shot 的有效性**：

| 场景 | 谁更有效 | 原因 |
|------|----------|------|
| 类别边界清晰、任务简单 | Task Description | 模型不需要示例就能理解 |
| 类别边界模糊、需要模仿格式 | Few-Shot | 光靠描述说不清"长什么样" |
| 需要控制输出风格/语气 | Few-Shot | "请用专业语气"不如给一个专业示例 |
| 模型对某类任务能力较弱 | Few-Shot | 示例相当于"教它怎么做" |

**开发阶段的实践**：上线前跑 20-50 个 case 做验收，但开发阶段就是 3-5 个 case 快速试，不用一开始就跑大量 case。

### Few-Shot 的本质定位

- 模型本身不会的任务 → few-shot 是**雪中送炭**
- 模型已经会的任务 → few-shot 是锦上添花（甚至可能添乱，如本次实验）

### Few-Shot 示例与 Task Description 冲突时的优先级

| 冲突类型 | 谁赢 | 原因 |
|----------|------|------|
| 描述说 A，示例全是 B | 示例赢 | 示例数量多，模式更明显，模型优先模仿 |
| 描述说 A，1 个示例是 B | 看描述强度 | 描述很明确时可能还是听描述的 |
| 描述模糊，示例清晰 | 示例赢 | 模型需要参考，描述没给方向 |

**原则**：描述定方向，示例定细节。示例的权重通常比描述高，因为它更具体。

---

## Step 3：Chain-of-Thought (CoT，思维链)

### 实验：Direct vs Zero-shot CoT vs Few-shot CoT

代码在 `phase2_step3_cot.py`。

任务：5 道需要多步推理的数学/逻辑题，对比直接回答和 CoT 方式的准确率。

#### 5 道测试题

1. 鸡兔同笼（35 头、94 脚）
2. 年龄差问题（小明 8 岁，爸爸是 4 倍，差多少岁）
3. 水池进出管问题（进水 6 小时注满，排水 8 小时排空）
4. 相向而行问题（甲 5km/h，乙 4km/h，距中点 3km 相遇）
5. 3 开关 3 灯泡逻辑谜题（只能去一次房间，如何对应）

#### 实验结果

| 方式 | 正确率 | 总 Token | 单题平均 Token |
|------|--------|----------|----------------|
| Direct（直接回答） | **80% (4/5)** | 560 | 112 |
| Zero-shot CoT | 60% (3/5) | 4,298 | 859 |
| Few-shot CoT | 60% (3/5) | 4,766 | 953 |

**CoT 反而降低了正确率，token 消耗翻了 7-8 倍。**

#### 逐题分析

| 题号 | Direct | Zero-shot CoT | Few-shot CoT | 说明 |
|------|--------|---------------|--------------|------|
| 1 鸡兔同笼 | 答对 | 答对 | 答对 | 三种方式都算出了正确答案 |
| 2 年龄差 | 答对 | 答对 | 答对 | 三方式全对 |
| 3 进出水管 | 答对 | 答对 | 答对 | 三方式全对 |
| 4 相向而行 | **答对** | ❌ 跑偏 | ❌ 跑偏 | Direct 直接给 54km，CoT 推理过程反而出错 |
| 5 开关灯泡 | 答对 | 答对 | 答对 | 三方式全对 |

#### 关键发现

- **对于 qwen-plus 来说，简单数学题不需要 CoT**——模型本身已经训练过这类题目，直接回答反而又快又准
- **CoT 的推理过程可能引入新错误**：第 4 题 Direct 模式直接给正确答案，CoT 模式一步步推导反而走偏了
- **CoT 的代价**：completion tokens 平均增长 7-8 倍，响应也更慢
- 这与 Step 2 的 few-shot 实验结论一致：**不是所有任务都需要额外的引导，模型本身已经会的任务，加引导反而可能添乱**

### CoT 什么时候有效？

| 场景 | CoT 是否有效 | 原因 |
|------|-------------|------|
| 简单数学题（鸡兔同笼、年龄差） | 无效甚至有害 | 模型已经训练过，直接回答更准 |
| 复杂多步推理题（需要中间变量转换） | 有效 | 给了模型更多思考空间 |
| 逻辑推理谜题 | 有效 | 需要逐步排除可能性 |
| 新知识/训练数据中较少的问题 | 有效 | 模型需要从头推导 |

### Zero-shot CoT 的触发方式

- `"请一步步思考，先列出已知条件，再推导"`
- `"Let's think step by step"`（英文原版效果最好）

### Few-shot CoT vs Zero-shot CoT

- Zero-shot CoT：只加一句引导语，简单但效果有限
- Few-shot CoT：示例中包含完整的推理步骤，效果更好但 token 消耗更大
- 本实验中两者准确率相同（3/5），Few-shot 更贵

### CoT 的本质机制

CoT 起作用的核心原因不是"教模型怎么想"，而是**改变了模型输出的 token 分布**。

**自回归（Autoregressive）特性**：模型生成的每个 token 都会成为下一轮生成的输入。

```
Direct 模式:
  输入 -> 模型直接跳到结论 -> 容易跳跃出错

CoT 模式:
  输入 -> 输出 "步骤1" -> "步骤1" 成为新输入
         -> 输出 "步骤2" -> "步骤1+步骤2" 成为新输入
         -> ...
         -> 中间步骤变成了"草稿纸"，降低跳跃错误
```

**关键理解**：不是"模型心里知道但忘了说"，而是**不写出来，模型真的可能想不到**。写出来后，前一步的结果就在上下文中，下一步的概率分布就完全变了。

**两层好处**：
1. 模型推理更准确（核心原因：中间步骤变成新输入）
2. 用户更容易纠错（调试便利：能看到错在哪一步）

### CoT 的通俗类比

| 模式 | 类比 | 特点 |
|------|------|------|
| Direct | 让模型"心算" | 快、省 token，但容易跳步出错 |
| CoT | 给一张草稿纸让模型"写算式" | 每一步都成为新提示，减少跳跃错误 |

草稿纸上写的每一步都算 token——这就是 CoT 的代价。

### CoT 适用条件的修正

原 Milestone 3 表述"CoT 方式正确率更高"不够准确。实验结果显示：
- **简单任务**（模型已掌握）→ Direct 模式更高效（更快、更省 token、准确率相同或更高）
- **复杂任务**（超出模型训练分布）→ CoT 才体现优势

**修正后的理解**：CoT 不是万能加速器，而是针对复杂推理问题的"减速带"——慢下来一步步想，反而不容易错。

---

