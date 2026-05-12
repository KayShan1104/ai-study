# AI Learning Lab

通过手写代码 + 实验的方式，从零理解 LLM API、Prompt Engineering、RAG、Agent 等核心技术。

## 学习路径

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | LLM API 实操 | 已完成 |
| Phase 2 | Prompt Engineering | 已完成 |
| Phase 3 | RAG（检索增强生成） | 待开始 |
| Phase 4 | Agent 框架 | 待开始 |
| Phase 5 | 探索方向 | 待开始 |
| Phase 6 | Agent 测试与评估 | 待开始 |

## 目录结构

```
.
├── plan/                          # 学习计划文档
│   ├── ai_learning_profile.md
│   ├── phase1_api_practice.md
│   ├── phase2_prompt_engineering.md
│   ├── phase3_rag.md
│   ├── phase4_agent_framework.md
│   ├── phase5_exploration.md
│   └── phase6_agent_testing_evaluation.md
├── code/                          # 实践代码
│   ├── phase1/
│   │   ├── phase1_step1_api_call.py
│   │   ├── phase1_step2_multiconversation.py
│   │   ├── phase1_step3_structured_output.py
│   │   ├── phase1_step4_function_calling.py
│   │   ├── phase1_step5_multi_tool_routing.py
│   │   └── assistant.py           # 阶段验收：CLI 智能助手
│   └── phase2/
│       ├── phase2_step1_prompt_structure.py
│       ├── phase2_step1b_system_vs_user.py
│       ├── phase2_step1c_prompt_templates.py
│       ├── phase2_step2_few_shot.py
│       ├── phase2_step3_cot.py
│       ├── phase2_step4_structured_output.py
│       ├── phase2_step5_system_prompt.py
│       ├── phase2_step5b_v3_and_code_guard.py
│       ├── phase2_step6_prompt_iteration.py
│       └── code_reviewer.py           # 阶段验收：结构化代码审查助手
├── obsidian/                      # 学习笔记
│   ├── Phase1 学习笔记.md
│   └── Phase2 学习笔记.md
└── .claude/
    └── CLAUDE.md                  # 项目约定与 Mentor 角色定义
```

## 快速开始

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env  # 填入你的 API key

# 运行 Phase 1 代码
python code/phase1/phase1_step1_api_call.py
python code/phase1/assistant.py
```

## 学习方式

本项目不是传统课程模式，而是**实验驱动**：

- 每个 Phase 的学习计划文档定义了目标和 Milestone
- 通过手写代码 + 跑实验来理解概念，不是看教程
- 所有学习过程中的问答都整理到 `obsidian/` 下的笔记中

## 技术栈

- **语言**: Python 3
- **SDK**: OpenAI 兼容格式、Dashscope（阿里云百炼）
- **模型**: qwen-plus（通过阿里云 Dashscope 调用）
- **笔记**: Obsidian（Markdown）

## 相关文档

- [学习计划](plan/phase1_api_practice.md)
- [学习笔记](obsidian/Phase1%20学习笔记.md)
- [项目约定](.claude/CLAUDE.md)
