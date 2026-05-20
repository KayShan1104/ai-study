# AI Learning Lab

通过手写代码 + 实验的方式，从零理解 LLM API、Prompt Engineering、RAG、Agent 等核心技术。

## 学习路径

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | LLM API 实操 | 已完成 |
| Phase 2 | Prompt Engineering | 已完成 |
| Phase 3 | RAG（检索增强生成） | 已完成 |
| Phase 4 | Agent 框架 | 已完成 |
| Phase 5 | 探索方向 | 已跳过 |
| Phase 6 | Agent 测试与评估 | 已完成 |
| Phase 7 | PyTorch 模型微调 | 已完成 |

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
│   └── phase7_pytorch_finetuning.md
├── code/                          # 实践代码
│   ├── phase1/
│   │   ├── phase1_step1_api_call.py
│   │   ├── phase1_step2_multiconversation.py
│   │   ├── phase1_step3_structured_output.py
│   │   ├── phase1_step4_function_calling.py
│   │   ├── phase1_step5_multi_tool_routing.py
│   │   └── assistant.py           # 阶段验收：CLI 智能助手
│   ├── phase2/
│   │   ├── phase2_step1_prompt_structure.py
│   │   ├── phase2_step1b_system_vs_user.py
│   │   ├── phase2_step1c_prompt_templates.py
│   │   ├── phase2_step2_few_shot.py
│   │   ├── phase2_step3_cot.py
│   │   ├── phase2_step4_structured_output.py
│   │   ├── phase2_step5_system_prompt.py
│   │   ├── phase2_step5b_v3_and_code_guard.py
│   │   ├── phase2_step6_prompt_iteration.py
│   │   └── code_reviewer.py       # 阶段验收：结构化代码审查助手
│   ├── phase3/
│   │   ├── phase3_step1_embedding.py
│   │   ├── phase3_step2_vector_db.py
│   │   ├── phase3_step3_chunking.py
│   │   ├── phase3_step4_rag_pipeline.py
│   │   ├── phase3_step5_evaluation.py
│   │   ├── phase3_step6_advanced.py
│   │   ├── phase3_acceptance.py
│   │   └── acceptance_results.json
│   ├── phase4/
│   │   ├── phase4_step1_langchain_basics.py
│   │   ├── phase4_step2_tool_and_agent.py
│   │   ├── phase4_step3_memory.py
│   │   ├── phase4_step3_postgres_example.py
│   │   ├── phase4_step4_multi_agent.py
│   │   ├── phase4_step5_handwritten_vs_framework.py
│   │   ├── phase4_step6_error_handling.py
│   │   ├── phase4_acceptance_handwritten.py    # 阶段验收：手写版
│   │   └── phase4_acceptance_framework.py      # 阶段验收：框架版
│   └── phase6/
│       ├── phase6_step0_cognitive_shift.py   # 认知升级：传统测试 vs Agent 测试
│       ├── phase6_step1_eval_metrics.py      # 评估指标体系：LLM-as-judge + 7 维度评分
│       ├── phase6_step2_dataset.py           # 评估数据集：100 用例 + 5 类型
│       ├── phase6_step3_eval_pipeline.py     # 自动化评估流水线：CI 集成 + 趋势对比
│       ├── phase6_step4_trace_debug.py       # Agent 行为监控：Trace 系统 + 错误分析
│       ├── phase6_step5_security_eval.py     # 安全评估：OWASP LLM Top 10 + 30 用例扫描
│       ├── test_cases.json                   # 测试数据集（100 用例）
│       ├── eval_results.json                 # Step 1 评估结果
│       ├── security_report.json              # 安全评估报告
│       ├── eval_results/                     # 流水线评估报告
│       │   ├── eval_history.json
│       │   └── report_*.json
│       └── traces/                           # Agent 执行追踪
│           └── trace_*.json
├── obsidian/                      # 学习笔记
│   ├── Phase1 学习笔记.md
│   ├── Phase2 学习笔记.md
│   ├── Phase3 学习笔记.md
│   ├── Phase4 学习笔记.md
│   └── Phase6 学习笔记.md
│   └── Phase7 学习笔记.md
├── .claude/
│   └── CLAUDE.md                  # 项目约定与 Mentor 角色定义
└── README.md

## Portfolio 项目

Phase 6 的综合成果已独立发布为 GitHub 项目：

- **[Agent Quality Platform](https://github.com/KayShan1104/agent-quality-platform)** — 完整的 Agent 质量评估平台 CLI 工具，包含 7 维度评估引擎、OWASP 安全扫描、数据集管理、CI/CD 流水线集成、18 个单元测试
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

- [学习计划](plan/)
- [Phase1 学习笔记](obsidian/Phase1%20学习笔记.md)
- [Phase2 学习笔记](obsidian/Phase2%20学习笔记.md)
- [Phase3 学习笔记](obsidian/Phase3%20学习笔记.md)
- [Phase4 学习笔记](obsidian/Phase4%20学习笔记.md)
- [Phase6 学习笔记](obsidian/Phase6%20学习笔记.md)
- [Agent Quality Platform](https://github.com/KayShan1104/agent-quality-platform)（Phase 6 综合项目）
