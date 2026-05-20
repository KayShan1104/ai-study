# Agent Quality Platform

> A CLI toolkit for systematic AI Agent evaluation, testing, and security auditing.

## Features

- **Multi-dimensional Evaluation**: Score agents across 7 metrics (tool correctness, relevance, faithfulness, format, latency, safety, cost)
- **Dataset Management**: Build and version test datasets with LLM-assisted case generation
- **Automated Pipeline**: CI/CD-ready evaluation with threshold gates and trend tracking
- **Trace System**: Full execution traces (Thought → Action → Observation) for debugging
- **Security Scanner**: 30+ test cases aligned with OWASP LLM Top 10
- **Report Generation**: JSON reports with pass/fail verdicts and trend comparison

## Quick Start

```bash
# Install
pip install -r requirements.txt
cp .env.example .env  # Add your API key

# Run evaluation
python -m src.evaluator --dataset data/test_cases.json

# Run security scan
python -m src.security --report

# Run full pipeline
python -m src.pipeline --all
```

## Architecture

```
├── src/
│   ├── metrics.py       # Evaluation metrics engine
│   ├── evaluator.py     # Single test case evaluation
│   ├── dataset.py       # Dataset builder and manager
│   ├── pipeline.py      # CI/CD evaluation pipeline
│   ├── trace.py         # Execution trace system
│   ├── security.py      # OWASP LLM security scanner
│   └── report.py        # Report generation and trend analysis
├── tests/               # Unit tests
├── data/                # Test datasets
└── reports/             # Evaluation reports
```

## Metrics

| Metric | Weight | Method | Threshold |
|--------|--------|--------|-----------|
| Tool Use Correctness | 30% | Rule-based | 80 |
| Answer Relevance | 20% | LLM-as-judge | 60 |
| Faithfulness | 15% | LLM-as-judge | - |
| Format Compliance | 10% | Rule-based | - |
| Latency | 10% | Threshold | 30s |
| Safety | 10% | LLM-as-judge | 70 |
| Cost | 5% | Threshold | 5k tokens |

## Requirements

- Python 3.9+
- OpenAI-compatible API (Dashscope, OpenAI, etc.)
- `.env` with `ANTHROPIC_API_KEY` and `API_BASE_URL`

## License

MIT
