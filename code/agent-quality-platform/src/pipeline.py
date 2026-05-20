"""CI/CD evaluation pipeline.

Loads dataset, runs evaluation, generates reports with trend comparison.
Supports --ci mode where failing threshold exits with code 1.
"""

import os
import sys
import json
import argparse
from datetime import datetime

from .metrics import METRICS, init_client
from .dataset import load_dataset
from .evaluator import evaluate_batch
from .report import save_report, print_report, load_history

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "eval_history.json")


def run_pipeline(dataset_path: str = "data/test_cases.json",
                 use_llm: bool = True,
                 ci_mode: bool = False,
                 api_key: str = None,
                 base_url: str = None,
                 model: str = "qwen-plus"):
    """Run the full evaluation pipeline.

    Args:
        dataset_path: Path to test dataset JSON.
        use_llm: Whether to use LLM-as-judge.
        ci_mode: Exit with code 1 on failure.
        api_key: OpenAI-compatible API key.
        base_url: API base URL.
        model: Model name for judging.
    """
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run: python -m src.dataset_gen to create one.")
        sys.exit(1)

    test_cases, metadata = load_dataset(dataset_path)
    print(f"Loaded dataset: {metadata.get('name', 'unknown')} v{metadata.get('version', '?')}")
    print(f"Total cases: {len(test_cases)}\n")

    # Initialize LLM client if needed
    if use_llm and api_key and base_url:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        init_client(client, model)
        print(f"LLM judge: {model}\n")

    # Simple simulated agent for demo (no real API calls)
    # In production, replace this with your actual agent call
    def mock_agent_fn(test_case):
        """Mock agent that simulates basic behavior."""
        import hashlib
        seed = int(hashlib.md5(test_case["id"].encode()).hexdigest()[:8], 16)
        quality = seed % 100
        expected = test_case.get("expected_tool", "code_review")
        cat = test_case.get("category", "normal")

        tool_called = expected if (quality < 85 or cat == "normal") else None

        return {
            "tool_called": tool_called,
            "final_response": f"代码审查完成。以下是发现的问题：\n- 建议添加类型注解\n- 注意异常处理\n```python\n# optimized version\n```",
            "tool_output": "审查完成，发现 2 个问题" if tool_called else None,
            "latency_ms": 1500 + (seed % 500),
            "estimated_tokens": 200 + (seed % 100),
        }

    # Run evaluation
    def progress(current, total, result):
        if current % 10 == 0 or current == total:
            print(f"  Progress: {current}/{total} ({current*100//total}%)")

    report = evaluate_batch(test_cases, mock_agent_fn,
                            use_llm_judge=False,  # Set True with real API
                            progress_callback=progress)

    report["dataset"] = metadata
    report["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print report
    print_report(report, METRICS)

    # Save report
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = save_report(report, reports_dir)

    # Trend comparison
    history = load_history(os.path.join(reports_dir, "eval_history.json"))
    history["runs"].append({
        "run_id": report["run_id"],
        "timestamp": report["timestamp"],
        "overall_score": report["overall_score"],
        "pass_rate": report["pass_rate"],
        "verdict": report["verdict"],
    })
    history["runs"] = history["runs"][-10:]

    history_path = os.path.join(reports_dir, "eval_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if len(history["runs"]) > 1:
        prev = history["runs"][-2]
        delta = report["overall_score"] - prev["overall_score"]
        arrow = "UP" if delta > 0 else "DOWN" if delta < 0 else "SAME"
        print(f"\n  Trend vs previous: {arrow} {abs(delta):.1f} points")

    # CI exit code
    ci_status = "PASS" if report["verdict"] == "PASS" else "FAIL"
    print(f"\n  CI Status: {ci_status}")
    print(f"  Report: {report_path}")

    if ci_mode and report["verdict"] == "FAIL":
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Agent Evaluation Pipeline")
    parser.add_argument("--dataset", default="data/test_cases.json",
                        help="Path to test dataset JSON")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM-as-judge")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: exit 1 on failure")
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        use_llm=not args.no_llm,
        ci_mode=args.ci,
    )


if __name__ == "__main__":
    main()
