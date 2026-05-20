"""Report generation, printing, and trend history management."""

import os
import json
from datetime import datetime


def print_report(report: dict, metrics: dict = None):
    """Print evaluation report to console."""
    print("\n" + "=" * 60)
    print("  Agent Evaluation Report")
    print("=" * 60)

    if "run_id" in report:
        print(f"  Run ID:     {report['run_id']}")
    print(f"  Time:       {report.get('timestamp', 'N/A')}")
    print(f"  Cases:      {report.get('total_cases', len(report.get('results', [])))}")
    print(f"  Duration:   {report.get('total_time_seconds', 0):.1f}s")
    print()

    verdict = report.get("verdict", "UNKNOWN")
    status_icon = "[PASS]" if verdict == "PASS" else "[FAIL]"
    print(f"  {status_icon} Verdict:  {verdict}")
    print(f"  Score:      {report.get('overall_score', 0)} (threshold: {report.get('threshold', 70)})")
    print(f"  Pass Rate:  {report.get('pass_rate', 0)}%")
    print()

    # Metric averages
    print("  Metric Averages:")
    if "metric_averages" in report:
        for key, avg in report["metric_averages"].items():
            name = key
            weight = ""
            if metrics and key in metrics:
                name = metrics[key].get("name", key)
                weight = f" (w={metrics[key]['weight']:.0%})"
            threshold = metrics[key].get("threshold") if metrics else None
            mark = "[OK]" if threshold is None or avg >= threshold else "[WARN]"
            print(f"    {mark} {name:<25} {avg:.1f}{weight}")
    print()

    # Category averages
    if "category_averages" in report:
        print("  By Category:")
        for cat, avg in sorted(report["category_averages"].items()):
            mark = "[OK]" if avg >= 70 else "[WARN]"
            print(f"    {mark} {cat:<20} {avg:.1f}")
    print()

    # Difficulty averages
    if "difficulty_averages" in report:
        print("  By Difficulty:")
        for diff, avg in sorted(report["difficulty_averages"].items()):
            print(f"       {diff:<10} {avg:.1f}")


def save_report(report: dict, reports_dir: str = "reports") -> str:
    """Save report to JSON file."""
    os.makedirs(reports_dir, exist_ok=True)
    run_id = report.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(reports_dir, f"report_{run_id}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return path


def load_history(history_path: str = "reports/eval_history.json") -> dict:
    """Load evaluation run history."""
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": []}
