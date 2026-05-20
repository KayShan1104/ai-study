"""CLI entry point for Agent Quality Platform."""

import sys
import os
import argparse
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(
        prog="aqp",
        description="Agent Quality Platform - CLI for AI Agent evaluation",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation pipeline")
    p_eval.add_argument("--dataset", default="data/test_cases.json")
    p_eval.add_argument("--no-llm", action="store_true", help="Disable LLM judge")
    p_eval.add_argument("--ci", action="store_true", help="CI mode")

    # security
    p_sec = sub.add_parser("security", help="Run security scan")
    p_sec.add_argument("--report", action="store_true", help="Save report")
    p_sec.add_argument("--output", default="reports/security_report.json")

    # init
    p_init = sub.add_parser("init", help="Initialize sample dataset")
    p_init.add_argument("--output", default="data/test_cases.json")

    # test
    sub.add_parser("test", help="Run unit tests")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "evaluate":
        from src.pipeline import run_pipeline
        load_dotenv()
        run_pipeline(
            dataset_path=args.dataset,
            use_llm=not args.no_llm,
            ci_mode=args.ci,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("API_BASE_URL"),
            model=os.getenv("MODEL_NAME", "qwen-plus"),
        )

    elif args.command == "security":
        from src.security import build_security_cases, evaluate_security
        import json

        cases = build_security_cases()
        print(f"Running security scan with {len(cases)} test cases...\n")
        report = evaluate_security(cases)

        # Print summary
        s = report["summary"]
        print(f"  {'='*50}")
        print(f"  Security Scan Results")
        print(f"  {'='*50}")
        print(f"  Defended:  {s['defended']}/{s['total']} ({s['pass_rate']}%)")
        print(f"  Breached:  {s['breached']}/{s['total']}")
        print(f"  Risk:      {s['risk_level'].upper()}")
        print()

        if report["breached_results"]:
            print(f"  Failed cases ({len(report['breached_results'])}):")
            for r in report["breached_results"]:
                print(f"    {r['id']} ({r['category']}/{r['threat_type']}/{r['difficulty']})")

        if args.report:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n  Report saved to: {args.output}")

    elif args.command == "init":
        from src.dataset import build_sample_dataset, save_dataset
        cases = build_sample_dataset()
        save_dataset(cases, args.output)
        print(f"Sample dataset created: {args.output} ({len(cases)} cases)")

    elif args.command == "test":
        import subprocess
        test_path = os.path.join(os.path.dirname(__file__), "tests", "test_all.py")
        result = subprocess.run([sys.executable, test_path])
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
