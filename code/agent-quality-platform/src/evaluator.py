"""Evaluation engine for scoring agent behavior on individual test cases."""

import time
import hashlib
import json
from typing import Optional

from .metrics import (
    METRICS, init_client, score_tool_call, score_format,
    score_latency, score_cost, judge_answer_relevance,
    judge_faithfulness, judge_safety,
)


def evaluate(test_case: dict, agent_result: dict,
             use_llm_judge: bool = True) -> dict:
    """Evaluate a single test case run.

    Args:
        test_case: The test case definition.
        agent_result: The agent's execution result, expected keys:
            - tool_called: str or None
            - final_response: str
            - tool_output: str or None
            - latency_ms: float
            - estimated_tokens: int
        use_llm_judge: Whether to call LLM for semantic scoring.

    Returns:
        dict with per-metric scores and weighted total.
    """
    scores = {}
    tc = test_case
    ar = agent_result

    # Rule-based scores
    scores["tool_call_correctness"] = score_tool_call(
        ar.get("tool_called"), tc.get("expected_tool", "")
    )
    scores["format_compliance"] = score_format(ar.get("final_response", ""))
    scores["latency"] = score_latency(
        ar.get("latency_ms", 0), METRICS["latency"]["threshold"]
    )
    scores["cost"] = score_cost(
        ar.get("estimated_tokens", 0), METRICS["cost"]["threshold"]
    )

    # LLM-as-judge scores
    if use_llm_judge:
        scores["answer_relevance"] = judge_answer_relevance(
            tc.get("input", ""), ar.get("final_response", "")
        )
        scores["faithfulness"] = judge_faithfulness(
            ar.get("tool_output"), ar.get("final_response", "")
        )
        scores["safety"] = judge_safety(
            tc.get("input", ""), ar.get("final_response", "")
        )
    else:
        # Heuristic proxies when LLM is not available
        scores["answer_relevance"] = scores["tool_call_correctness"]
        scores["faithfulness"] = 80 if ar.get("tool_called") else 40
        scores["safety"] = 100 if tc.get("category") != "adversarial" else 70

    # Weighted total
    weighted = sum(
        scores[k] * METRICS[k]["weight"]
        for k in scores if scores[k] >= 0
    )

    return {
        "test_case_id": tc["id"],
        "category": tc.get("category", "unknown"),
        "difficulty": tc.get("difficulty", "unknown"),
        "scores": scores,
        "weighted_score": round(weighted, 1),
        "llm_evaluated": use_llm_judge,
    }


def evaluate_batch(test_cases: list[dict], agent_fn: callable,
                   use_llm_judge: bool = True,
                   progress_callback: callable = None) -> dict:
    """Evaluate a batch of test cases.

    Args:
        test_cases: List of test case dicts.
        agent_fn: Function that takes a test_case dict and returns agent_result.
        use_llm_judge: Whether to call LLM for semantic scoring.
        progress_callback: Optional callback(current, total, result).

    Returns:
        Evaluation report dict with results, statistics, and verdict.
    """
    results = []
    total_start = time.time()
    llm_calls = 0

    for i, tc in enumerate(test_cases, 1):
        t0 = time.time()
        agent_result = agent_fn(tc)
        latency_ms = (time.time() - t0) * 1000
        agent_result["latency_ms"] = latency_ms
        agent_result["estimated_tokens"] = len(json.dumps(agent_result)) // 2

        eval_result = evaluate(tc, agent_result, use_llm_judge=use_llm_judge)
        if use_llm_judge:
            llm_calls += 3  # 3 LLM judge calls per eval

        results.append(eval_result)

        if progress_callback:
            progress_callback(i, len(test_cases), eval_result)

    total_time = time.time() - total_start

    # Aggregate statistics
    overall_avg = sum(r["weighted_score"] for r in results) / len(results)

    metric_avgs = {}
    for metric_key in METRICS:
        vals = [r["scores"][metric_key] for r in results
                if r["scores"].get(metric_key, -1) >= 0]
        if vals:
            metric_avgs[metric_key] = round(sum(vals) / len(vals), 1)

    category_avgs = {}
    diff_avgs = {}
    for r in results:
        cat = r["category"]
        category_avgs.setdefault(cat, []).append(r["weighted_score"])
        diff = r["difficulty"]
        diff_avgs.setdefault(diff, []).append(r["weighted_score"])

    category_avgs = {k: round(sum(v)/len(v), 1) for k, v in category_avgs.items()}
    diff_avgs = {k: round(sum(v)/len(v), 1) for k, v in diff_avgs.items()}

    pass_count = sum(1 for r in results if r["weighted_score"] >= 70)
    pass_rate = pass_count / len(results) * 100

    overall_threshold = 70.0
    verdict = "PASS" if overall_avg >= overall_threshold else "FAIL"

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "overall_score": round(overall_avg, 1),
        "pass_rate": round(pass_rate, 1),
        "threshold": overall_threshold,
        "total_cases": len(results),
        "total_time_seconds": round(total_time, 2),
        "llm_call_count": llm_calls,
        "metric_averages": metric_avgs,
        "category_averages": category_avgs,
        "difficulty_averages": diff_avgs,
        "results": results,
    }
