"""Unit tests for metrics, evaluator, dataset, and security modules."""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import (
    score_tool_call, score_format, score_latency, score_cost, METRICS
)
from src.dataset import make_test_case, build_sample_dataset, save_dataset, load_dataset
from src.security import build_security_cases, OWASP_LLM_TOP_10


# --- Metrics Tests ---

def test_score_tool_call_exact_match():
    assert score_tool_call("code_review", "code_review") == 100

def test_score_tool_call_wrong_tool():
    assert score_tool_call("suggest_fix", "code_review") == 30

def test_score_tool_call_none():
    assert score_tool_call(None, "code_review") == 0

def test_score_format_with_code_block():
    resp = "Here's the code:\n```python\ndef foo(): pass\n```"
    assert score_format(resp) >= 65

def test_score_format_plain():
    resp = "This is plain text with no formatting."
    assert score_format(resp) == 50

def test_score_latency_fast():
    assert score_latency(5000, 30000) == 100

def test_score_latency_at_threshold():
    assert score_latency(30000, 30000) == 0

def test_score_latency_over():
    assert score_latency(60000, 30000) == 0

def test_score_cost_low():
    assert score_cost(1000, 5000) == 100

def test_score_cost_over():
    assert score_cost(10000, 5000) == 0

def test_metrics_weights_sum():
    total = sum(m["weight"] for m in METRICS.values())
    assert abs(total - 1.0) < 0.001


# --- Dataset Tests ---

def test_make_test_case():
    tc = make_test_case("TC_001", "normal", "easy", "print(1)",
                        "code_review", ["output"], ["error"])
    assert tc["id"] == "TC_001"
    assert tc["category"] == "normal"
    assert tc["difficulty"] == "easy"
    assert "expected_tool" in tc

def test_build_sample_dataset():
    cases = build_sample_dataset()
    assert len(cases) == 5
    assert all("id" in tc for tc in cases)
    assert all("input" in tc for tc in cases)
    assert all("expected_tool" in tc for tc in cases)

def test_save_and_load_dataset():
    cases = build_sample_dataset()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_dataset(cases, path, name="test", version="1.0")
        loaded, meta = load_dataset(path)
        assert len(loaded) == 5
        assert meta["name"] == "test"
        assert meta["total_cases"] == 5
    finally:
        os.unlink(path)


# --- Security Tests ---

def test_security_cases_count():
    cases = build_security_cases()
    assert len(cases) >= 30, f"Expected 30+ cases, got {len(cases)}"

def test_security_cases_have_required_fields():
    cases = build_security_cases()
    required = {"id", "owasp_category", "category", "difficulty", "input",
                "expected_defense", "anti_patterns"}
    for tc in cases:
        assert required.issubset(tc.keys()), f"Missing fields in {tc['id']}"

def test_owasp_top_10_defined():
    assert len(OWASP_LLM_TOP_10) == 10
    for code, info in OWASP_LLM_TOP_10.items():
        assert "name" in info
        assert "description" in info
        assert "mitigation" in info

def test_security_coverage():
    cases = build_security_cases()
    categories = set(tc["category"] for tc in cases)
    # Should cover at least these categories
    assert "prompt_injection" in categories
    assert "info_leak" in categories
    assert "excessive_agency" in categories
    assert "unsafe_output" in categories
    assert "dos" in categories


if __name__ == "__main__":
    tests = [
        test_score_tool_call_exact_match,
        test_score_tool_call_wrong_tool,
        test_score_tool_call_none,
        test_score_format_with_code_block,
        test_score_format_plain,
        test_score_latency_fast,
        test_score_latency_at_threshold,
        test_score_latency_over,
        test_score_cost_low,
        test_score_cost_over,
        test_metrics_weights_sum,
        test_make_test_case,
        test_build_sample_dataset,
        test_save_and_load_dataset,
        test_security_cases_count,
        test_security_cases_have_required_fields,
        test_owasp_top_10_defined,
        test_security_coverage,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed > 0 else 0)
