"""Execution trace system for debugging agent behavior.

Provides three-level trace model: RunTrace → StepTrace → ToolCallTrace.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class ToolCallTrace:
    """Trace of a single tool call."""
    name: str
    input_args: dict
    output: Optional[str] = None
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: float = 0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class StepTrace:
    """Trace of a single agent execution step."""
    step_number: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    tool_call: Optional[ToolCallTrace] = None
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: float = 0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class RunTrace:
    """Complete trace of one agent run."""
    run_id: str
    test_case_id: str
    category: str
    difficulty: str
    user_input: str
    steps: list = field(default_factory=list)
    final_response: str = ""
    overall_status: str = "pending"
    total_duration_ms: float = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "test_case_id": self.test_case_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "user_input": self.user_input,
            "steps": [asdict(s) for s in self.steps],
            "final_response": self.final_response,
            "overall_status": self.overall_status,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class TraceCollector:
    """Collects and manages agent execution traces."""

    def __init__(self, output_dir: str = "traces"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.traces: list[RunTrace] = []

    def start_run(self, run_id: str, test_case: dict) -> RunTrace:
        trace = RunTrace(
            run_id=run_id,
            test_case_id=test_case["id"],
            category=test_case.get("category", "unknown"),
            difficulty=test_case.get("difficulty", "unknown"),
            user_input=test_case.get("input", ""),
            start_time=datetime.now().isoformat(),
        )
        self.traces.append(trace)
        return trace

    def add_step(self, trace: RunTrace, step_number: int,
                 thought: str = "", action: str = "") -> StepTrace:
        step = StepTrace(
            step_number=step_number,
            thought=thought,
            action=action,
            status="running",
            start_time=datetime.now().isoformat(),
        )
        trace.steps.append(step)
        return step

    def complete_step(self, step: StepTrace, observation: str = "",
                      status: str = "success", error: Optional[str] = None):
        step.observation = observation
        step.status = error if error else status
        step.end_time = datetime.now().isoformat()
        step.error = error
        if step.start_time and step.end_time:
            t1 = datetime.fromisoformat(step.start_time)
            t2 = datetime.fromisoformat(step.end_time)
            step.duration_ms = (t2 - t1).total_seconds() * 1000

    def complete_run(self, trace: RunTrace, final_response: str = "",
                     status: str = "success", error: Optional[str] = None):
        trace.final_response = final_response
        trace.overall_status = status
        trace.end_time = datetime.now().isoformat()
        trace.error = error
        if trace.start_time and trace.end_time:
            t1 = datetime.fromisoformat(trace.start_time)
            t2 = datetime.fromisoformat(trace.end_time)
            trace.total_duration_ms = (t2 - t1).total_seconds() * 1000
        trace.total_tokens = sum(s.tokens_used for s in trace.steps)
        trace.total_tool_calls = sum(1 for s in trace.steps if s.tool_call)

    def save_trace(self, trace: RunTrace):
        """Save a single trace to JSON."""
        filename = f"trace_{trace.run_id}_{trace.test_case_id}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)

    def save_summary(self):
        """Save summary of all traces."""
        summary_path = os.path.join(self.output_dir, "trace_summary.json")
        summary = {
            "total_runs": len(self.traces),
            "timestamp": datetime.now().isoformat(),
            "traces": [t.to_dict() for t in self.traces],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def get_error_runs(self) -> list[RunTrace]:
        return [t for t in self.traces if t.overall_status in ("error", "timeout")]

    def get_slow_runs(self, threshold_ms: float = 100) -> list[RunTrace]:
        return [t for t in self.traces if t.total_duration_ms > threshold_ms]
