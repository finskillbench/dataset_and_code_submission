#!/usr/bin/env python3
"""Lightweight task explorer — no API keys, no model calls.

Loads one task per subtask (or a specific task), prints the agent prompt,
shows the expected output schema, and scores a dummy response plus the
ground truth to demonstrate the verifier.

Usage:
    python3.12 demo.py
    python3.12 demo.py --subtask normalization
    python3.12 demo.py --task-id fa_norm_AAPL_2024-03-31

Requires: python3.12 -m pip install -r analysis/requirements.txt  (pandas, numpy, scipy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add evaluation code to sys.path
# ---------------------------------------------------------------------------
SUBMISSION_ROOT = Path(__file__).resolve().parent
EVAL_DIR = SUBMISSION_ROOT / "evaluation" / "finskillbench_agent"
sys.path.insert(0, str(EVAL_DIR))

from tasks import (  # noqa: E402
    load_all_tasks,
    build_instruction,
    derive_output_schema,
    score_task,
)

SUBTASK_ORDER = [
    "normalization", "earnings_quality", "driver_decomposition",
    "unconstrained_optimization", "constrained_optimization", "rebalancing",
    "black_litterman", "tool_use_parameterization",
    "constraint_monitoring", "risk_identification", "risk_remediation", "stress_testing",
]


def demo_task(task: dict) -> None:
    """Print task details and score a dummy + ground-truth response."""
    tid = task.get("task_id", "?")
    domain = task.get("skill", task.get("domain", "?"))
    subtask = task.get("sub_task", "?")
    difficulty = task.get("difficulty", "?")
    as_of = task.get("as_of_date", "?")

    print(f"\n{'='*72}")
    print(f"  Task ID:    {tid}")
    print(f"  Domain:     {domain}")
    print(f"  Subtask:    {subtask}")
    print(f"  Difficulty: {difficulty}")
    print(f"  As-of date: {as_of}")
    print(f"{'='*72}")

    # Build the prompt the agent would see
    instruction = build_instruction(task, condition="no_skill")
    print(f"\n--- Agent Prompt (first 2000 chars) ---")
    print(instruction[:2000])
    if len(instruction) > 2000:
        print(f"... ({len(instruction) - 2000} more chars)")

    # Show expected output schema (NOT the answer)
    schema = derive_output_schema(task["expected_output"])
    print(f"\n--- Expected Output Schema ---")
    print(json.dumps(schema, indent=2))

    # Score a dummy all-zeros response
    dummy = _make_dummy(task["expected_output"])
    dummy_result = score_task(task, json.dumps(dummy))
    print(f"\n--- Dummy Response Score ---")
    print(f"  Score:  {dummy_result.get('score', '?')}")
    print(f"  Method: {dummy_result.get('scoring_method', '?')}")

    # Score the ground truth against itself
    gt_result = score_task(task, json.dumps(task["expected_output"]))
    print(f"\n--- Ground Truth Self-Score ---")
    print(f"  Score:  {gt_result.get('score', '?')}")
    print(f"  Method: {gt_result.get('scoring_method', '?')}")


def _make_dummy(expected: dict) -> dict:
    """Create a zero-valued response matching the expected output structure."""
    out = {}
    for key, val in expected.items():
        if isinstance(val, dict):
            out[key] = _make_dummy(val)
        elif isinstance(val, list):
            if val and isinstance(val[0], dict):
                out[key] = [_make_dummy(val[0])]
            else:
                out[key] = [0]
        elif isinstance(val, bool):
            out[key] = False
        elif isinstance(val, (int, float)):
            out[key] = 0
        elif isinstance(val, str):
            out[key] = ""
        else:
            out[key] = None
    return out


def main():
    parser = argparse.ArgumentParser(description="Explore FinSkillBench tasks (no API keys).")
    parser.add_argument("--subtask", help="Show one task from this subtask")
    parser.add_argument("--task-id", help="Show a specific task by ID")
    parser.add_argument("--all", action="store_true", help="Show one task per subtask")
    args = parser.parse_args()

    print("Loading tasks...")
    all_tasks = load_all_tasks()
    print(f"Loaded {len(all_tasks)} tasks across {len(set(t.get('sub_task','') for t in all_tasks))} subtasks\n")

    if args.task_id:
        matches = [t for t in all_tasks if t.get("task_id") == args.task_id]
        if not matches:
            print(f"Task not found: {args.task_id}")
            sys.exit(1)
        demo_task(matches[0])
    elif args.subtask:
        matches = [t for t in all_tasks if t.get("sub_task") == args.subtask]
        if not matches:
            print(f"No tasks for subtask: {args.subtask}")
            sys.exit(1)
        demo_task(matches[0])
    else:
        # Default: one task per subtask
        shown = set()
        for st in SUBTASK_ORDER:
            if st in shown:
                continue
            matches = [t for t in all_tasks if t.get("sub_task") == st]
            if matches:
                demo_task(matches[0])
                shown.add(st)
            if not args.all and len(shown) >= 3:
                remaining = len(SUBTASK_ORDER) - len(shown)
                print(f"\n... ({remaining} more subtasks available, use --all to see all)")
                break


if __name__ == "__main__":
    main()
