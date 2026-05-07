#!/usr/bin/env python3
"""Verify paper table values, inline claims, and packaged coverage.

This is a strict golden-value guard for reviewer-facing reproducibility. It
compares the current shipped result files against `expected_results.json`, which
is kept in sync with the final paper tables and inline numeric claims.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import reproduce_tables
import verify_claims

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = Path(__file__).with_name("expected_results.json")


def _quiet_call(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("task_id"),
        row.get("model"),
        row.get("condition"),
        row.get("sub_task"),
        row.get("run_idx"),
        row.get("score"),
        row.get("valid_json"),
        row.get("scoring_method"),
    )


def compute_tables() -> list[dict[str, Any]]:
    df = reproduce_tables.load_main_results()
    return [
        _quiet_call(reproduce_tables.table_2, df),
        _quiet_call(reproduce_tables.table_3, df),
        _quiet_call(reproduce_tables.table_4, df),
        _quiet_call(reproduce_tables.table_5, df),
        _quiet_call(reproduce_tables.table_6, df),
        _quiet_call(reproduce_tables.table_7),
    ]


def compute_claims() -> dict[str, dict[str, Any]]:
    claims = {}
    for name, fn in verify_claims.ALL_CHECKS.items():
        result = fn()
        if not bool(result.pop("pass")):
            raise AssertionError(f"Claim checker failed before golden compare: {name}")
        result.pop("claim", None)
        claims[name] = result
    return claims


def compute_coverage() -> dict[str, Any]:
    results_all_path = SUBMISSION_ROOT / "results" / "finskillbench_agent" / "results_all.jsonl"
    by_subtask_dir = SUBMISSION_ROOT / "results" / "finskillbench_agent" / "by_subtask"
    hermes_no_skill_path = SUBMISSION_ROOT / "results" / "hermes_agent" / "hermes_no_skill.jsonl"
    hermes_curated_path = SUBMISSION_ROOT / "results" / "hermes_agent" / "hermes_curated.jsonl"

    results_all = _load_jsonl(results_all_path)
    by_subtask = []
    for path in sorted(by_subtask_dir.glob("*.jsonl")):
        by_subtask.extend(_load_jsonl(path))

    episode_counts = {}
    total_episodes = 0
    episode_dirs = {
        "portfolio_construction": SUBMISSION_ROOT / "data" / "portfolio_construction" / "episodes" / "layer_a",
        "risk_management": SUBMISSION_ROOT / "data" / "risk_management" / "episodes" / "layer_a",
        "fundamental_analysis": SUBMISSION_ROOT / "data" / "fundamentals" / "episodes" / "layer_a",
    }
    for domain, base in episode_dirs.items():
        count = sum(1 for path in base.rglob("*.json") if path.name != "_manifest.json")
        episode_counts[domain] = count
        total_episodes += count
    episode_counts["total"] = total_episodes

    return {
        "results_all_rows": len(results_all),
        "by_subtask_rows": len(by_subtask),
        "by_subtask_matches_results_all": sorted(map(_row_key, by_subtask)) == sorted(map(_row_key, results_all)),
        "hermes_no_skill_rows": sum(1 for _ in open(hermes_no_skill_path)),
        "hermes_curated_rows": sum(1 for _ in open(hermes_curated_path)),
        "episode_counts": episode_counts,
    }


def compare(name: str, actual: Any, expected: Any) -> list[str]:
    if actual == expected:
        return []
    return [
        f"{name} mismatch",
        f"expected: {json.dumps(expected, sort_keys=True)}",
        f"actual:   {json.dumps(actual, sort_keys=True)}",
    ]


def main() -> int:
    expected = json.loads(EXPECTED_PATH.read_text())
    errors: list[str] = []
    errors.extend(compare("tables", compute_tables(), expected["tables"]))
    errors.extend(compare("claims", compute_claims(), expected["claims"]))
    errors.extend(compare("coverage", compute_coverage(), expected["coverage"]))

    if errors:
        print("verify_expected_results.py: expected paper values do not match.", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("verify_expected_results.py: all expected paper values match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
