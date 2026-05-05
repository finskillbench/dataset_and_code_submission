#!/usr/bin/env python3
"""Verify every inline numeric claim in the paper against shipped data.

Usage:
    python analysis/verify_claims.py --claim all
    python analysis/verify_claims.py --claim episode_count
    python analysis/verify_claims.py --claim all --format json

Requires: pip install -r analysis/requirements.txt  (pandas, numpy, scipy)
No API keys needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = SUBMISSION_ROOT / "results"
DATA_DIR = SUBMISSION_ROOT / "data"
MAIN_RESULTS = RESULTS_DIR / "finskillbench_agent" / "results_all.jsonl"
BY_SUBTASK_DIR = RESULTS_DIR / "finskillbench_agent" / "by_subtask"
HERMES_NO_SKILL = RESULTS_DIR / "hermes_agent" / "hermes_no_skill.jsonl"
HERMES_CURATED = RESULTS_DIR / "hermes_agent" / "hermes_curated.jsonl"


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_main_results() -> pd.DataFrame:
    """Load main results, merging from by_subtask/ if results_all.jsonl is absent."""
    if MAIN_RESULTS.exists():
        return load_jsonl(MAIN_RESULTS)
    rows = []
    for f in sorted(BY_SUBTASK_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Claim checks
# ---------------------------------------------------------------------------
def check_episode_count() -> dict:
    """Paper says 2,603 task episodes across all episode JSONs.

    This is a hybrid count: all FA data episodes (2,243) + evaluated PC tasks
    (200 = 40 per subtask × 5) + evaluated RM tasks (160 = 40 per subtask × 4).
    The benchmark ships more PC episodes than were evaluated; the runner uses
    the first 40 per subtask.
    """
    expected = 2603

    # Count FA episodes from data directory
    fa_count = 0
    fa_dir = DATA_DIR / "fundamentals" / "episodes" / "layer_a"
    if fa_dir.exists():
        for subtask_dir in fa_dir.iterdir():
            if subtask_dir.is_dir():
                fa_count += len(list(subtask_dir.glob("*.json")))

    # Count evaluated unique tasks from results
    df = load_main_results()
    pc_tasks = df[df["domain"] == "portfolio_construction"]["task_id"].nunique()
    rm_tasks = df[df["domain"] == "risk_management"]["task_id"].nunique()

    actual = fa_count + pc_tasks + rm_tasks
    return {
        "claim": "episode_count",
        "paper_value": expected,
        "computed_value": actual,
        "detail": f"FA={fa_count} + PC={pc_tasks} + RM={rm_tasks}",
        "pass": actual == expected,
    }


def check_eval_count() -> dict:
    """Paper says 17,820 total evaluations."""
    expected = 17820
    df = load_main_results()
    actual = len(df)
    return {
        "claim": "eval_count",
        "paper_value": expected,
        "computed_value": actual,
        "pass": actual == expected,
    }


def check_validity() -> dict:
    """Paper says 96.4% valid JSON."""
    expected = 96.4
    df = load_main_results()
    actual = round(df["valid_json"].mean() * 100, 1)
    return {
        "claim": "validity",
        "paper_value": f"{expected}%",
        "computed_value": f"{actual}%",
        "pass": actual == expected,
    }


def check_invalid_count() -> dict:
    """Paper says 644 invalid_submission."""
    expected = 644
    df = load_main_results()
    actual = int((df["scoring_method"] == "invalid_submission").sum())
    return {
        "claim": "invalid_count",
        "paper_value": expected,
        "computed_value": actual,
        "pass": actual == expected,
    }


def check_max_turns() -> dict:
    """Paper says 3,736 max_turns_exhausted."""
    expected = 3736
    df = load_main_results()
    actual = int(df["error"].fillna("").str.contains("max_turns", case=False).sum())
    return {
        "claim": "max_turns",
        "paper_value": expected,
        "computed_value": actual,
        "pass": actual == expected,
    }


def check_phi4() -> dict:
    """Paper says Phi-4 mean 0.000."""
    expected = 0.000
    df = load_main_results()
    phi4 = df[df["model"] == "Phi-4"]
    actual = round(float(phi4["score"].mean()), 3)
    return {
        "claim": "phi4",
        "paper_value": expected,
        "computed_value": actual,
        "pass": actual == expected,
    }


def check_curated_delta() -> dict:
    """Paper says curated Δ +0.162 (excl. Phi-4)."""
    expected = 0.162
    df = load_main_results()
    d = df[df["model"] != "Phi-4"]
    ns = d[d["condition"] == "no_skill"]["score"].mean()
    cur = d[d["condition"] == "curated"]["score"].mean()
    actual = round(cur - ns, 3)
    return {
        "claim": "curated_delta",
        "paper_value": f"+{expected}",
        "computed_value": f"+{actual}",
        "pass": actual == expected,
    }


def check_selfgen_delta() -> dict:
    """Paper says self-gen Δ +0.005 (excl. Phi-4)."""
    expected = 0.005
    df = load_main_results()
    d = df[df["model"] != "Phi-4"]
    ns = d[d["condition"] == "no_skill"]["score"].mean()
    sg = d[d["condition"] == "self_generated"]["score"].mean()
    actual = round(sg - ns, 3)
    return {
        "claim": "selfgen_delta",
        "paper_value": f"+{expected}",
        "computed_value": f"+{actual}",
        "pass": actual == expected,
    }


def check_hermes_delta() -> dict:
    """Paper says Hermes overall Δ +0.325."""
    expected = 0.325
    ns = load_jsonl(HERMES_NO_SKILL)
    cur = load_jsonl(HERMES_CURATED)
    actual = round(cur["score"].mean() - ns["score"].mean(), 3)
    return {
        "claim": "hermes_delta",
        "paper_value": f"+{expected}",
        "computed_value": f"+{actual}",
        "pass": actual == expected,
    }


def check_selfgen_load() -> dict:
    """Paper says 97.7% self-gen skill loading (excl. Phi-4)."""
    expected = 97.7
    df = load_main_results()
    sg = df[(df["condition"] == "self_generated") & (df["model"] != "Phi-4")]
    loaded = sg["skills_loaded"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    actual = round(loaded.mean() * 100, 1)
    return {
        "claim": "selfgen_load",
        "paper_value": f"{expected}%",
        "computed_value": f"{actual}%",
        "pass": actual == expected,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_CHECKS = {
    "episode_count": check_episode_count,
    "eval_count": check_eval_count,
    "validity": check_validity,
    "invalid_count": check_invalid_count,
    "max_turns": check_max_turns,
    "phi4": check_phi4,
    "curated_delta": check_curated_delta,
    "selfgen_delta": check_selfgen_delta,
    "hermes_delta": check_hermes_delta,
    "selfgen_load": check_selfgen_load,
}


def main():
    parser = argparse.ArgumentParser(description="Verify inline paper claims.")
    parser.add_argument("--claim", required=True, help="Claim name or 'all'")
    parser.add_argument("--format", default="text", choices=["text", "json"])
    args = parser.parse_args()

    claims = list(ALL_CHECKS.keys()) if args.claim == "all" else [args.claim]
    results = []
    all_pass = True

    for name in claims:
        if name not in ALL_CHECKS:
            print(f"Unknown claim: {name}", file=sys.stderr)
            sys.exit(1)
        result = ALL_CHECKS[name]()
        results.append(result)
        status = "✓" if result["pass"] else "✗"
        if not result["pass"]:
            all_pass = False
        if args.format == "text":
            detail = f" ({result['detail']})" if "detail" in result else ""
            print(f"  {status} {result['claim']}: paper says {result['paper_value']} "
                  f"— computed {result['computed_value']}{detail}")

    if args.format == "json":
        print(json.dumps(results, indent=2, default=str))

    if args.format == "text":
        print(f"\n{'All checks passed.' if all_pass else 'SOME CHECKS FAILED.'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
