#!/usr/bin/env python3
"""Aggregate Experiment 05 results.jsonl rows for a shared --run-id across all subtasks.

Usage (from ``experiments/zqbok_experiment05``)::

    python scripts/aggregate_run_results.py full_run_concurrent_v1

Prints counts, means, and per-subtask lines suitable for pasting into results docs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parents[1]

SUBTASKS = [
    "unconstrained_optimization",
    "constrained_optimization",
    "tool_use_parameterization",
    "rebalancing",
    "black_litterman",
    "constraint_monitoring",
    "risk_identification",
    "stress_testing",
    "risk_remediation",
    "normalization",
    "earnings_quality",
    "driver_decomposition",
]

SUB_DOMAIN = {s: "PC" for s in SUBTASKS[:5]}
SUB_DOMAIN.update({s: "RM" for s in SUBTASKS[5:9]})
SUB_DOMAIN.update({s: "FA" for s in SUBTASKS[9:]})


def is_phi(row: dict) -> bool:
    m = (row.get("model") or "").lower()
    return "phi-4" in m


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "full_run_concurrent_v1"
    rows: list[dict] = []
    for sub in SUBTASKS:
        p = EXPT_DIR / "runs" / sub / run_id / "results.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_sub"] = sub
            rows.append(r)

    n = len(rows)
    non_phi = [r for r in rows if not is_phi(r)]
    print(f"run_id={run_id}  rows={n}")
    if not rows:
        return

    scores = [float(r.get("score", 0) or 0) for r in rows]
    print(f"mean_all {sum(scores)/n:.4f}")
    if non_phi:
        np_s = [float(r.get("score", 0) or 0) for r in non_phi]
        print(f"mean_excl_phi {sum(np_s)/len(np_s):.4f}")
    vj = sum(1 for r in rows if r.get("valid_json"))
    print(f"valid_json {vj}/{n} ({100*vj/n:.1f}%)")
    sm = Counter(r.get("scoring_method") for r in rows)
    for k, v in sorted(sm.items(), key=lambda x: (-x[1], str(x[0]))):
        print(f"  {k}: {v}")
    print(f"rows_with_error {sum(1 for r in rows if r.get('error'))}")

    print("\n--- by domain (pooled) ---")
    for dom in ("PC", "RM", "FA"):
        dr = [r for r in rows if SUB_DOMAIN.get(r["_sub"]) == dom]
        if not dr:
            continue
        m = sum(float(r.get("score", 0) or 0) for r in dr) / len(dr)
        vjc = sum(1 for r in dr if r.get("valid_json"))
        inv = sum(1 for r in dr if r.get("scoring_method") == "invalid_submission")
        print(f"{dom} n={len(dr)} mean={m:.4f} valid%={100*vjc/len(dr):.1f}% invalid_submission={inv}")

    print("\n--- per subtask ---")
    for sub in SUBTASKS:
        srs = [r for r in rows if r["_sub"] == sub]
        if not srs:
            print(f"{sub}: (no rows)")
            continue
        n2 = len(srs)
        mean = sum(float(r.get("score", 0) or 0) for r in srs) / n2
        np2 = [r for r in srs if not is_phi(r)]
        mnp = (
            sum(float(r.get("score", 0) or 0) for r in np2) / len(np2)
            if np2
            else 0.0
        )
        vj = sum(1 for r in srs if r.get("valid_json"))
        inv = sum(1 for r in srs if r.get("scoring_method") == "invalid_submission")
        mt = sum(1 for r in srs if r.get("scoring_method") == "max_turns_exhausted")
        inc = sum(1 for r in srs if r.get("scoring_method") == "incomplete_submission")
        print(
            f"{sub}: n={n2} mean={mean:.4f} mean_np={mnp:.4f} "
            f"valid={vj}/{n2} invalid_sub={inv} max_turns={mt} incomplete={inc}"
        )


if __name__ == "__main__":
    main()
