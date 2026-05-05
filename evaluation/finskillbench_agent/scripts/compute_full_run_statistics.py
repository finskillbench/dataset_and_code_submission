#!/usr/bin/env python3
"""Compute statistics for full_run_concurrent_v1 (same definitions as results doc)."""
from __future__ import annotations

import json
import sys
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
    return "phi-4" in (row.get("model") or "").lower()


def load_rows(run_id: str) -> list[dict]:
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
    return rows


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "full_run_concurrent_v1"
    rows = load_rows(run_id)
    n = len(rows)
    non_phi = [r for r in rows if not is_phi(r)]
    scores = [float(r.get("score", 0) or 0) for r in rows]
    np_scores = [float(r.get("score", 0) or 0) for r in non_phi]
    phi_rows = [r for r in rows if is_phi(r)]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    stats = {
        "run_id": run_id,
        "n_rows": n,
        "mean_all": mean(scores),
        "mean_excl_phi": mean(np_scores),
        "mean_phi_only": mean([float(r.get("score", 0) or 0) for r in phi_rows]),
        "n_phi": len(phi_rows),
        "valid_json": sum(1 for r in rows if r.get("valid_json")),
        "invalid_submission": sum(1 for r in rows if r.get("scoring_method") == "invalid_submission"),
        "max_turns_exhausted": sum(1 for r in rows if r.get("scoring_method") == "max_turns_exhausted"),
        "incomplete_submission": sum(1 for r in rows if r.get("scoring_method") == "incomplete_submission"),
        "rows_with_error": sum(1 for r in rows if r.get("error")),
    }
    print(json.dumps(stats, indent=2))

    # Domain table
    print("\n--- by domain ---")
    for dom in ("PC", "RM", "FA"):
        dr = [r for r in rows if SUB_DOMAIN.get(r["_sub"]) == dom]
        if not dr:
            continue
        m = mean([float(r.get("score", 0) or 0) for r in dr])
        vj = sum(1 for r in dr if r.get("valid_json"))
        inv = sum(1 for r in dr if r.get("scoring_method") == "invalid_submission")
        dr_np = [r for r in dr if not is_phi(r)]
        mnp = mean([float(r.get("score", 0) or 0) for r in dr_np]) if dr_np else 0.0
        print(f"{dom} n={len(dr)} mean={m:.4f} mean_excl_phi={mnp:.4f} valid={vj}/{len(dr)} invalid_sub={inv}")

    # Condition means (excl phi) — pooled all subtasks
    print("\n--- condition means (excl phi) ---")
    for cond in ("no_skill", "curated", "self_generated"):
        cr = [r for r in non_phi if r.get("condition") == cond]
        if cr:
            print(f"{cond} n={len(cr)} mean={mean([float(r.get('score', 0) or 0) for r in cr]):.4f}")

    # Per subtask
    print("\n--- per subtask ---")
    for sub in SUBTASKS:
        srs = [r for r in rows if r["_sub"] == sub]
        if not srs:
            continue
        m = mean([float(r.get("score", 0) or 0) for r in srs])
        srs_np = [r for r in srs if not is_phi(r)]
        mnp = mean([float(r.get("score", 0) or 0) for r in srs_np]) if srs_np else 0.0
        vj = sum(1 for r in srs if r.get("valid_json"))
        inv = sum(1 for r in srs if r.get("scoring_method") == "invalid_submission")
        mt = sum(1 for r in srs if r.get("scoring_method") == "max_turns_exhausted")
        inc = sum(1 for r in srs if r.get("scoring_method") == "incomplete_submission")
        errn = sum(1 for r in srs if r.get("error"))
        print(f"{sub}: mean={m:.4f} mean_np={mnp:.4f} valid={vj}/{len(srs)} inv={inv} max_turns={mt} incomplete={inc} err={errn}")

    # Condition × subtask (excl phi)
    print("\n--- condition x subtask (excl phi) ---")

    def cell_mean(sub: str, cond: str) -> float:
        cr = [r for r in non_phi if r["_sub"] == sub and r.get("condition") == cond]
        return mean([float(r.get("score", 0) or 0) for r in cr]) if cr else 0.0

    pc_subs = SUBTASKS[:5]
    rm_subs = SUBTASKS[5:9]
    fa_subs = SUBTASKS[9:]
    for label, subs in ("PC", pc_subs), ("RM", rm_subs), ("FA", fa_subs):
        print(f"\n{label}:")
        for cond in ("no_skill", "curated", "self_generated"):
            parts = [f"{cell_mean(s, cond):.3f}" for s in subs]
            comb = mean([float(r.get("score", 0) or 0) for r in non_phi if r["_sub"] in subs and r.get("condition") == cond])
            print(f"  {cond}: {' | '.join(parts)} | combined={comb:.3f}")


if __name__ == "__main__":
    main()
