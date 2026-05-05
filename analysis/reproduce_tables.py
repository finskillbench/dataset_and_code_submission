#!/usr/bin/env python3
"""Reproduce every paper table from shipped result JSONL files.

Usage:
    python analysis/reproduce_tables.py --table all
    python analysis/reproduce_tables.py --table 2
    python analysis/reproduce_tables.py --table 2 --format json

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
MAIN_RESULTS = RESULTS_DIR / "finskillbench_agent" / "results_all.jsonl"
BY_SUBTASK_DIR = RESULTS_DIR / "finskillbench_agent" / "by_subtask"
HERMES_NO_SKILL = RESULTS_DIR / "hermes_agent" / "hermes_no_skill.jsonl"
HERMES_CURATED = RESULTS_DIR / "hermes_agent" / "hermes_curated.jsonl"

DOMAIN_MAP = {
    "black_litterman": "PC",
    "constrained_optimization": "PC",
    "rebalancing": "PC",
    "tool_use_parameterization": "PC",
    "unconstrained_optimization": "PC",
    "constraint_monitoring": "RM",
    "risk_identification": "RM",
    "risk_remediation": "RM",
    "stress_testing": "RM",
    "driver_decomposition": "FA",
    "earnings_quality": "FA",
    "normalization": "FA",
}

DOMAIN_FULL = {"PC": "Portfolio construction", "RM": "Risk management", "FA": "Fundamental analysis"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    # Merge from per-subtask files
    rows = []
    for f in sorted(BY_SUBTASK_DIR.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray | None = None, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap 95% CI for mean(a) or mean(a - b) if b is provided."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if b is not None:
        diff = a - b
        stats = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    else:
        stats = np.array([a[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def fmt_delta(val: float) -> str:
    return f"{val:+.3f}" if val != 0 else "---"


# ---------------------------------------------------------------------------
# Table 2: Aggregate condition means (excl. Phi-4)
# ---------------------------------------------------------------------------
def table_2(df: pd.DataFrame) -> dict:
    d = df[df["model"] != "Phi-4"].copy()
    results = {}
    no_skill_scores = d[d["condition"] == "no_skill"]["score"].values
    for cond in ["no_skill", "curated", "self_generated"]:
        scores = d[d["condition"] == cond]["score"].values
        mean = float(scores.mean())
        if cond == "no_skill":
            delta = 0.0
            ci = "---"
        else:
            delta = float(scores.mean() - no_skill_scores.mean())
            lo, hi = bootstrap_ci(scores, no_skill_scores)
            ci = fmt_ci(lo, hi)
        results[cond] = {"mean": round(mean, 3), "delta": round(delta, 3), "ci": ci}

    print("\n=== Table 2: Aggregate Condition Means (excl. Phi-4) ===")
    print(f"{'Condition':<20} {'Mean Score':>12} {'Δ vs No-Skill':>15} {'95% CI':>22}")
    print("-" * 72)
    for cond, v in results.items():
        label = cond.replace("_", "-").title()
        print(f"{label:<20} {v['mean']:>12.3f} {fmt_delta(v['delta']):>15} {v['ci']:>22}")
    return {"table": 2, "data": results}


# ---------------------------------------------------------------------------
# Table 3: Domain-level results (excl. Phi-4)
# ---------------------------------------------------------------------------
def table_3(df: pd.DataFrame) -> dict:
    d = df[df["model"] != "Phi-4"].copy()
    d["domain_short"] = d["sub_task"].map(DOMAIN_MAP)
    results = {}
    for dom in ["PC", "RM", "FA"]:
        dd = d[d["domain_short"] == dom]
        row = {}
        ns = dd[dd["condition"] == "no_skill"]["score"].values
        for cond in ["no_skill", "curated", "self_generated"]:
            scores = dd[dd["condition"] == cond]["score"].values
            row[cond] = round(float(scores.mean()), 3)
        cur = dd[dd["condition"] == "curated"]["score"].values
        sg = dd[dd["condition"] == "self_generated"]["score"].values
        row["curated_delta"] = round(float(cur.mean() - ns.mean()), 3)
        row["selfgen_delta"] = round(float(sg.mean() - ns.mean()), 3)
        lo_c, hi_c = bootstrap_ci(cur, ns)
        lo_s, hi_s = bootstrap_ci(sg, ns)
        row["curated_ci"] = fmt_ci(lo_c, hi_c)
        row["selfgen_ci"] = fmt_ci(lo_s, hi_s)
        results[dom] = row

    print("\n=== Table 3: Domain-Level Results (excl. Phi-4) ===")
    print(f"{'Domain':<25} {'No-Skill':>10} {'Curated':>10} {'Self-Gen':>10} {'Cur Δ':>10} {'SG Δ':>10}")
    print("-" * 78)
    for dom in ["PC", "RM", "FA"]:
        r = results[dom]
        print(f"{DOMAIN_FULL[dom]:<25} {r['no_skill']:>10.3f} {r['curated']:>10.3f} "
              f"{r['self_generated']:>10.3f} {r['curated_delta']:>+10.3f} {r['selfgen_delta']:>+10.3f}")
    return {"table": 3, "data": results}


# ---------------------------------------------------------------------------
# Table 4: Subtask-level results (excl. Phi-4)
# ---------------------------------------------------------------------------
def table_4(df: pd.DataFrame) -> dict:
    d = df[df["model"] != "Phi-4"].copy()
    results = {}
    for st in sorted(d["sub_task"].unique()):
        dd = d[d["sub_task"] == st]
        row = {"domain": DOMAIN_MAP.get(st, "?")}
        for cond in ["no_skill", "curated", "self_generated"]:
            scores = dd[dd["condition"] == cond]["score"].values
            row[cond] = round(float(scores.mean()), 3)
        row["delta"] = round(row["curated"] - row["no_skill"], 3)
        results[st] = row

    # Sort by delta descending
    sorted_st = sorted(results.keys(), key=lambda s: results[s]["delta"], reverse=True)

    print("\n=== Table 4: Subtask-Level Results (excl. Phi-4) ===")
    print(f"{'Subtask':<30} {'Dom':>4} {'No-Skill':>10} {'Curated':>10} {'Self-Gen':>10} {'Δ':>8}")
    print("-" * 76)
    for st in sorted_st:
        r = results[st]
        label = st.replace("_", " ").title()
        print(f"{label:<30} {r['domain']:>4} {r['no_skill']:>10.3f} {r['curated']:>10.3f} "
              f"{r['self_generated']:>10.3f} {r['delta']:>+8.3f}")
    return {"table": 4, "data": results}


# ---------------------------------------------------------------------------
# Table 5: Per-model results (all 9 models)
# ---------------------------------------------------------------------------
def table_5(df: pd.DataFrame) -> dict:
    results = {}
    for model in sorted(df["model"].unique()):
        dm = df[df["model"] == model]
        row = {"overall": round(float(dm["score"].mean()), 3)}
        for cond in ["no_skill", "curated", "self_generated"]:
            scores = dm[dm["condition"] == cond]["score"].values
            row[cond] = round(float(scores.mean()), 3)
        row["curated_delta"] = round(row["curated"] - row["no_skill"], 3)
        results[model] = row

    # Sort by curated_delta descending
    sorted_models = sorted(results.keys(), key=lambda m: results[m]["curated_delta"], reverse=True)

    print("\n=== Table 5: Per-Model Results (all 9 models) ===")
    print(f"{'Model':<25} {'Overall':>8} {'No-Skill':>10} {'Curated':>10} {'Self-Gen':>10} {'Cur Δ':>8}")
    print("-" * 74)
    for m in sorted_models:
        r = results[m]
        print(f"{m:<25} {r['overall']:>8.3f} {r['no_skill']:>10.3f} {r['curated']:>10.3f} "
              f"{r['self_generated']:>10.3f} {r['curated_delta']:>+8.3f}")
    return {"table": 5, "data": results}


# ---------------------------------------------------------------------------
# Table 6: Cost and interaction overhead (excl. Phi-4)
# ---------------------------------------------------------------------------
def table_6(df: pd.DataFrame) -> dict:
    d = df[df["model"] != "Phi-4"].copy()
    d["total_tokens"] = d["total_input_tokens"] + d["total_output_tokens"]
    results = {}
    for cond in ["no_skill", "curated", "self_generated"]:
        dc = d[d["condition"] == cond]
        tokens = dc["total_tokens"].values
        latency = dc["latency_seconds"].values if "latency_seconds" in dc.columns else np.zeros(len(dc))
        episodes = dc["episodes"].values
        results[cond] = {
            "median_tokens": int(np.median(tokens)),
            "mean_tokens": int(np.mean(tokens)),
            "median_latency": round(float(np.median(latency)), 2),
            "mean_episodes": round(float(np.mean(episodes)), 2),
        }

    print("\n=== Table 6: Cost and Interaction Overhead (excl. Phi-4) ===")
    print(f"{'Condition':<20} {'Med Tokens':>12} {'Mean Tokens':>12} {'Med Latency':>14} {'Mean Episodes':>14}")
    print("-" * 76)
    for cond in ["no_skill", "curated", "self_generated"]:
        r = results[cond]
        label = cond.replace("_", "-")
        print(f"{label:<20} {r['median_tokens']:>12,} {r['mean_tokens']:>12,} "
              f"{r['median_latency']:>14.2f} {r['mean_episodes']:>14.2f}")
    return {"table": 6, "data": results}


# ---------------------------------------------------------------------------
# Table 7: Hermes cross-harness results
# ---------------------------------------------------------------------------
def table_7() -> dict:
    ns = load_jsonl(HERMES_NO_SKILL)
    cur = load_jsonl(HERMES_CURATED)

    # Map subtasks to domains
    for frame in [ns, cur]:
        frame["domain_short"] = frame["sub_task"].map(DOMAIN_MAP)

    results = {"overall": {}, "by_domain": {}, "by_model": {}}

    # Overall
    ns_scores = ns["score"].values
    cur_scores_all = cur["score"].values
    ns_mean = float(ns_scores.mean())
    cur_mean = float(cur_scores_all.mean())
    lo_ns, hi_ns = bootstrap_ci(ns_scores)
    lo_cur, hi_cur = bootstrap_ci(cur_scores_all)
    results["overall"] = {
        "n_no_skill": len(ns), "no_skill": round(ns_mean, 3), "no_skill_ci": fmt_ci(lo_ns - ns_mean, hi_ns - ns_mean),
        "n_curated": len(cur), "curated": round(cur_mean, 3), "curated_ci": fmt_ci(lo_cur - cur_mean, hi_cur - cur_mean),
        "delta": round(cur_mean - ns_mean, 3),
    }

    # By domain
    for dom in ["FA", "PC", "RM"]:
        ns_d = ns[ns["domain_short"] == dom]["score"].values
        cur_d = cur[cur["domain_short"] == dom]["score"].values
        ns_m = float(ns_d.mean())
        cur_m = float(cur_d.mean())
        lo_n, hi_n = bootstrap_ci(ns_d)
        lo_c, hi_c = bootstrap_ci(cur_d)
        results["by_domain"][dom] = {
            "n_no_skill": len(ns_d), "no_skill": round(ns_m, 3),
            "no_skill_ci": f"[{lo_n:.3f}, {hi_n:.3f}]",
            "n_curated": len(cur_d), "curated": round(cur_m, 3),
            "curated_ci": f"[{lo_c:.3f}, {hi_c:.3f}]",
            "delta": round(cur_m - ns_m, 3),
        }

    # By model
    for model in sorted(ns["model"].unique()):
        ns_m_scores = ns[ns["model"] == model]["score"].values
        cur_m_scores = cur[cur["model"] == model]["score"].values
        ns_m = float(ns_m_scores.mean())
        cur_m = float(cur_m_scores.mean())
        lo_n, hi_n = bootstrap_ci(ns_m_scores)
        lo_c, hi_c = bootstrap_ci(cur_m_scores)
        results["by_model"][model] = {
            "n_no_skill": len(ns_m_scores), "no_skill": round(ns_m, 3),
            "no_skill_ci": f"[{lo_n:.3f}, {hi_n:.3f}]",
            "n_curated": len(cur_m_scores), "curated": round(cur_m, 3),
            "curated_ci": f"[{lo_c:.3f}, {hi_c:.3f}]",
            "delta": round(cur_m - ns_m, 3),
        }

    print("\n=== Table 7: Hermes Cross-Harness Results ===")
    print(f"{'Group':<25} {'N(NS)':>6} {'No-Skill':>10} {'N(Cur)':>7} {'Curated':>10} {'Δ':>8}")
    print("-" * 70)
    o = results["overall"]
    print(f"{'Overall':<25} {o['n_no_skill']:>6} {o['no_skill']:>10.3f} {o['n_curated']:>7} {o['curated']:>10.3f} {o['delta']:>+8.3f}")
    print()
    for dom in ["FA", "PC", "RM"]:
        r = results["by_domain"][dom]
        print(f"{DOMAIN_FULL[dom]:<25} {r['n_no_skill']:>6} {r['no_skill']:>10.3f} {r['n_curated']:>7} {r['curated']:>10.3f} {r['delta']:>+8.3f}")
    print()
    for model in sorted(results["by_model"].keys()):
        r = results["by_model"][model]
        print(f"{model:<25} {r['n_no_skill']:>6} {r['no_skill']:>10.3f} {r['n_curated']:>7} {r['curated']:>10.3f} {r['delta']:>+8.3f}")

    return {"table": 7, "data": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TABLE_FUNCS = {2: table_2, 3: table_3, 4: table_4, 5: table_5, 6: table_6}


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper tables from shipped results.")
    parser.add_argument("--table", required=True, help="Table number (2-7) or 'all'")
    parser.add_argument("--format", default="text", choices=["text", "json"], help="Output format")
    args = parser.parse_args()

    tables = [2, 3, 4, 5, 6, 7] if args.table == "all" else [int(args.table)]
    all_results = []

    df = None
    for t in tables:
        if t in TABLE_FUNCS:
            if df is None:
                df = load_main_results()
            result = TABLE_FUNCS[t](df)
        elif t == 7:
            result = table_7()
        else:
            print(f"Unknown table: {t}", file=sys.stderr)
            sys.exit(1)
        all_results.append(result)

    if args.format == "json":
        print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
