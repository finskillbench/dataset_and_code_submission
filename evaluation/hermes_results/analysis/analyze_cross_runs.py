#!/usr/bin/env python3
"""Compare two FinSkillBench Hermes runs — typically no_skill vs curated.

Computes mean score deltas (run_b − run_a) with bootstrap 95% CIs across:
  1. Coverage — task/model/domain overlap between the two runs
  2. Overall
  3. By domain
  4. By model
  5. By sub-task
  6. By (model × domain)
  7. By (model × sub-task)

A † marker flags groups where the 95% CIs do not overlap (non-overlapping
bands are a heuristic indicator of a likely meaningful difference).

Scores are computed independently per run over each group — no pairing is
enforced. The coverage section reports shared vs run-only task_ids so you
can judge comparability.

Usage:
    uv run python experiments/jb_experiment/analysis/analyze_cross_runs.py \\
        experiments/jb_experiment/runs/2026-04-20_rm_noskills_test20 \\
        experiments/jb_experiment/runs/2026-04-20_rm_skills_test20 \\
        [--label-a no_skill] [--label-b curated] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading + dedupe  (mirrors analyze_results.py)
# ---------------------------------------------------------------------------

def load_rows(results_path: Path) -> list[dict]:
    """Load results.jsonl, dedupe on eval_id (last wins)."""
    raw: list[dict] = []
    with results_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARN: malformed JSON at line {ln}: {exc}", file=sys.stderr)
    per_id: dict[str, dict] = {}
    for row in raw:
        eid = row.get("eval_id")
        if eid:
            per_id[eid] = row
    return list(per_id.values())


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    if arr.size < 2:
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    boots = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return m, lo, hi


def cis_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return hi_a >= lo_b and hi_b >= lo_a


def md_table(
    headers: list[str],
    body: list[list[str]],
    aligns: list[str] | None = None,
) -> str:
    n = len(headers)
    aligns = aligns or ["l"] * n
    sep_map = {"l": ":---", "r": "---:", "c": ":---:"}
    sep = [sep_map[a] for a in aligns]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in body:
        cells = [str(c) for c in row]
        if len(cells) < n:
            cells += [""] * (n - len(cells))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _scores_for(rows: list[dict], key_fn) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(float(r.get("score", 0.0)))
    return dict(buckets)


def compare_groups(
    rows_a: list[dict],
    rows_b: list[dict],
    key_fn,
    label_a: str,
    label_b: str,
) -> str:
    """Build a markdown comparison table for a given grouping key."""
    ba = _scores_for(rows_a, key_fn)
    bb = _scores_for(rows_b, key_fn)
    all_keys = sorted(set(ba) | set(bb))

    headers = [
        "group",
        f"n_{label_a}", f"mean_{label_a}", f"95% CI",
        f"n_{label_b}", f"mean_{label_b}", f"95% CI",
        "Δ (b−a)", "",
    ]
    aligns = ["l", "r", "r", "l", "r", "r", "l", "r", "c"]
    body: list[list[str]] = []

    for key in all_keys:
        vals_a = ba.get(key, [])
        vals_b = bb.get(key, [])
        ma, lo_a, hi_a = bootstrap_ci(vals_a)
        mb, lo_b, hi_b = bootstrap_ci(vals_b)
        delta = mb - ma
        sig = "" if (not vals_a or not vals_b or cis_overlap(lo_a, hi_a, lo_b, hi_b)) else "†"
        delta_str = f"{delta:+.3f}"
        body.append([
            key,
            str(len(vals_a)), f"{ma:.3f}", f"[{lo_a:.3f}, {hi_a:.3f}]",
            str(len(vals_b)), f"{mb:.3f}", f"[{lo_b:.3f}, {hi_b:.3f}]",
            delta_str, sig,
        ])

    return md_table(headers, body, aligns)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def section_coverage(
    rows_a: list[dict],
    rows_b: list[dict],
    label_a: str,
    label_b: str,
    run_a: Path,
    run_b: Path,
) -> str:
    lines = ["## 1. Coverage", ""]

    def _summarise(rows: list[dict], label: str, run: Path) -> dict:
        tasks = {r["task_id"] for r in rows}
        models = sorted({r["model"] for r in rows})
        domains = sorted({r.get("skill_domain", "unknown") for r in rows})
        subtasks = sorted({r["sub_task"] for r in rows})
        conditions = sorted({r.get("condition", "?") for r in rows})
        errors = sum(1 for r in rows if r.get("error"))
        return dict(
            label=label, run=run, rows=len(rows), tasks=tasks,
            models=models, domains=domains, subtasks=subtasks,
            conditions=conditions, errors=errors,
        )

    sa = _summarise(rows_a, label_a, run_a)
    sb = _summarise(rows_b, label_b, run_b)

    shared_tasks = sa["tasks"] & sb["tasks"]
    only_a = sa["tasks"] - sb["tasks"]
    only_b = sb["tasks"] - sa["tasks"]

    for s in (sa, sb):
        lines.append(f"### {s['label']}  (`{s['run'].name}`)")
        lines.append(f"- rows:      {s['rows']}")
        lines.append(f"- tasks:     {len(s['tasks'])}")
        lines.append(f"- models:    {s['models']}")
        lines.append(f"- domains:   {s['domains']}")
        lines.append(f"- subtasks:  {s['subtasks']}")
        lines.append(f"- condition: {s['conditions']}")
        lines.append(f"- errors:    {s['errors']}")
        lines.append("")

    lines.append(f"**shared tasks:** {len(shared_tasks)}")
    lines.append(f"**only in {label_a}:** {len(only_a)}")
    lines.append(f"**only in {label_b}:** {len(only_b)}")
    if only_a:
        for t in sorted(only_a)[:5]:
            lines.append(f"  - {t}")
        if len(only_a) > 5:
            lines.append(f"  ... ({len(only_a) - 5} more)")
    if only_b:
        for t in sorted(only_b)[:5]:
            lines.append(f"  - {t}")
        if len(only_b) > 5:
            lines.append(f"  ... ({len(only_b) - 5} more)")
    return "\n".join(lines)


def _score_section(
    title: str,
    rows_a: list[dict],
    rows_b: list[dict],
    key_fn,
    label_a: str,
    label_b: str,
) -> str:
    table = compare_groups(rows_a, rows_b, key_fn, label_a, label_b)
    return f"{title}\n\n{table}\n\n† 95% CIs do not overlap."


def section_scores(
    rows_a: list[dict],
    rows_b: list[dict],
    label_a: str,
    label_b: str,
) -> str:
    parts = [
        "## 2. Overall",
        "",
        compare_groups(rows_a, rows_b, lambda r: "all", label_a, label_b),
        "",
        "† 95% CIs do not overlap.",
        "",
        _score_section("## 3. By domain", rows_a, rows_b,
                       lambda r: r.get("skill_domain", "unknown"), label_a, label_b),
        _score_section("## 4. By model", rows_a, rows_b,
                       lambda r: r["model"], label_a, label_b),
        _score_section("## 5. By sub-task", rows_a, rows_b,
                       lambda r: r["sub_task"], label_a, label_b),
        _score_section("## 6. By (model × domain)", rows_a, rows_b,
                       lambda r: f"{r['model']} / {r.get('skill_domain', '?')}",
                       label_a, label_b),
        _score_section("## 7. By (model × sub-task)", rows_a, rows_b,
                       lambda r: f"{r['model']} / {r['sub_task']}",
                       label_a, label_b),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def compare(
    run_a: Path,
    run_b: Path,
    label_a: str,
    label_b: str,
    out_path: Path | None,
) -> str:
    for run in (run_a, run_b):
        if not (run / "results.jsonl").exists():
            raise SystemExit(f"results.jsonl not found in {run}")

    rows_a = load_rows(run_a / "results.jsonl")
    rows_b = load_rows(run_b / "results.jsonl")

    if not rows_a:
        raise SystemExit(f"No rows loaded from {run_a}")
    if not rows_b:
        raise SystemExit(f"No rows loaded from {run_b}")

    parts = [
        f"# Cross-run comparison: `{label_a}` vs `{label_b}`",
        f"",
        f"- **{label_a}**: `{run_a}`",
        f"- **{label_b}**: `{run_b}`",
        f"- Δ = {label_b} − {label_a}  (positive = {label_b} is better)",
        "",
        section_coverage(rows_a, rows_b, label_a, label_b, run_a, run_b),
        "",
        section_scores(rows_a, rows_b, label_a, label_b),
        "",
    ]
    report = "\n".join(parts)

    if out_path:
        out_path.write_text(report, encoding="utf-8")
        print(f"[written to {out_path}]", file=sys.stderr)

    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_a", type=Path,
                   help="Baseline run directory (e.g. *_noskills_*)")
    p.add_argument("run_b", type=Path,
                   help="Treatment run directory (e.g. *_skills_*)")
    p.add_argument("--label-a", default=None,
                   help="Human label for run_a. Defaults to run dir name.")
    p.add_argument("--label-b", default=None,
                   help="Human label for run_b. Defaults to run dir name.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the report to this path in addition to stdout. "
                        "Defaults to <run_b>/cross_run_comparison.md.")
    args = p.parse_args()

    label_a = args.label_a or args.run_a.name
    label_b = args.label_b or args.run_b.name
    out = args.out if args.out is not None else args.run_b / "cross_run_comparison.md"

    report = compare(args.run_a, args.run_b, label_a, label_b, out)
    print(report)


if __name__ == "__main__":
    main()
