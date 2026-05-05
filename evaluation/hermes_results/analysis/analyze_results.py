#!/usr/bin/env python3
"""Analyze a FinSkillBench Hermes run's results.jsonl.

Reports in four sections:
  1. Completeness + dedupe + error audit
  2. Scoring integrity (valid_json, submitted_json_valid, scoring_error,
     partial vs completed)
  3. Score tables with bootstrap 95% CI — overall, by model, by sub-task,
     by (model x sub-task)
  4. Execution mechanics — turns, tool calls, latency by model and sub-task

Usage:
    uv run python experiments/jb_experiment/analysis/analyze_results.py \
        experiments/jb_experiment/runs/2026-04-17_hermes_test12_v1

If multiple runs of the same (run_id, model, task_id, condition) exist
(e.g. from a --resume cycle), the last occurrence per eval_id wins, which
matches how resume actually appends. The dedupe report flags duplicates
so stale rows aren't silently masked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


# ---------------------------------------------------------------------------
# Loading + dedupe
# ---------------------------------------------------------------------------

def load_rows(results_path: Path) -> tuple[list[dict], dict]:
    """Load results.jsonl, dedupe on eval_id (last wins). Return (rows, audit)."""
    raw: list[dict] = []
    with results_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARN: malformed JSON at {results_path.name}:{ln}: {exc}", file=sys.stderr)

    per_id: dict[str, dict] = {}
    dup_counter: Counter[str] = Counter()
    for row in raw:
        eid = row.get("eval_id")
        if not eid:
            continue
        if eid in per_id:
            dup_counter[eid] += 1
        per_id[eid] = row

    rows = list(per_id.values())
    audit = {
        "raw_rows": len(raw),
        "unique_eval_ids": len(per_id),
        "duplicates": sum(dup_counter.values()),
        "duplicate_eval_ids": dict(dup_counter),
    }
    return rows, audit


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 42,
                 ci: float = 0.95) -> tuple[float, float, float]:
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
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1 - alpha))
    return m, low, high


def fmt_ci(m: float, low: float, high: float) -> str:
    return f"{m:.3f} [{low:.3f}, {high:.3f}]"


def md_table(
    headers: list[str],
    body: list[list[str]],
    aligns: list[str] | None = None,
) -> str:
    """Render a GitHub-flavored markdown pipe table. aligns: 'l' | 'r' | 'c' per column."""
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
# Section renderers
# ---------------------------------------------------------------------------

def section_completeness(rows: list[dict], audit: dict) -> str:
    lines = ["## 1. Completeness", ""]
    lines.append(f"- rows read:           {audit['raw_rows']}")
    lines.append(f"- unique eval_ids:     {audit['unique_eval_ids']}")
    lines.append(f"- duplicate rows:      {audit['duplicates']}")
    if audit["duplicate_eval_ids"]:
        lines.append(f"- repeated eval_ids:   {len(audit['duplicate_eval_ids'])} (kept last)")

    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task_id"] for r in rows})
    subtasks = sorted({r["sub_task"] for r in rows})
    expected = len(models) * len(tasks)
    observed = len({(r["model"], r["task_id"]) for r in rows})

    lines.append("")
    lines.append(f"- models ({len(models)}):    {models}")
    lines.append(f"- sub-tasks ({len(subtasks)}): {subtasks}")
    lines.append(f"- tasks ({len(tasks)}):     {len(tasks)}")
    lines.append(f"- matrix expected:     {expected}   (|models| * |tasks|)")
    lines.append(f"- matrix observed:     {observed}")
    missing = expected - observed
    if missing > 0:
        present = {(r["model"], r["task_id"]) for r in rows}
        gaps = [(m, t) for m in models for t in tasks if (m, t) not in present]
        lines.append(f"- matrix missing:      {missing}")
        for m, t in gaps[:10]:
            lines.append(f"    - {m}  {t}")
        if len(gaps) > 10:
            lines.append(f"    ... ({len(gaps) - 10} more)")

    errs = [r for r in rows if r.get("error")]
    lines.append("")
    lines.append(f"- runner errors:       {len(errs)}")
    for r in errs[:5]:
        lines.append(f"    - {r['model']:15s} {r['task_id']}  -> {r['error']}")

    # Domain breakdown table
    domains = sorted({r.get("skill_domain", "unknown") for r in rows})
    lines.append("")
    lines.append(f"- domains ({len(domains)}):   {domains}")
    lines.append("")
    dom_body: list[list[str]] = []
    for dom in domains:
        dom_rows = [r for r in rows if r.get("skill_domain") == dom]
        dom_tasks = len({r["task_id"] for r in dom_rows})
        dom_subtasks = sorted({r["sub_task"] for r in dom_rows})
        dom_observed = len({(r["model"], r["task_id"]) for r in dom_rows})
        dom_expected = len(models) * dom_tasks
        dom_body.append([
            dom,
            str(dom_tasks),
            str(len(dom_subtasks)),
            ", ".join(dom_subtasks),
            str(dom_observed),
            str(dom_expected),
        ])
    lines.append(md_table(
        ["domain", "tasks", "subtasks_n", "subtasks", "matrix_obs", "matrix_exp"],
        dom_body,
        aligns=["l", "r", "r", "l", "r", "r"],
    ))
    return "\n".join(lines)


def section_scoring_integrity(rows: list[dict]) -> str:
    lines = ["## 2. Scoring integrity", ""]
    n = len(rows)
    submitted_ok = sum(1 for r in rows if r.get("submitted_json_valid"))
    valid_json = sum(1 for r in rows if r.get("valid_json"))
    via_tool = sum(1 for r in rows if r.get("submitted_via_tool"))
    score_err = [r for r in rows if r.get("scoring_error")]
    completed = sum(1 for r in rows if r.get("completed"))
    partial = sum(1 for r in rows if r.get("partial"))
    leak_ok = sum(1 for r in rows if r.get("leakage_passed", True))

    def pct(x: int) -> str:
        return f"{x}/{n} ({100 * x / n:.1f}%)" if n else f"{x}/0"

    lines.append(f"- submitted via tool:   {pct(via_tool)}")
    lines.append(f"- submitted JSON valid: {pct(submitted_ok)}")
    lines.append(f"- scorer valid_json:    {pct(valid_json)}")
    lines.append(f"- scoring errors:       {len(score_err)}")
    for r in score_err[:5]:
        lines.append(f"    - {r['model']:15s} {r['task_id']}  -> {r['scoring_error']}")
    lines.append(f"- completed (agent exited loop naturally, decided to stop): {pct(completed)}")
    lines.append(f"- partial (hit turn/tool budget, MAX_TURN): {pct(partial)}")
    lines.append(f"- leakage_passed (TODO): {pct(leak_ok)}")

    # Per-domain integrity breakdown
    domains = sorted({r.get("skill_domain", "unknown") for r in rows})
    if len(domains) > 1:
        lines.append("")
        lines.append("### By domain")
        lines.append("")
        dom_body: list[list[str]] = []
        for dom in domains:
            dr = [r for r in rows if r.get("skill_domain") == dom]
            dn = len(dr)
            def dpct(x: int) -> str:
                return f"{x}/{dn} ({100 * x / dn:.0f}%)" if dn else f"{x}/0"
            dom_body.append([
                dom,
                str(dn),
                dpct(sum(1 for r in dr if r.get("submitted_via_tool"))),
                dpct(sum(1 for r in dr if r.get("submitted_json_valid"))),
                dpct(sum(1 for r in dr if r.get("completed"))),
                dpct(sum(1 for r in dr if r.get("partial"))),
                str(sum(1 for r in dr if r.get("scoring_error"))),
            ])
        lines.append(md_table(
            ["domain", "n", "via_tool", "json_valid", "completed", "partial", "score_errs"],
            dom_body,
            aligns=["l", "r", "r", "r", "r", "r", "r"],
        ))
    return "\n".join(lines)


def score_table(rows: list[dict], group_key) -> list[tuple[str, int, float, float, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[group_key(r)].append(float(r.get("score", 0.0)))
    out = []
    for key in sorted(buckets):
        vals = buckets[key]
        m, lo, hi = bootstrap_ci(vals)
        out.append((key, len(vals), m, lo, hi))
    return out


def render_score_table(title: str, rows: list[tuple]) -> str:
    body = [
        [key, str(n), f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]"]
        for key, n, m, lo, hi in rows
    ]
    table = md_table(
        ["group", "n", "mean", "95% CI"],
        body,
        aligns=["l", "r", "r", "l"],
    )
    return f"{title}\n\n{table}"


def section_scores(rows: list[dict]) -> str:
    overall    = score_table(rows, lambda r: "all")
    by_model   = score_table(rows, lambda r: r["model"])
    by_domain  = score_table(rows, lambda r: r.get("skill_domain", "unknown"))
    by_subtask = score_table(rows, lambda r: r["sub_task"])
    by_model_domain  = score_table(rows, lambda r: f"{r['model']} / {r.get('skill_domain', '?')}")
    by_domain_subtask = score_table(rows, lambda r: f"{r.get('skill_domain', '?')} / {r['sub_task']}")
    by_cell    = score_table(rows, lambda r: f"{r['model']} / {r['sub_task']}")

    parts = ["## 3. Scores (mean score, bootstrap 95% CI, n=10k)", ""]
    parts.append(render_score_table("### Overall", overall))
    parts.append("")
    parts.append(render_score_table("### By model", by_model))
    parts.append("")
    parts.append(render_score_table("### By domain", by_domain))
    parts.append("")
    parts.append(render_score_table("### By (model x domain)", by_model_domain))
    parts.append("")
    parts.append(render_score_table("### By sub-task", by_subtask))
    parts.append("")
    parts.append(render_score_table("### By (domain x sub-task)", by_domain_subtask))
    parts.append("")
    parts.append(render_score_table("### By (model x sub-task)", by_cell))
    return "\n".join(parts)


def _agg_numeric(rows: list[dict], field: str) -> tuple[float, float, float]:
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    if not vals:
        return 0.0, 0.0, 0.0
    return mean(vals), median(vals), max(vals)


def section_mechanics(rows: list[dict]) -> str:
    body: list[list[str]] = []

    def add(label: str, subset: list[dict]) -> None:
        if not subset:
            return
        ta, tm, tx = _agg_numeric(subset, "turns")
        ca, cm, cx = _agg_numeric(subset, "tool_calls_total")
        la, lm, lx = _agg_numeric(subset, "latency_seconds")
        body.append([
            label,
            str(len(subset)),
            f"{ta:.1f} / {tm:.0f} / {tx:.0f}",
            f"{ca:.1f} / {cm:.0f} / {cx:.0f}",
            f"{la:.1f} / {lm:.1f} / {lx:.1f}",
        ])

    add("all", rows)
    for model in sorted({r["model"] for r in rows}):
        add(f"model={model}", [r for r in rows if r["model"] == model])
    for dom in sorted({r.get("skill_domain", "unknown") for r in rows}):
        add(f"domain={dom}", [r for r in rows if r.get("skill_domain") == dom])
    for sub in sorted({r["sub_task"] for r in rows}):
        add(f"sub_task={sub}", [r for r in rows if r["sub_task"] == sub])

    lines = ["## 4. Execution mechanics", ""]
    lines.append(md_table(
        ["group", "n", "turns (avg / med / max)", "tools (avg / med / max)", "latency s (avg / med / max)"],
        body,
        aligns=["l", "r", "r", "r", "r"],
    ))

    # Summary table: median/mean tokens, median latency, mean turns — by overall, model, domain
    def _row_total_tokens(row: dict) -> float | None:
        u = row.get("usage")
        if not isinstance(u, dict):
            return None
        total = u.get("total_tokens")
        if total is not None:
            return float(total)
        inp, out = u.get("input_tokens"), u.get("output_tokens")
        if inp is not None and out is not None:
            return float(inp) + float(out)
        return None

    def _summary_row(label: str, subset: list[dict]) -> list[str]:
        toks  = [v for r in subset if (v := _row_total_tokens(r)) is not None]
        lats  = [float(r["latency_seconds"]) for r in subset if r.get("latency_seconds") is not None]
        turns = [float(r["turns"]) for r in subset if r.get("turns") is not None]
        fmt_k = lambda v: f"{v:,.0f}" if v is not None else "—"
        med_tok  = fmt_k(median(toks)  if toks  else None)
        mean_tok = fmt_k(mean(toks)    if toks  else None)
        med_lat  = f"{median(lats):.1f}"  if lats  else "—"
        mean_trn = f"{mean(turns):.1f}"   if turns else "—"
        n_tok = f"{len(toks)}/{len(subset)}"
        return [label, str(len(subset)), n_tok, med_tok, mean_tok, med_lat, mean_trn]

    sum_body: list[list[str]] = []
    sum_body.append(_summary_row("all", rows))
    for model in sorted({r["model"] for r in rows}):
        sum_body.append(_summary_row(f"model={model}", [r for r in rows if r["model"] == model]))
    for dom in sorted({r.get("skill_domain", "unknown") for r in rows}):
        sum_body.append(_summary_row(f"domain={dom}", [r for r in rows if r.get("skill_domain") == dom]))

    lines.append("")
    lines.append("### Summary: tokens, latency, turns")
    lines.append("")
    lines.append(md_table(
        ["group", "n", "n_tok", "median_tokens", "mean_tokens", "median_latency_s", "mean_turns"],
        sum_body,
        aligns=["l", "r", "r", "r", "r", "r", "r"],
    ))

    # Tool usage histogram
    lines.append("")
    lines.append("### Tool usage (# rows that called each tool)")
    lines.append("")
    tool_counts: Counter[str] = Counter()
    for r in rows:
        for t in r.get("tools_used") or []:
            tool_counts[t] += 1
    # Ensure Hermes skill-discovery tools always appear even when unused,
    # so a "skills never loaded" outcome is visible rather than silent.
    for t in ("skills_list", "skill_view"):
        tool_counts.setdefault(t, 0)
    tool_body = [[tool, str(c)] for tool, c in tool_counts.most_common()]
    lines.append(md_table(["tool", "n_rows"], tool_body, aligns=["l", "r"]))

    # Tool usage by model (model x tool matrix)
    lines.append("")
    lines.append("### Tool usage by model (# rows per (model, tool))")
    lines.append("")
    models = sorted({r["model"] for r in rows})
    tool_cols = [t for t, _ in tool_counts.most_common()]
    per_cell: Counter[tuple[str, str]] = Counter()
    for r in rows:
        m = r["model"]
        for t in r.get("tools_used") or []:
            per_cell[(m, t)] += 1
    matrix_body: list[list[str]] = []
    for m in models:
        matrix_body.append([m] + [str(per_cell.get((m, t), 0)) for t in tool_cols])
    matrix_body.append(
        ["**total**"] + [str(sum(per_cell.get((mm, t), 0) for mm in models)) for t in tool_cols]
    )
    headers = ["model"] + tool_cols
    aligns = ["l"] + ["r"] * len(tool_cols)
    lines.append(md_table(headers, matrix_body, aligns=aligns))

    # Token usage by model
    lines.append("")
    lines.append("### Token usage by model (avg per episode; totals where marked)")
    lines.append("")
    _TOKEN_FIELDS = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )

    def _usage_val(row: dict, field: str) -> float | None:
        u = row.get("usage")
        if not isinstance(u, dict):
            return None
        v = u.get(field)
        return float(v) if v is not None else None

    tok_body: list[list[str]] = []
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        n_data = sum(1 for r in mrows if isinstance(r.get("usage"), dict) and r["usage"].get("input_tokens") is not None)

        def avg_tok(field: str) -> str:
            vals = [v for r in mrows if (v := _usage_val(r, field)) is not None]
            return f"{mean(vals):,.0f}" if vals else "—"

        def total_cost() -> str:
            vals = [float(u["estimated_cost_usd"])
                    for r in mrows
                    if isinstance(r.get("usage"), dict)
                    and r["usage"].get("estimated_cost_usd") is not None
                    for u in [r["usage"]]]
            return f"${sum(vals):.4f}" if vals else "—"

        tok_body.append([
            m,
            f"{n_data}/{len(mrows)}",
            avg_tok("input_tokens"),
            avg_tok("output_tokens"),
            avg_tok("reasoning_tokens"),
            avg_tok("cache_read_tokens"),
            avg_tok("cache_write_tokens"),
            total_cost(),
        ])

    tok_headers = [
        "model", "n_with_data",
        "input_tok (avg)", "output_tok (avg)", "reasoning_tok (avg)",
        "cache_read (avg)", "cache_write (avg)",
        "cost_usd (total)",
    ]
    tok_aligns = ["l", "r", "r", "r", "r", "r", "r", "r"]
    lines.append(md_table(tok_headers, tok_body, aligns=tok_aligns))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def analyze(run_dir: Path, out_path: Path | None) -> str:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"results.jsonl not found at {results_path}")
    rows, audit = load_rows(results_path)

    parts = [
        f"# Analysis: {run_dir.name}",
        f"source: {results_path}",
        "",
        section_completeness(rows, audit),
        "",
        section_scoring_integrity(rows),
        "",
        section_scores(rows),
        "",
        section_mechanics(rows),
        "",
    ]
    report = "\n".join(parts)

    if out_path:
        out_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="e.g. experiments/jb_experiment/runs/2026-04-17_hermes_test12_v1")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the report to this path in addition to stdout. "
                        "Defaults to <run_dir>/analysis.md.")
    args = p.parse_args()
    out = args.out if args.out is not None else args.run_dir / "analysis.md"
    report = analyze(args.run_dir, out)
    print(report)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main()
