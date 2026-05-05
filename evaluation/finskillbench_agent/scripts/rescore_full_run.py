#!/usr/bin/env python3
"""Rescore saved experiment 05 runs using current lib/scorers (no model re-run).

Usage (from ``experiments/zqbok_experiment05``)::

    python3 scripts/rescore_full_run.py full_run_concurrent_v1

Updates each subtask's ``results.jsonl`` and matching ``logs/.../result.json`` scoring blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(EXPT_DIR))

from lib.task_loader import load_all_tasks as load_all_tasks_unfiltered  # noqa: E402
from tasks import score_incomplete_eval, score_task  # noqa: E402

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


def _task_index() -> dict[str, dict]:
    """Full manifest (no FA XBRL panel filter) so rescored rows always resolve."""
    tasks = load_all_tasks_unfiltered()
    return {t["task_id"]: t for t in tasks}


def _extract_final_answer(row: dict, log_path: Path) -> str:
    fa = row.get("final_answer")
    if fa:
        return str(fa)
    for tc in reversed(row.get("tool_calls_log") or []):
        if tc.get("name") == "submit_answer":
            args = tc.get("arguments") or {}
            ans = args.get("answer")
            if ans:
                return str(ans)
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            return str(data.get("final_answer") or "")
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def _update_result_json(log_path: Path, scoring: dict) -> None:
    if not log_path.exists():
        return
    try:
        data = json.loads(log_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    data["scoring"] = {
        "score": scoring.get("score", 0.0),
        "valid_json": scoring.get("valid_json", False),
        "scoring_method": scoring.get("method", "unknown"),
        "scoring_details": scoring.get("details", {}),
    }
    log_path.write_text(json.dumps(data, indent=2, default=str))


def rescore_run(run_id: str, dry_run: bool = False) -> int:
    index = _task_index()
    total = updated = 0
    for sub in SUBTASKS:
        run_dir = EXPT_DIR / "runs" / sub / run_id
        res_path = run_dir / "results.jsonl"
        if not res_path.exists():
            continue
        lines_out = []
        for line in res_path.read_text().splitlines():
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            task_id = row.get("task_id")
            task = index.get(task_id)
            model = row.get("model", "")
            condition = row.get("condition", "")
            run_idx = row.get("run_idx", 0)
            log_path = run_dir / "logs" / model / condition / task_id / f"run_{run_idx}" / "result.json"
            final_answer = _extract_final_answer(row, log_path)
            if not task:
                lines_out.append(json.dumps(row, default=str))
                continue
            if not final_answer:
                score_result = score_incomplete_eval(row.get("error"))
            else:
                score_result = score_task(task, final_answer)
            row["score"] = score_result.get("score", 0.0)
            row["valid_json"] = score_result.get("valid_json", False)
            row["scoring_method"] = score_result.get("method", "unknown")
            row["scoring_details"] = score_result.get("details", {})
            updated += 1
            lines_out.append(json.dumps(row, default=str))
            if not dry_run:
                _update_result_json(log_path, score_result)
        if not dry_run:
            res_path.write_text("\n".join(lines_out) + "\n")
        print(f"  {sub}: rewrote {len(lines_out)} rows")
    print(f"Rescored {updated} / {total} result rows for run_id={run_id}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Rescore experiment 05 results.jsonl with current scorers")
    p.add_argument("run_id", nargs="?", default="full_run_concurrent_v1")
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = p.parse_args()
    sys.exit(rescore_run(args.run_id, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
