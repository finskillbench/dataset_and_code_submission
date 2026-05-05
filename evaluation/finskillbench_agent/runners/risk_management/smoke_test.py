#!/usr/bin/env python3
"""Smoke test for all 4 risk management subtask runners.

Runs 1 task × 3 conditions × 4 subtasks = 12 evals with a single model (gpt-4.1).
Validates: data loading, skill pickup, scoring logic, end-to-end pipeline.

Usage:
    python experiments/zqbok_experiment05/runners/risk_management/smoke_test.py
    python experiments/zqbok_experiment05/runners/risk_management/smoke_test.py --model gpt-5.4
    python experiments/zqbok_experiment05/runners/risk_management/smoke_test.py --model all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPT_DIR))

from _base import (
    MODELS, DOMAIN, run_single_eval, _load_completed, _eval_key,
)
from tasks import load_subtask_tasks, CONDITIONS

SUB_TASKS = ["constraint_monitoring", "risk_identification", "stress_testing", "risk_remediation"]
RUN_ID = "rm_smoke_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="RM smoke test — 1 task per subtask, all conditions")
    parser.add_argument("--model", nargs="*", dest="models", default=["gpt-4.1"])
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--eval-retries", type=int, default=3,
                        help="Extra full-eval attempts on transient failures (0 = off).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()

    models = args.models
    if models == ["all"]:
        models = list(MODELS.keys())

    # Load 1 task per subtask
    task_map: dict[str, dict] = {}
    for st in SUB_TASKS:
        tasks = load_subtask_tasks(DOMAIN, st)
        if not tasks:
            print(f"  WARNING: no tasks found for {st}, skipping")
            continue
        task_map[st] = tasks[0]
        print(f"  {st}: loaded {len(tasks)} tasks, using '{tasks[0]['task_id']}'")

    if not task_map:
        print("ERROR: no tasks loaded for any subtask")
        sys.exit(1)

    conditions = CONDITIONS  # no_skill, curated, self_generated
    total = len(task_map) * len(models) * len(conditions)
    print(f"\nSmoke test: {len(task_map)} subtasks × {len(models)} models × {len(conditions)} conditions = {total} evals")

    run_dir = EXPT_DIR / "runs" / "rm_smoke" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(run_dir)
    if completed:
        print(f"  Resuming: {len(completed)} already completed")

    results_file = run_dir / "results.jsonl"
    all_results: list[dict] = []
    done = 0
    failed = 0
    started = time.time()

    for model_name in models:
        model_info = MODELS.get(model_name, {})
        litellm_model = model_info.get("litellm", model_name)

        for st, task in task_map.items():
            for condition in conditions:
                key = _eval_key(model_name, condition, task["task_id"], 0)
                if key in completed:
                    print(f"  SKIP {model_name}/{condition}/{st} (already done)")
                    continue

                tag = f"{model_name}/{condition}/{st}"
                print(f"\n  [{done+1}/{total}] Running {tag} ...")

                try:
                    row = run_single_eval(
                        task=task, condition=condition,
                        model_name=model_name, litellm_model=litellm_model,
                        run_idx=0, run_dir=run_dir, max_turns=args.max_turns,
                        eval_retries=args.eval_retries,
                    )
                except Exception as exc:
                    row = {
                        "model": model_name, "domain": DOMAIN,
                        "condition": condition, "task_id": task["task_id"],
                        "sub_task": st, "run_idx": 0, "score": 0.0,
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }
                    failed += 1

                with open(results_file, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")

                all_results.append(row)
                done += 1

                score = row.get("score", 0.0)
                valid = "✓" if row.get("valid_json") else "✗"
                eps = row.get("episodes", "?")
                skills = row.get("skills_loaded", [])
                lat = row.get("latency_seconds", "?")
                err = row.get("error", "")
                if err:
                    print(f"    → FAILED: {err[:120]}")
                else:
                    print(f"    → score={score:.4f} json={valid} eps={eps} skills={skills} lat={lat}s")

    elapsed = time.time() - started

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SMOKE TEST COMPLETE: {done} evals in {elapsed/60:.1f}m ({failed} failed)")
    print(f"  Results: {results_file}")
    print(f"{'='*70}")

    # Per-subtask summary
    for st in SUB_TASKS:
        st_results = [r for r in all_results if r.get("sub_task") == st]
        if not st_results:
            continue
        scores = [r.get("score", 0.0) for r in st_results]
        valid_count = sum(1 for r in st_results if r.get("valid_json"))
        err_count = sum(1 for r in st_results if r.get("error"))
        avg = sum(scores) / len(scores) if scores else 0
        print(f"\n  {st}:")
        print(f"    Evals: {len(st_results)}, Valid JSON: {valid_count}/{len(st_results)}, "
              f"Errors: {err_count}, Avg score: {avg:.3f}")
        for r in st_results:
            cond = r.get("condition", "?")
            model = r.get("model", "?")
            sc = r.get("score", 0.0)
            vj = "✓" if r.get("valid_json") else "✗"
            ep = r.get("episodes", "?")
            sk = r.get("skills_loaded", [])
            lt = r.get("latency_seconds", "?")
            er = r.get("error", "")
            if er:
                print(f"      {model}/{cond}: FAILED — {er[:80]}")
            else:
                print(f"      {model}/{cond}: score={sc:.4f} json={vj} eps={ep} skills={sk} lat={lt}s")


if __name__ == "__main__":
    main()
