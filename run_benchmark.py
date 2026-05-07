#!/usr/bin/env python3
"""Unified CLI for running FinSkillBench evaluations.

Usage:
    # Smoke test: 1 task per subtask, cheapest model, curated condition (~$2)
    python3.12 run_benchmark.py --smoke

    # Specific slice
    python3.12 run_benchmark.py --model gpt-4.1 --condition curated \\
        --subtask unconstrained_optimization --limit 3

    # Re-score an existing run
    python3.12 run_benchmark.py --score-only --run-id <id>

    # Full paper replication (hours, ~$500+)
    python3.12 run_benchmark.py --models all --conditions all --subtasks all \\
        --workers 16 --max-turns 12 --run-id full_replication

Requires: python3.12 -m pip install -r requirements.txt  (litellm, cvxpy, pandas, etc.)
API keys: set OPENAI_API_KEY, AZURE_AI_API_KEY, etc. in .env or environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
SUBMISSION_ROOT = Path(__file__).resolve().parent
EVAL_DIR = SUBMISSION_ROOT / "evaluation" / "finskillbench_agent"
sys.path.insert(0, str(EVAL_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(SUBMISSION_ROOT / ".env", override=False)

from tasks import load_subtask_tasks, score_task, build_instruction  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry: short name → LiteLLM route string
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "gpt-5.4": {"litellm": "openai/gpt-5.4", "tier": "frontier"},
    "gpt-4.1": {"litellm": "openai/gpt-4.1", "tier": "frontier"},
    "claude-sonnet-4.6": {"litellm": "openai/claude-sonnet-4.6", "tier": "frontier"},
    "gemini-2.5-pro": {"litellm": "vertex_ai/gemini-2.5-pro", "tier": "frontier"},
    "grok-4": {"litellm": "openrouter/x-ai/grok-4", "tier": "frontier"},
    "DeepSeek-V3.2": {"litellm": "openrouter/deepseek/deepseek-chat-v3-0324", "tier": "open_weight"},
    "Phi-4": {"litellm": "openrouter/microsoft/phi-4", "tier": "open_weight"},
    "glm-5.1": {"litellm": "openrouter/thudm/glm-5.1", "tier": "open_weight"},
    "gemini-3.1-flash-lite": {"litellm": "vertex_ai/gemini-3.1-flash-lite", "tier": "frontier"},
}

SUBTASK_DOMAIN = {
    "normalization": "fundamental_analysis",
    "earnings_quality": "fundamental_analysis",
    "driver_decomposition": "fundamental_analysis",
    "unconstrained_optimization": "portfolio_construction",
    "constrained_optimization": "portfolio_construction",
    "rebalancing": "portfolio_construction",
    "black_litterman": "portfolio_construction",
    "tool_use_parameterization": "portfolio_construction",
    "constraint_monitoring": "risk_management",
    "risk_identification": "risk_management",
    "risk_remediation": "risk_management",
    "stress_testing": "risk_management",
}

ALL_SUBTASKS = list(SUBTASK_DOMAIN.keys())
ALL_CONDITIONS = ["no_skill", "curated", "self_generated"]


def run_single(task, condition, model_name, litellm_model, run_dir, max_turns):
    """Run one evaluation using the function-calling agent loop."""
    from runners.fundamental_analysis._base import run_single_eval as fa_eval
    from runners.portfolio_construction._base import run_single_eval as pc_eval
    from runners.risk_management._base import run_single_eval as rm_eval

    domain = task.get("skill", task.get("domain", ""))
    dispatch = {
        "fundamental_analysis": fa_eval,
        "portfolio_construction": pc_eval,
        "risk_management": rm_eval,
    }
    eval_fn = dispatch.get(domain)
    if eval_fn is None:
        return {"error": f"Unknown domain: {domain}", "score": 0.0, "task_id": task["task_id"]}

    return eval_fn(
        task=task,
        condition=condition,
        model_name=model_name,
        litellm_model=litellm_model,
        run_idx=0,
        run_dir=run_dir,
        max_turns=max_turns,
    )


def score_only(run_id: str):
    """Re-score an existing run and print summary."""
    import pandas as pd
    run_dir = SUBMISSION_ROOT / "runs" / run_id
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"No results found at {results_path}")
        sys.exit(1)
    rows = [json.loads(l) for l in open(results_path)]
    df = pd.DataFrame(rows)
    print(f"\n=== Run {run_id}: {len(df)} evaluations ===")
    print(f"Mean score: {df['score'].mean():.3f}")
    print(f"\nBy condition:")
    print(df.groupby("condition")["score"].mean().to_string())
    print(f"\nBy subtask:")
    print(df.groupby("sub_task")["score"].mean().sort_values(ascending=False).to_string())


def main():
    parser = argparse.ArgumentParser(description="FinSkillBench unified evaluation CLI.")
    parser.add_argument("--model", "--models", nargs="*", dest="models",
                        help="Model name(s) or 'all'. Default: gpt-4.1")
    parser.add_argument("--condition", "--conditions", nargs="*", dest="conditions",
                        help="Condition(s) or 'all'. Default: curated")
    parser.add_argument("--subtask", "--subtasks", nargs="*", dest="subtasks",
                        help="Subtask(s) or 'all'. Default: all")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max tasks per subtask (0 = all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers")
    parser.add_argument("--max-turns", type=int, default=12,
                        help="Max LLM turns per task")
    parser.add_argument("--run-id", default=None,
                        help="Custom run ID")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 task/subtask, gpt-4.1, curated")
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score existing run (requires --run-id)")
    args = parser.parse_args()

    if args.score_only:
        if not args.run_id:
            print("--score-only requires --run-id")
            sys.exit(1)
        score_only(args.run_id)
        return

    # Resolve parameters
    if args.smoke:
        models = ["gpt-4.1"]
        conditions = ["curated"]
        subtasks = ALL_SUBTASKS
        limit = 1
    else:
        models = args.models or ["gpt-4.1"]
        if models == ["all"]:
            models = list(MODEL_REGISTRY.keys())
        conditions = args.conditions or ["curated"]
        if conditions == ["all"]:
            conditions = ALL_CONDITIONS
        subtasks = args.subtasks or ALL_SUBTASKS
        if subtasks == ["all"]:
            subtasks = ALL_SUBTASKS
        limit = args.limit

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SUBMISSION_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    # Build work items
    work = []
    for subtask in subtasks:
        domain = SUBTASK_DOMAIN.get(subtask)
        if not domain:
            print(f"Unknown subtask: {subtask}")
            continue
        tasks = load_subtask_tasks(domain, subtask)
        if limit > 0:
            tasks = tasks[:limit]
        for task in tasks:
            for model in models:
                for condition in conditions:
                    work.append((task, condition, model))

    total = len(work)
    print(f"\n{'='*60}")
    print(f"  FinSkillBench Evaluation")
    print(f"  Models:     {models}")
    print(f"  Conditions: {conditions}")
    print(f"  Subtasks:   {len(subtasks)}")
    print(f"  Total evals: {total}")
    print(f"  Run ID:     {run_id}")
    print(f"  Output:     {results_path}")
    print(f"{'='*60}\n")

    completed = 0
    errors = 0
    start = time.time()

    with open(results_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for task, condition, model in work:
                reg = MODEL_REGISTRY.get(model, {"litellm": model, "tier": "unknown"})
                fut = pool.submit(
                    run_single, task, condition, model, reg["litellm"],
                    run_dir, args.max_turns
                )
                futures[fut] = (task["task_id"], model, condition)

            for fut in as_completed(futures):
                tid, model, cond = futures[fut]
                completed += 1
                try:
                    result = fut.result()
                    result["eval_attempt"] = 1
                    out_f.write(json.dumps(result, default=str) + "\n")
                    out_f.flush()
                    score = result.get("score", 0.0)
                    elapsed = time.time() - start
                    print(f"  [{completed}/{total}] {tid} | {model} | {cond} → {score:.3f}  "
                          f"({elapsed:.0f}s elapsed)")
                except Exception as e:
                    errors += 1
                    print(f"  [{completed}/{total}] {tid} | {model} | {cond} → ERROR: {e}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Done: {completed} evals in {elapsed:.0f}s ({errors} errors)")
    print(f"  Results: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
