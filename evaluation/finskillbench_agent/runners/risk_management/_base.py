#!/usr/bin/env python3
"""Shared runner logic for risk management subtask experiments.

Each subtask script (run_constraint_monitoring.py, run_risk_identification.py, etc.)
calls ``run_subtask_experiment()`` with the appropriate sub_task name.
Tasks within each (model, condition) group run in parallel via ThreadPoolExecutor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPT_DIR = Path(__file__).resolve().parents[2]

# Only exp05 on sys.path — standalone, no exp02 or scripts/ dependency
sys.path.insert(0, str(EXPT_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# ── Environment setup (Azure / Vertex / OpenRouter) ──────────────────────

_azure_key = os.environ.get("AZURE_AI_API_KEY", "")
if _azure_key:
    os.environ.setdefault("AZURE_FOUNDRY_API_KEY", _azure_key)
    os.environ.setdefault("AZURE_FOUNDRY_BASE_URL",
                          "https://research-collab.services.ai.azure.com/openai/v1/")
    os.environ.setdefault("AZURE_OPENAI_API_KEY", _azure_key)
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT",
                          "https://research-collab.cognitiveservices.azure.com/")
    os.environ.setdefault("AZURE_API_VERSION", "2024-12-01-preview")
    os.environ.setdefault("OPENAI_API_KEY", _azure_key)
    os.environ.setdefault("OPENAI_API_BASE",
                          "https://research-collab.services.ai.azure.com/openai/v1/")

_vertex_key = os.environ.get("VERTEX_API_KEY", "")
if _vertex_key:
    os.environ.setdefault("GEMINI_API_KEY", _vertex_key)

from tasks import (
    load_subtask_tasks, score_task, score_incomplete_eval, build_instruction, get_skill_dirs,
    inline_task_data, CONDITIONS,
)
from lib.eval_retry import run_eval_with_retries
from agent.fc_loop import run_agent
from agent.tools import (
    get_tool_registry, set_task_context, clear_task_context,
    set_active_skill_dir, clear_active_skill_dir,
)

# ── Model registry ────────────────────────────────────────────────────────

MODELS = {
    "gpt-5.4":              {"litellm": "openai/gpt-5.4",              "tier": "frontier"},
    "claude-sonnet-4.6":    {"litellm": "openrouter/anthropic/claude-sonnet-4.6", "tier": "frontier"},
    "gpt-4.1":              {"litellm": "openai/gpt-4.1",              "tier": "mid"},
    "gemini-2.5-pro":       {"litellm": "gemini/gemini-2.5-pro",       "tier": "mid"},
    "grok-4":               {"litellm": "openai/grok-4",               "tier": "mid"},
    "DeepSeek-V3.2":        {"litellm": "openai/DeepSeek-V3.2",        "tier": "economical"},
    "Phi-4":                {"litellm": "openai/Phi-4",                "tier": "economical"},
    "glm-5.1":              {"litellm": "openrouter/z-ai/glm-5.1",     "tier": "economical"},
    "gemini-3.1-flash-lite": {"litellm": "gemini/gemini-3.1-flash-lite-preview", "tier": "economical"},
}

DOMAIN = "risk_management"
TOOL_DOMAIN = "risk_management"


# ── Helpers ───────────────────────────────────────────────────────────────

def _eval_key(model: str, condition: str, task_id: str, run_idx: int) -> str:
    return f"{model}|{condition}|{task_id}|{run_idx}"


def _load_completed(run_dir: Path) -> set[str]:
    completed: set[str] = set()
    results_file = run_dir / "results.jsonl"
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = _eval_key(row["model"], row["condition"],
                                row["task_id"], row.get("run_idx", 0))
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def run_single_eval(
    task: dict,
    condition: str,
    model_name: str,
    litellm_model: str,
    run_idx: int,
    run_dir: Path,
    max_turns: int = 12,
    eval_retries: int = 3,
) -> dict:
    """Run a single evaluation with function-calling agent.

    Condition handling (aligned with SkillsBench):
      - no_skill:       skill_dirs=[] (empty). load_skill tool is still exposed
                         but returns "not found". No save_skill.
      - curated:        skill_dirs=[curated_skills_root]. Agent discovers and
                         loads SKILL.md + references on demand. No save_skill.
      - self_generated: skill_dirs=[per-task temp dir] (starts empty). save_skill
                         tool is exposed so the agent can write SKILL.md files,
                         then load them back via load_skill in later turns.

    Retries (``eval_retries``): on transient provider errors, empty LLM choices, or
    early ``invalid_submission`` (possible truncation), re-run the whole eval up to
    ``1 + eval_retries`` times, overwriting logs for that run slot.
    """
    import shutil
    import tempfile

    def run_once() -> dict:
        task_id = task["task_id"]
        sub_task = task.get("sub_task", "unknown")
        logs_dir = run_dir / "logs" / model_name / condition / task_id / f"run_{run_idx}"

        instruction = build_instruction(task, condition)
        tool_registry = get_tool_registry(TOOL_DOMAIN)

        # ── Per-condition skill directory setup ────────────────────────────
        # Mirrors SkillsBench: curated → skills on filesystem; no_skill → empty
        # filesystem; self_generated → empty filesystem + save_skill tool.
        tmp_skill_dir = None
        allow_save_skill = False

        if condition == "curated":
            skill_dirs = get_skill_dirs(DOMAIN, condition)
        elif condition == "self_generated":
            tmp_skill_dir = Path(tempfile.mkdtemp(prefix=f"skills_{task_id}_"))
            skill_dirs = [tmp_skill_dir]
            allow_save_skill = True
        else:  # no_skill
            skill_dirs = []

        task_input = inline_task_data(task)
        set_task_context(task_input)

        if skill_dirs:
            active_dir = skill_dirs[0]
            if condition == "curated":
                from tasks import DOMAIN_TO_SKILL_NAME, CURATED_SKILLS_DIR
                skill_name = DOMAIN_TO_SKILL_NAME.get(DOMAIN, DOMAIN)
                candidate = CURATED_SKILLS_DIR / skill_name
                if candidate.exists():
                    active_dir = candidate
            set_active_skill_dir(active_dir)

        try:
            agent_result = run_agent(
                model_name=litellm_model,
                instruction=instruction,
                skill_dirs=skill_dirs,
                tool_registry=tool_registry,
                task_context=task_input,
                logs_dir=logs_dir,
                max_turns=max_turns,
                temperature=0.7,
                allow_save_skill=allow_save_skill,
            )
        finally:
            clear_task_context()
            clear_active_skill_dir()
            if tmp_skill_dir and tmp_skill_dir.exists():
                shutil.rmtree(tmp_skill_dir, ignore_errors=True)

        if agent_result.final_answer:
            score_result = score_task(task, agent_result.final_answer)
        else:
            score_result = score_incomplete_eval(agent_result.error)

        result_json_path = logs_dir / "result.json"
        if result_json_path.exists():
            try:
                existing = json.loads(result_json_path.read_text())
                existing["scoring"] = {
                    "score": score_result.get("score", 0.0),
                    "valid_json": score_result.get("valid_json", False),
                    "scoring_method": score_result.get("method", "unknown"),
                    "scoring_details": score_result.get("details", {}),
                }
                result_json_path.write_text(json.dumps(existing, indent=2, default=str))
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "model": model_name, "litellm_model": litellm_model,
            "model_tier": MODELS.get(model_name, {}).get("tier", "unknown"),
            "domain": DOMAIN, "condition": condition, "task_id": task_id,
            "sub_task": sub_task,
            "difficulty": task.get("difficulty", "medium"),
            "as_of_date": task.get("as_of_date", ""),
            "run_idx": run_idx,
            "score": score_result.get("score", 0.0),
            "valid_json": score_result.get("valid_json", False),
            "scoring_method": score_result.get("method", "unknown"),
            "scoring_details": score_result.get("details", {}),
            "episodes": agent_result.episodes,
            "total_input_tokens": agent_result.total_input_tokens,
            "total_output_tokens": agent_result.total_output_tokens,
            "skills_loaded": agent_result.skills_loaded,
            "tool_calls_log": agent_result.tool_calls_log,
            "error": agent_result.error,
            "latency_seconds": agent_result.latency_seconds,
        }

    return run_eval_with_retries(
        run_once, eval_retries=eval_retries, max_turns=max_turns,
    )


def build_parser(sub_task: str) -> argparse.ArgumentParser:
    """Build CLI argument parser for a subtask runner."""
    parser = argparse.ArgumentParser(
        description=f"Experiment 05 — {sub_task} subtask runner"
    )
    parser.add_argument("--model", nargs="*", dest="models", default=None,
                        help="Model name(s). Default: gpt-4.1. Use 'all' for every model.")
    parser.add_argument("--condition", nargs="*", dest="conditions", default=None,
                        help="Condition(s): no_skill, curated, self_generated. Default: all three.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max tasks to run (0 = all).")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per task.")
    parser.add_argument("--max-turns", type=int, default=7,
                        help="Max LLM turns per task.")
    parser.add_argument("--eval-retries", type=int, default=3,
                        help="Extra full-eval attempts on transient failures (0 = off).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers.")
    parser.add_argument("--run-id", default=None,
                        help="Custom run ID. Default: timestamp.")
    parser.add_argument("--resume", default=None,
                        help="Resume from existing run directory (relative to experiment dir).")
    return parser


def run_subtask_experiment(sub_task: str, args: argparse.Namespace) -> None:
    """Main entry point: load tasks for *sub_task*, fan out across models × conditions."""
    models = args.models or ["gpt-4.1"]
    if models == ["all"]:
        models = list(MODELS.keys())
    conditions = args.conditions or CONDITIONS
    if conditions == ["all"]:
        conditions = CONDITIONS
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume:
        run_dir = EXPT_DIR / args.resume
    else:
        run_dir = EXPT_DIR / "runs" / sub_task / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT 05 — {sub_task.upper()}")
    print(f"  Models:     {models}")
    print(f"  Conditions: {conditions}")
    print(f"  Runs/task:  {args.runs}")
    print(f"  Workers:    {args.workers}")
    print(f"  Eval retries: {args.eval_retries}")
    print(f"  Run dir:    {run_dir}")
    print(f"{'#'*60}")

    print("\nLoading tasks...")
    tasks = load_subtask_tasks(DOMAIN, sub_task)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"  {sub_task}: {len(tasks)} tasks")

    total_evals = len(tasks) * len(models) * len(conditions) * args.runs
    print(f"  Total evaluations planned: {total_evals}")

    completed = _load_completed(run_dir)
    if completed:
        print(f"  Resuming: {len(completed)} already completed")

    # Build work items
    work_items: list[dict] = []
    for model_name in models:
        model_info = MODELS.get(model_name, {})
        litellm_model = model_info.get("litellm", model_name)
        for condition in conditions:
            for task in tasks:
                for run_idx in range(args.runs):
                    key = _eval_key(model_name, condition, task["task_id"], run_idx)
                    if key not in completed:
                        work_items.append({
                            "task": task, "condition": condition,
                            "model_name": model_name, "litellm_model": litellm_model,
                            "run_idx": run_idx,
                        })

    pending = len(work_items)
    print(f"  Pending: {pending} evals\n")

    # Save config
    (run_dir / "config.json").write_text(json.dumps({
        "sub_task": sub_task, "domain": DOMAIN,
        "models": models, "conditions": conditions,
        "runs_per_task": args.runs, "max_turns": args.max_turns,
        "eval_retries": args.eval_retries,
        "workers": args.workers, "run_id": run_id,
        "task_count": len(tasks), "total_evals": total_evals,
        "pending_evals": pending,
    }, indent=2))

    results_file = run_dir / "results.jsonl"
    write_lock = threading.Lock()
    counter = {"done": 0, "failed": 0}
    started = time.time()

    def _run_and_record(item: dict) -> None:
        try:
            row = run_single_eval(
                task=item["task"], condition=item["condition"],
                model_name=item["model_name"], litellm_model=item["litellm_model"],
                run_idx=item["run_idx"], run_dir=run_dir, max_turns=args.max_turns,
                eval_retries=args.eval_retries,
            )
        except Exception as exc:
            row = {
                "model": item["model_name"], "domain": DOMAIN,
                "condition": item["condition"], "task_id": item["task"]["task_id"],
                "sub_task": sub_task, "run_idx": item["run_idx"], "score": 0.0,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            with write_lock:
                counter["failed"] += 1

        with write_lock:
            with open(results_file, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            counter["done"] += 1
            done = counter["done"]
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0
            eta = (pending - done) / rate if rate > 0 else 0
            tag = f"{item['model_name']}/{item['condition']}/{item['task']['task_id']}"
            score = row.get("score", 0.0)
            valid = "✓" if row.get("valid_json") else "✗"
            eps = row.get("episodes", "?")
            skills = row.get("skills_loaded", [])
            lat = row.get("latency_seconds", "?")
            err = row.get("error", "")
            att = row.get("eval_attempt")
            retry_note = f" attempts={att}" if att and att > 1 else ""
            status = f"score={score:.4f} json={valid} eps={eps} skills={skills} lat={lat}s{retry_note}"
            if err:
                status = f"FAILED: {err[:80]}{retry_note}"
            print(f"  [{done}/{pending}] {tag} → {status}  (ETA: {eta/60:.0f}m)")

    print(f"Running {pending} evals with {args.workers} workers...\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_and_record, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"  Unexpected error: {exc}")

    elapsed = time.time() - started
    print(f"\n{'#'*60}")
    print(f"  DONE: {counter['done']} evals in {elapsed/60:.1f}m "
          f"({counter['failed']} failed, {args.workers} workers)")
    if elapsed > 0:
        print(f"  Throughput: {counter['done']/elapsed:.1f} evals/s")
    print(f"  Results: {results_file}")
    print(f"{'#'*60}")
