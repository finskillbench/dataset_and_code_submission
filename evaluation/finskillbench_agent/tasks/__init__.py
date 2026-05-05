"""Task loading, instruction building, and scoring for experiment 05.

Standalone — uses lib/ instead of exp02/inspect_tasks or scripts/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[3]

from lib.task_loader import load_all_tasks as _load_all_tasks_raw, inline_task_data  # noqa: E402
from lib.scorers import score_task as _score_task_raw  # noqa: E402
from lib.scorers import (
    score_incomplete_eval as _score_incomplete_eval_raw,
    score_no_answer as _score_no_answer_raw,
)  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────

# In the submission layout, curated skills are at the top level
CURATED_SKILLS_DIR = REPO_ROOT / "skills"
XBRL_PANEL_PATH = REPO_ROOT / "data" / "fundamentals" / "processed" / "xbrl_panel.parquet"

DOMAINS = ["portfolio_construction", "risk_management", "fundamental_analysis"]
CONDITIONS = ["no_skill", "curated", "self_generated"]

DOMAIN_TO_SKILL_NAME = {
    "portfolio_construction": "portfolio-construction",
    "risk_management": "risk-management",
    "fundamental_analysis": "fundamental-analysis",
}

SELF_GEN_PROMPT = """
Important: Generate Skills First

Before attempting to solve this task, use the save_skill tool to capture
domain knowledge as reusable skill documents:
1. Analyze the task requirements and identify what financial domain knowledge,
   data sources, or analytical techniques are needed.
2. Call save_skill 1-3 times to write modular skill documents capturing the
   procedural knowledge (workflows, formulas, data access patterns, edge cases).
3. Then call load_skill to read them back and solve the task using those skills.
"""


# ── XBRL panel coverage filter ───────────────────────────────────────────

def _load_xbrl_coverage() -> set[tuple[str, str]]:
    if not XBRL_PANEL_PATH.exists():
        return set()
    import pandas as pd
    df = pd.read_parquet(XBRL_PANEL_PATH, columns=["symbol", "period_end"])
    return set(zip(df["symbol"], df["period_end"]))


def _fa_task_has_data(task: dict, panel_keys: set[tuple[str, str]]) -> bool:
    inp = task.get("input", task.get("input_data", {}))
    symbol = inp.get("symbol", "")
    sub = task.get("sub_task", "")
    if sub == "driver_decomposition":
        cp = inp.get("current_period_end", "")
        pp = inp.get("prior_period_end", "")
        return (symbol, cp) in panel_keys and (symbol, pp) in panel_keys
    else:
        period = inp.get("period_end", inp.get("current_period_end", ""))
        return (symbol, period) in panel_keys


def load_all_tasks(domains: list[str] | None = None) -> list[dict]:
    """Load tasks, filtering FA tasks to those with XBRL panel data coverage."""
    tasks = _load_all_tasks_raw(domains)
    if domains and "fundamental_analysis" not in domains:
        return tasks
    panel_keys = _load_xbrl_coverage()
    if not panel_keys:
        return tasks
    before = sum(1 for t in tasks if t.get("skill") == "fundamental_analysis")
    filtered = [
        t for t in tasks
        if t.get("skill") != "fundamental_analysis" or _fa_task_has_data(t, panel_keys)
    ]
    after = sum(1 for t in filtered if t.get("skill") == "fundamental_analysis")
    dropped = before - after
    if dropped > 0:
        print(f"  Filtered FA tasks to XBRL panel coverage: {before} → {after} ({dropped} dropped)")
    return filtered


def load_subtask_tasks(domain: str, sub_task: str) -> list[dict]:
    """Load tasks for a single (domain, sub_task) pair."""
    all_tasks = load_all_tasks([domain])
    return [t for t in all_tasks if t.get("sub_task") == sub_task]


# ── Scoring ───────────────────────────────────────────────────────────────

def score_task(task: dict, response: str) -> dict:
    return _score_task_raw(task, response)


def score_incomplete_eval(agent_error: str | None = None) -> dict:
    return _score_incomplete_eval_raw(agent_error)


def score_no_answer(error: str | None = None) -> dict:
    return _score_no_answer_raw(error)


# ── Instruction building ─────────────────────────────────────────────────

def derive_output_schema(expected_output: dict) -> dict:
    schema: dict = {}
    for key, val in expected_output.items():
        if isinstance(val, dict):
            schema[key] = derive_output_schema(val)
        elif isinstance(val, list):
            if val and isinstance(val[0], dict):
                schema[key] = [derive_output_schema(val[0])]
            else:
                schema[key] = ["<value>"]
        elif isinstance(val, bool):
            schema[key] = "<boolean>"
        elif isinstance(val, (int, float)):
            schema[key] = "<number>"
        elif isinstance(val, str):
            schema[key] = "<string>"
        else:
            schema[key] = "<value>"
    return schema


def _compact_input(inp: dict) -> dict:
    import copy
    out = copy.deepcopy(inp)
    cov = out.get("covariance_matrix")
    if isinstance(cov, dict) and "matrix" in cov:
        symbols = cov.get("symbols", [])
        n = len(cov.get("matrix", []))
        out["covariance_matrix"] = {
            "symbols": symbols, "shape": f"{n}x{n}",
            "note": "Full matrix available via tools. Not shown to save context.",
        }
    elif isinstance(cov, list) and len(cov) > 5:
        n = len(cov)
        out["covariance_matrix"] = {
            "shape": f"{n}x{n}",
            "note": "Full matrix available via tools. Not shown to save context.",
        }
    rd = out.get("returns_data")
    if isinstance(rd, list) and len(rd) > 10:
        out["returns_data"] = {
            "shape": f"{len(rd)} periods x {len(rd[0]) if rd else '?'} assets",
            "note": "Full data available via tools. Not shown to save context.",
        }
    for key in ("expected_returns", "weights", "portfolio_weights",
                "current_portfolio", "benchmark_weights"):
        w = out.get(key)
        if isinstance(w, dict):
            out[key] = {k: round(v, 3) if isinstance(v, float) else v for k, v in w.items()}
    return out


def build_instruction(task: dict, condition: str) -> str:
    inp = inline_task_data(task)
    compact_inp = _compact_input(inp)
    schema = derive_output_schema(task["expected_output"])
    prompt_data = task.get("prompt", {})
    task_desc = prompt_data.get("task_description", "")
    output_instr = prompt_data.get("output_instructions", "")
    if not task_desc:
        task_desc = f"Task: {task.get('sub_task', 'unknown')}"
    parts: list[str] = [
        f"## Task: {task.get('sub_task', 'unknown')}",
        f"as_of_date: {task.get('as_of_date', 'unknown')}",
        f"difficulty: {task.get('difficulty', 'medium')}",
        "", "## Description", task_desc,
    ]
    if output_instr:
        parts.extend(["", output_instr])
    parts.append("")
    parts.append("## Input Data")
    inp_str = json.dumps(compact_inp, indent=2)
    if len(inp_str) > 12_000:
        inp_str = inp_str[:12_000] + "\n... (truncated for context length)"
    parts.append(inp_str)
    parts.extend(["", "## Expected Output Schema",
                   "Return a JSON object matching this structure:",
                   json.dumps(schema, indent=2)])
    instruction = "\n".join(parts)
    if condition == "self_generated":
        instruction += "\n\n" + SELF_GEN_PROMPT
    return instruction


def get_skill_dirs(domain: str, condition: str) -> list[Path]:
    if condition != "curated":
        return []
    skill_name = DOMAIN_TO_SKILL_NAME.get(domain, domain)
    skill_dir = CURATED_SKILLS_DIR / skill_name
    if skill_dir.exists():
        return [CURATED_SKILLS_DIR]
    return []


def sample_tasks(
    all_tasks: list[dict], domain: str,
    max_per_subtask: int = 100, seed: int = 42,
) -> list[dict]:
    domain_tasks = [t for t in all_tasks if t.get("skill") == domain]
    if not domain_tasks:
        return []
    by_subtask: dict[str, list[dict]] = {}
    for t in domain_tasks:
        by_subtask.setdefault(t.get("sub_task", "unknown"), []).append(t)
    sampled: list[dict] = []
    for st, tasks in sorted(by_subtask.items()):
        if len(tasks) <= max_per_subtask:
            sampled.extend(tasks)
        else:
            scored = [(int(hashlib.md5(f"{seed}:{t['task_id']}".encode()).hexdigest(), 16), t)
                      for t in tasks]
            scored.sort(key=lambda x: x[0])
            sampled.extend(t for _, t in scored[:max_per_subtask])
    return sampled
