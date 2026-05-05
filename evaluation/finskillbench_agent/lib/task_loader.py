"""Task loader: reads episode JSONs from domain directories.

Internalized from inspect_tasks/__init__.py for standalone use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

PC_TASKS_DIR = DATA_DIR / "portfolio_construction" / "episodes" / "layer_a"
RM_TASKS_DIR = DATA_DIR / "risk_management" / "episodes" / "layer_a"
FA_TASKS_DIR = DATA_DIR / "fundamentals" / "episodes" / "layer_a"

MANDATES_DIR = DATA_DIR / "risk_management" / "constructed" / "mandates"
STRESS_SCENARIOS_DIR = DATA_DIR / "risk_management" / "constructed" / "stress_scenarios"
SECURITY_MASTER_PATH = DATA_DIR / "universe" / "sp500_security_master_latest.json"

DOMAINS = ["portfolio_construction", "risk_management", "fundamental_analysis"]


def _infer_verification_method(task: dict) -> str:
    sub = task.get("sub_task", task.get("task_type", ""))
    mapping = {
        "unconstrained_optimization": "l2_distance_and_objective",
        "constrained_optimization": "constraint_satisfaction_and_objective",
        "rebalancing": "turnover_compliance_and_objective",
        "tool_use_parameterization": "parameter_match",
        "infeasibility_detection": "infeasibility_detection",
        "black_litterman": "view_specification_and_weights",
        "constraint_monitoring": "exact_match",
        "stress_testing": "absolute_error",
        "risk_identification": "ranked_list_recall",
        "risk_remediation": "constraint_satisfaction_plus_cost",
        "normalization": "metric_absolute_error",
    }
    return mapping.get(sub, "l2_distance_and_objective")


def _normalize_task(task: dict, domain: str) -> dict:
    if task.get("skill") and task.get("expected_output"):
        return task
    task_type = task.get("task_type", task.get("sub_task", ""))
    task_id = task.get("task_id", "")
    if "skill" not in task:
        if task_id.startswith("pc_") or domain == "portfolio_construction":
            task["skill"] = "portfolio_construction"
        elif task_id.startswith("rm_") or domain == "risk_management":
            task["skill"] = "risk_management"
        else:
            task["skill"] = domain
    if "sub_task" not in task:
        task["sub_task"] = task_type or task.get("description", "unknown")
    if "input" not in task and "inputs" in task:
        task["input"] = task.pop("inputs")
    if "expected_output" not in task and "ground_truth_file" in task:
        source_dir = Path(task["_source_file"]).parent
        gt_path = source_dir / task["ground_truth_file"]
        if gt_path.exists():
            try:
                task["expected_output"] = json.loads(gt_path.read_text())
            except json.JSONDecodeError:
                task["expected_output"] = {}
        else:
            task["expected_output"] = {}
    if "verification" not in task:
        task["verification"] = {"method": _infer_verification_method(task)}
    if "difficulty" not in task:
        task["difficulty"] = "medium"
    if "metadata" not in task:
        task["metadata"] = {}
    return task


def _load_mandates() -> dict[str, dict]:
    mandates: dict[str, dict] = {}
    if not MANDATES_DIR.exists():
        return mandates
    for f in MANDATES_DIR.glob("*.json"):
        try:
            m = json.loads(f.read_text())
            mandates[m.get("mandate_id", f.stem)] = m
        except json.JSONDecodeError:
            pass
    return mandates


def _resolve_mandate_refs(tasks: list[dict]) -> list[dict]:
    mandates = _load_mandates()
    if not mandates:
        return tasks
    filled = 0
    for t in tasks:
        if t.get("skill") != "risk_management":
            continue
        inp = t.get("input", {})
        mandate_ref = inp.get("mandate", {})
        if isinstance(mandate_ref, dict) and "mandate_id" in mandate_ref:
            mid = mandate_ref["mandate_id"]
            if mid in mandates and "constraints" not in mandate_ref:
                mandate_ref["name"] = mandates[mid].get("name", "")
                mandate_ref["constraints"] = mandates[mid].get("constraints", [])
                filled += 1
    if filled:
        print(f"  Resolved mandate constraints for {filled} tasks")
    return tasks


def _load_stress_scenarios() -> dict[str, dict]:
    scenarios: dict[str, dict] = {}
    if not STRESS_SCENARIOS_DIR.exists():
        return scenarios
    for f in STRESS_SCENARIOS_DIR.glob("*.json"):
        try:
            s = json.loads(f.read_text())
            scenarios[s.get("scenario_id", f.stem)] = s
        except json.JSONDecodeError:
            pass
    return scenarios


def _resolve_stress_scenario_refs(tasks: list[dict]) -> list[dict]:
    """Inline full stress scenario data (shocks, type, etc.) into task input."""
    scenarios = _load_stress_scenarios()
    if not scenarios:
        return tasks
    filled = 0
    for t in tasks:
        if t.get("skill") != "risk_management":
            continue
        inp = t.get("input", {})
        scenario_ref = inp.get("stress_scenario", {})
        if isinstance(scenario_ref, dict) and "scenario_id" in scenario_ref:
            sid = scenario_ref["scenario_id"]
            if sid in scenarios and "market_shocks" not in scenario_ref:
                # Merge full scenario data into the reference
                full = scenarios[sid]
                for key, val in full.items():
                    if key not in scenario_ref:
                        scenario_ref[key] = val
                filled += 1
    if filled:
        print(f"  Resolved stress scenario data for {filled} tasks")
    return tasks


def _load_sector_map() -> dict[str, str]:
    """Load symbol→sector mapping from the security master."""
    if not SECURITY_MASTER_PATH.exists():
        return {}
    try:
        data = json.loads(SECURITY_MASTER_PATH.read_text())
        constituents = data.get("constituents", [])
        return {s["symbol"]: s.get("sector", "Unknown") for s in constituents}
    except (json.JSONDecodeError, KeyError):
        return {}


def _inject_sector_map(tasks: list[dict]) -> list[dict]:
    """Add sector_map to task input for all RM tasks that need it."""
    sector_map = _load_sector_map()
    if not sector_map:
        return tasks
    injected = 0
    for t in tasks:
        if t.get("skill") != "risk_management":
            continue
        inp = t.get("input", {})
        if "sector_map" not in inp:
            # Only inject symbols that appear in the portfolio holdings
            holdings = inp.get("portfolio", {}).get("holdings", [])
            if holdings:
                portfolio_symbols = {h["symbol"] for h in holdings}
                inp["sector_map"] = {
                    sym: sector_map[sym]
                    for sym in portfolio_symbols
                    if sym in sector_map
                }
                injected += 1
    if injected:
        print(f"  Injected sector_map for {injected} RM tasks")
    return tasks


def _fill_missing_returns(tasks: list[dict]) -> list[dict]:
    returns_lookup: dict[tuple, dict] = {}
    bw_lookup: dict[tuple, dict] = {}
    for t in tasks:
        if t.get("skill") != "portfolio_construction":
            continue
        inp = t.get("input", {})
        er = inp.get("expected_returns")
        if er and len(er) > 0:
            returns_lookup[("portfolio_construction", t.get("as_of_date", ""))] = er
        bw = inp.get("benchmark_weights")
        if bw and len(bw) > 0:
            bw_lookup[("portfolio_construction", t.get("as_of_date", ""))] = bw
    filled = 0
    for t in tasks:
        if t.get("skill") != "portfolio_construction":
            continue
        inp = t.get("input", {})
        key = ("portfolio_construction", t.get("as_of_date", ""))
        if not inp.get("expected_returns") or len(inp.get("expected_returns", {})) == 0:
            if key in returns_lookup:
                inp["expected_returns"] = returns_lookup[key]
                filled += 1
        if not inp.get("benchmark_weights") or len(inp.get("benchmark_weights", {})) == 0:
            if key in bw_lookup:
                inp["benchmark_weights"] = bw_lookup[key]
    if filled:
        print(f"  Filled expected_returns for {filled} tasks from same-date tasks")
    return tasks


def _apply_scoring_verification_defaults(tasks: list[dict]) -> list[dict]:
    """Inject scorer defaults aligned with scoring_methodology_review when absent."""
    for t in tasks:
        sub = t.get("sub_task", "")
        v = t.setdefault("verification", {})
        if sub == "earnings_quality":
            v.setdefault("numeric_ratio_weight", 0.3)
        elif sub == "stress_testing":
            v.setdefault("pnl_score_weight", 0.7)
            v.setdefault("attribution_score_weight", 0.3)
    return tasks


def load_all_tasks(domains: list[str] | None = None) -> list[dict]:
    """Load all tasks from episode JSON files."""
    domains = domains or DOMAINS
    tasks: list[dict] = []
    for domain in domains:
        if domain == "portfolio_construction":
            base = PC_TASKS_DIR
        elif domain == "risk_management":
            base = RM_TASKS_DIR
        elif domain == "fundamental_analysis":
            base = FA_TASKS_DIR
        else:
            continue
        if not base.exists():
            print(f"  WARNING: {base} does not exist, skipping {domain}")
            continue
        count = 0
        for subtask_dir in sorted(base.iterdir()):
            if not subtask_dir.is_dir():
                continue
            for f in sorted(subtask_dir.glob("*.json")):
                try:
                    task = json.loads(f.read_text())
                    task["_source_file"] = str(f)
                    task = _normalize_task(task, domain)
                    if not task.get("expected_output"):
                        continue
                    tasks.append(task)
                    count += 1
                except json.JSONDecodeError:
                    print(f"  WARNING: could not parse {f}")
        print(f"  Loaded {count} tasks from {domain}")
    tasks = _fill_missing_returns(tasks)
    tasks = _resolve_mandate_refs(tasks)
    tasks = _resolve_stress_scenario_refs(tasks)
    tasks = _inject_sector_map(tasks)
    tasks = _apply_scoring_verification_defaults(tasks)
    return tasks


def resolve_covariance(ref: str) -> dict | None:
    """Load a covariance matrix from an .npz file."""
    path = REPO_ROOT / ref
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    symbols = list(data["symbols"])
    cov = data["cov"]
    return {"symbols": symbols, "matrix": cov.tolist()}


def inline_task_data(task: dict) -> dict:
    """Resolve file references in task input (covariance .npz, etc.)."""
    inp = json.loads(json.dumps(task["input"]))
    if "covariance_matrix" in inp and isinstance(inp["covariance_matrix"], dict):
        ref = inp["covariance_matrix"].get("ref")
        if ref:
            cov = resolve_covariance(ref)
            if cov:
                inp["covariance_matrix"] = cov
    if "market_data" in inp and isinstance(inp["market_data"], dict):
        for key, val in list(inp["market_data"].items()):
            if isinstance(val, str) and val.endswith(".npz"):
                cov = resolve_covariance(val)
                if cov:
                    inp["market_data"][key] = cov
    return inp


def split_tasks(tasks: list[dict], dev_frac: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    dev, held = [], []
    for t in tasks:
        h = int(hashlib.md5(t["task_id"].encode()).hexdigest(), 16)
        if (h % 100) < int(dev_frac * 100):
            dev.append(t)
        else:
            held.append(t)
    return dev, held
