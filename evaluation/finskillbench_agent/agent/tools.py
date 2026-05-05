"""Domain tool registry for function-calling agent.

Wraps the deterministic financial engines from lib/engines/ into a registry
format that the function-calling loop can use to build schemas and dispatch calls.

Tools auto-read from task context when arguments are omitted, so the model
doesn't need to echo back large numeric payloads (covariance matrices, etc.).

Standalone — no dependency on exp02 or scripts/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPT_DIR = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable  # use the current interpreter (must have cvxpy, numpy, etc.)

# ── Task context ──────────────────────────────────────────────────────────
# Thread-local to avoid races when running parallel evaluations.

def set_task_context(task_input: dict) -> None:
    _thread_local.task_context = task_input


def clear_task_context() -> None:
    _thread_local.task_context = {}


def _get_task_context() -> dict:
    return getattr(_thread_local, 'task_context', {})

# ── Active skill directory ────────────────────────────────────────────────
# Thread-local to avoid races when running parallel evaluations.
import threading
_thread_local = threading.local()


def set_active_skill_dir(skill_dir: Path | None) -> None:
    _thread_local.active_skill_dir = skill_dir


def clear_active_skill_dir() -> None:
    _thread_local.active_skill_dir = None


def _get_active_skill_dir() -> Path | None:
    return getattr(_thread_local, 'active_skill_dir', None)


def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    return obj


# ── Data access tool ──────────────────────────────────────────────────────

def tool_get_task_data(
    field: str,
) -> dict:
    """Retrieve a field from the task input data.

    Use this to access large data objects (covariance matrices, returns data,
    factor exposures, etc.) that are summarized in the instruction but available
    in full. Returns the raw data as JSON.

    Common fields: expected_returns, covariance_matrix (returns {symbols, matrix}),
    constraints, benchmark_weights, current_portfolio, risk_free_rate, objective,
    sector_mapping, factor_betas, returns_data, mandate, scenario.
    """
    ctx = _get_task_context()
    if not ctx:
        return {"error": "No task context available"}
    if field not in ctx:
        available = [k for k in ctx.keys() if not k.startswith("_")]
        return {"error": f"Field '{field}' not found. Available: {available}"}
    value = ctx[field]
    return _numpy_to_python({"field": field, "data": value})


def _is_path_within(child: Path, parent: Path) -> bool:
    """Return True if *child* is inside *parent* (resolves symlinks)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def tool_query_xbrl(
    ticker: str,
    period: str | None = None,
    metrics: list[str] | None = None,
) -> dict:
    """Query XBRL financial data panel for a specific ticker/period."""
    panel_path = REPO_ROOT / "data" / "fundamentals" / "processed" / "xbrl_panel.parquet"
    if not panel_path.exists():
        return {"error": "XBRL panel not found", "status": "error"}
    import pandas as pd
    df = pd.read_parquet(panel_path)
    mask = df["symbol"] == ticker.upper()
    if period:
        mask &= df["period_end"] == period
    filtered = df[mask]
    if filtered.empty:
        return {"error": f"No data found for {ticker} period={period}", "status": "empty"}
    if metrics:
        filtered = filtered[filtered["canonical_name"].isin(metrics)]
        if filtered.empty:
            available = sorted(df[df["symbol"] == ticker.upper()]["canonical_name"].unique())
            return {"error": f"No matching metrics for {ticker}. Available: {available}", "status": "empty"}
    deduped = (
        filtered.sort_values("duration_days")
        .drop_duplicates(subset=["symbol", "period_end", "canonical_name"], keep="first")
    )
    records = []
    for (sym, pe), grp in deduped.groupby(["symbol", "period_end"]):
        rec = {"symbol": sym, "period_end": pe}
        for _, row in grp.iterrows():
            rec[row["canonical_name"]] = row["value"]
        rec["fiscal_year"] = int(grp["fiscal_year"].iloc[0])
        rec["fiscal_quarter"] = grp["fiscal_quarter"].iloc[0]
        records.append(rec)
    return _numpy_to_python({
        "status": "ok", "n_periods": len(records),
        "metrics": sorted(deduped["canonical_name"].unique().tolist()),
        "data": records,
    })


# The curated skills root — only scripts under this tree are allowed.
CURATED_SKILLS_DIR = EXPT_DIR / "skills" / "curated"


def tool_run_skill_script(
    script_path: str,
    input_data: dict,
    inject_task_fields: list[str] | None = None,
    timeout_seconds: int = 30,
) -> dict:
    """Execute a skill script and return its JSON output.

    Args:
        script_path: Relative path to the script (e.g. 'scripts/optimize.py').
        input_data: JSON-serializable dict passed to the script. You can include
            small fields (objective, constraints, risk_free_rate) here directly.
        inject_task_fields: List of field names to auto-inject from the task
            input data at full precision. Use this for large numeric payloads
            (covariance_matrix, expected_returns) that lose precision when passed
            through the LLM. Example: ["covariance_matrix", "expected_returns"].
            The injected covariance_matrix will have its "matrix" and "symbols"
            extracted automatically.
        timeout_seconds: Max execution time.
    """
    # Reject absolute paths — the agent should never supply them.
    cleaned = script_path.strip('"').strip("'")
    if Path(cleaned).is_absolute():
        return {"error": "Absolute script paths are not allowed.", "exit_code": -1}

    active_dir = _get_active_skill_dir()
    script: Path | None = None

    # Resolve only relative to the active skill directory.
    if active_dir:
        candidate = active_dir / cleaned
        if candidate.exists():
            script = candidate

    if script is None:
        available = []
        if active_dir:
            scripts_dir = active_dir / "scripts"
            if scripts_dir.exists():
                available = [f"scripts/{f.name}" for f in sorted(scripts_dir.glob("*.py"))]
        hint = f" Available scripts: {available}" if available else ""
        return {"error": f"Script not found: {script_path}.{hint}", "exit_code": -1}

    # Path-containment check: the resolved script must live inside either
    # the curated skills tree or the active skill directory itself (which
    # covers the self_generated temp dir).  This blocks ``../`` traversal.
    if not (_is_path_within(script, CURATED_SKILLS_DIR)
            or (active_dir and _is_path_within(script, active_dir))):
        return {
            "error": "Script path resolves outside the allowed skills directory.",
            "exit_code": -1,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Inject task data fields at full precision (avoids LLM float corruption)
        merged_input = dict(input_data)
        if inject_task_fields:
            ctx = _get_task_context()
            for field in inject_task_fields:
                if field in ctx:
                    value = ctx[field]
                    # For covariance_matrix, auto-extract matrix and symbols
                    if field == "covariance_matrix" and isinstance(value, dict):
                        merged_input["covariance_matrix"] = _numpy_to_python(value.get("matrix", value))
                        if "symbols" in value and "symbols" not in merged_input:
                            merged_input["symbols"] = value["symbols"]
                    else:
                        merged_input[field] = _numpy_to_python(value)

        input_file = Path(tmpdir) / "input.json"
        input_file.write_text(json.dumps(merged_input))
        try:
            # Inherit enough environment for the virtualenv's packages to work
            import os as _os
            script_env = {
                "PATH": _os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": tmpdir,
                "VIRTUAL_ENV": _os.environ.get("VIRTUAL_ENV", ""),
                "PYTHONPATH": _os.environ.get("PYTHONPATH", ""),
            }
            proc = subprocess.run(
                [PYTHON_BIN, str(script.resolve()), "--input", str(input_file)],
                capture_output=True, text=True, timeout=timeout_seconds,
                cwd=tmpdir, env=script_env,
            )
            if proc.returncode != 0:
                # Include both stderr and stdout — many scripts write errors to stdout as JSON
                return {"error": f"Script exited with code {proc.returncode}",
                        "stderr": proc.stderr[:500],
                        "stdout": proc.stdout[:500],
                        "exit_code": proc.returncode}
            try:
                output = json.loads(proc.stdout)
                return {"output": output, "exit_code": 0}
            except json.JSONDecodeError:
                return {"error": "Script output was not valid JSON",
                        "stdout": proc.stdout[:500], "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"error": f"Script timed out after {timeout_seconds}s", "exit_code": -1}
        except Exception as e:
            return {"error": str(e), "exit_code": -1}


# ── Tool registry ─────────────────────────────────────────────────────────

TOOLS = {
    "get_task_data": {
        "fn": tool_get_task_data,
        "description": (
            "Retrieve a field from the task input data. Use this to access large "
            "data objects (covariance matrices, expected returns, etc.) that are "
            "summarized in the instruction but available in full. "
            "Pass the field name (e.g. 'covariance_matrix', 'expected_returns')."
        ),
        "domains": ["portfolio_construction", "risk_management", "fundamental_analysis"],
    },
    "run_skill_script": {
        "fn": tool_run_skill_script,
        "description": (
            "Execute a skill script with JSON I/O. "
            "Pass the script path (e.g. 'scripts/optimize.py') and an input_data dict. "
            "The script receives the input as a JSON file and prints JSON to stdout. "
            "Use load_skill first to discover available scripts and their expected input format."
        ),
        "domains": ["portfolio_construction", "risk_management", "fundamental_analysis"],
    },
    "query_xbrl": {
        "fn": tool_query_xbrl,
        "description": "Query XBRL financial data panel for a specific ticker/period.",
        "domains": ["fundamental_analysis"],
    },
}


def get_tools_for_domain(domain: str) -> list[dict]:
    """Return tool definitions available for a given domain."""
    return [
        {"name": name, "description": t["description"], "fn": t["fn"]}
        for name, t in TOOLS.items()
        if domain in t["domains"]
    ]


def get_tool_registry(domain: str) -> dict[str, dict]:
    """Get the tool registry for a domain.

    Returns dict of {tool_name: {"fn": callable, "description": str}}.
    """
    tools_for_domain = get_tools_for_domain(domain)
    registry: dict[str, dict] = {}
    for tool_def in tools_for_domain:
        registry[tool_def["name"]] = {
            "fn": tool_def["fn"],
            "description": tool_def["description"],
        }
    return registry
