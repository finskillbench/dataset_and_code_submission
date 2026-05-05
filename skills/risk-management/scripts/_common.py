"""Shared helpers for curated risk-management scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXPT_DIR = Path(__file__).resolve().parents[4]
REPO_ROOT = Path(__file__).resolve().parents[6]
if str(EXPT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPT_DIR))


def numpy_to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    return obj


def normalize_weights(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, list):
        weights = {}
        for item in value:
            if isinstance(item, dict) and "symbol" in item and "weight" in item:
                weights[str(item["symbol"])] = float(item["weight"])
        return weights or None
    return None


def portfolio_weights(data: dict[str, Any]) -> dict[str, float] | None:
    weights = normalize_weights(
        data.get("portfolio_weights") or data.get("weights") or data.get("current_portfolio")
    )
    if weights is not None:
        return weights
    portfolio = data.get("portfolio", {})
    if isinstance(portfolio, dict):
        return normalize_weights(portfolio.get("holdings"))
    return None


def parse_factor_exposures_df(df) -> dict[str, dict[str, float]] | None:
    if df.empty:
        return None
    beta_cols = [c for c in df.columns if c.endswith("_beta")]
    if beta_cols and "symbol" in df.columns:
        if "as_of_date" in df.columns:
            df = df.sort_values("as_of_date").drop_duplicates("symbol", keep="last")
        result = {}
        for _, row in df.iterrows():
            result[str(row["symbol"])] = {
                col.replace("_beta", ""): float(row[col])
                for col in beta_cols
            }
        return result
    if {"symbol", "factor", "beta"}.issubset(df.columns):
        result = {}
        for sym, grp in df.groupby("symbol"):
            result[str(sym)] = {
                str(row["factor"]): float(row["beta"])
                for _, row in grp.iterrows()
            }
        return result
    return None


def resolve_factor_betas(data: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    factor_betas = data.get("factor_betas") or data.get("factor_exposures")
    if isinstance(factor_betas, dict):
        return factor_betas
    refs = []
    if isinstance(factor_betas, str):
        refs.append(factor_betas)
    market_data = data.get("market_data", {})
    if isinstance(market_data, dict) and isinstance(market_data.get("factor_exposures"), str):
        refs.append(market_data["factor_exposures"])
    for ref in refs:
        if not ref.endswith(".parquet"):
            continue
        path = Path(ref) if Path(ref).is_absolute() else REPO_ROOT / ref
        if not path.exists():
            continue
        try:
            import pandas as pd

            parsed = parse_factor_exposures_df(pd.read_parquet(path))
            if parsed:
                return parsed
        except Exception:
            continue
    return None


def resolve_covariance(data: dict[str, Any]) -> tuple[np.ndarray | None, list[str] | None]:
    cov_ref = data.get("covariance_matrix")
    if isinstance(cov_ref, dict):
        matrix = cov_ref.get("matrix")
        symbols = cov_ref.get("symbols") or cov_ref.get("tickers")
        if matrix is not None:
            return np.array(matrix), symbols
    if isinstance(cov_ref, list):
        return np.array(cov_ref), data.get("symbols")

    market_data = data.get("market_data", {})
    if isinstance(market_data, dict):
        cov_ref = market_data.get("covariance")
    if isinstance(cov_ref, str):
        cov_path = Path(cov_ref) if Path(cov_ref).is_absolute() else REPO_ROOT / cov_ref
        if cov_path.is_dir():
            files = sorted(cov_path.glob("*.npz"))
            cov_path = files[-1] if files else cov_path
        if cov_path.exists() and cov_path.suffix == ".npz":
            npz = np.load(cov_path, allow_pickle=True)
            cov_key = "cov" if "cov" in npz else "cov_matrix"
            matrix = npz[cov_key] if cov_key in npz else None
            symbols = [str(s) for s in npz["symbols"]] if "symbols" in npz else None
            if matrix is not None:
                return np.array(matrix), symbols
    return None, None


def extend_sector_map(sector_map: dict[str, str], symbols: list[str] | None) -> dict[str, str]:
    if not symbols:
        return sector_map
    missing = [s for s in symbols if s not in sector_map]
    if not missing:
        return sector_map
    master_path = REPO_ROOT / "data" / "universe" / "sp500_security_master_latest.json"
    if not master_path.exists():
        return sector_map
    try:
        full_data = json.loads(master_path.read_text())
        full_map = {
            item["symbol"]: item.get("sector", "Unknown")
            for item in full_data.get("constituents", [])
        }
        for sym in missing:
            if sym in full_map:
                sector_map[sym] = full_map[sym]
    except Exception:
        pass
    return sector_map
