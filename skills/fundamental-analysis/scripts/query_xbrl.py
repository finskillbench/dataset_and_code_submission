#!/usr/bin/env python3
"""Query normalized XBRL metrics from the processed panel parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[6]
PANEL_PATH = REPO_ROOT / "data" / "fundamentals" / "processed" / "xbrl_panel.parquet"


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


def parse_args():
    parser = argparse.ArgumentParser(description="Query XBRL financial data panel")
    parser.add_argument("--input", type=str, default=None)
    return parser.parse_args()


def read_input(path):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    return json.load(sys.stdin)


def query_xbrl(data):
    ticker = data.get("ticker") or data.get("symbol")
    period = data.get("period") or data.get("period_end")
    metrics = data.get("metrics")

    if not ticker:
        return {"error": "ticker is required", "status": "error"}
    if not PANEL_PATH.exists():
        return {"error": f"XBRL panel not found: {PANEL_PATH}", "status": "error"}

    df = pd.read_parquet(PANEL_PATH)
    mask = df["symbol"] == str(ticker).upper()
    if period:
        mask &= df["period_end"] == period
    filtered = df[mask]
    if filtered.empty:
        return {"error": f"No data found for {ticker} period={period}", "status": "empty"}

    if metrics:
        filtered = filtered[filtered["canonical_name"].isin(metrics)]
        if filtered.empty:
            available = sorted(df[df["symbol"] == str(ticker).upper()]["canonical_name"].unique())
            return {
                "error": f"No matching metrics for {ticker}. Available: {available}",
                "status": "empty",
            }

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

    return {
        "status": "ok",
        "n_periods": len(records),
        "metrics": sorted(deduped["canonical_name"].unique().tolist()),
        "data": records,
    }


def main():
    try:
        data = read_input(parse_args().input)
        result = query_xbrl(data)
        print(json.dumps(numpy_to_python(result), indent=2))
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "error": f"Invalid JSON input: {e}"}))
        sys.exit(1)
    except FileNotFoundError as e:
        print(json.dumps({"status": "error", "error": f"Input file not found: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
