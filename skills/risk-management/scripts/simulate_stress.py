#!/usr/bin/env python3
"""Estimate portfolio P&L under a stress scenario."""

from __future__ import annotations

import argparse
import json
import sys

from _common import numpy_to_python, portfolio_weights, resolve_factor_betas
from lib.engines.risk_engine import compute_stress_pnl


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate stress-test P&L")
    parser.add_argument("--input", type=str, default=None)
    return parser.parse_args()


def read_input(path):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    return json.load(sys.stdin)


def main():
    data = read_input(parse_args().input)
    weights = portfolio_weights(data)
    scenario = data.get("scenario") or data.get("stress_scenario")
    factor_betas = resolve_factor_betas(data)
    sector_map = data.get("sector_map")

    if weights is None:
        print(json.dumps({"error": "weights not provided"}))
        sys.exit(1)
    if scenario is None:
        print(json.dumps({"error": "scenario not provided"}))
        sys.exit(1)
    if factor_betas is None:
        print(json.dumps({"error": "factor_betas not provided or resolvable"}))
        sys.exit(1)
    if sector_map is None:
        print(json.dumps({"error": "sector_map not provided"}))
        sys.exit(1)

    if "market_shocks" not in scenario and any(
        key in scenario
        for key in ("equity_drawdown_pct", "credit_spread_widening_bp", "rate_change_bp")
    ):
        scenario = {"type": "historical_replay", "market_shocks": scenario}

    result = compute_stress_pnl(weights, scenario, factor_betas, sector_map)
    print(json.dumps(numpy_to_python(result), indent=2))


if __name__ == "__main__":
    main()
