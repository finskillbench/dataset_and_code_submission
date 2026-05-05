#!/usr/bin/env python3
"""
Stress test P&L estimation for a portfolio.

Estimates portfolio P&L under a hypothetical stress scenario using factor
betas and sector shocks, with attribution by sector and factor.

Usage:
    python stress_pnl.py --input stress_input.json
    echo '{"weights": ...}' | python stress_pnl.py

Input format:
    {
        "weights": {"TICKER": float, ...},
        "scenario": {
            "type": str,
            "shocks": [
                {"factor": "equity_market", "value": -0.20},
                {"factor": "credit_spread", "value": 0.02},
                {"sector": "Technology", "value": -0.25}
            ]
        },
        "factor_betas": {
            "TICKER": {"equity_market": 1.1, "credit_spread": -0.3, ...}
        },
        "sector_map": {"TICKER": "sector", ...}
    }

Output format:
    {
        "pnl_pct": float,
        "attribution": {
            "by_sector": [{"sector": str, "pnl_pct": float}],
            "by_factor": [{"factor": str, "pnl_pct": float}]
        }
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate portfolio P&L under a stress scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to JSON input file. If omitted, reads from stdin.",
    )
    return parser.parse_args()


def read_input(path):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    return json.load(sys.stdin)


def compute_stress_pnl(data):
    weights = data["weights"]
    scenario = data.get("scenario", {})
    factor_betas = data.get("factor_betas", {})
    sector_map = data.get("sector_map", {})

    shocks = scenario.get("shocks", [])

    # Separate factor shocks from sector shocks
    factor_shocks = {}
    sector_shocks = {}
    for shock in shocks:
        if "factor" in shock:
            factor_shocks[shock["factor"]] = shock["value"]
        elif "sector" in shock:
            sector_shocks[shock["sector"]] = shock["value"]

    # --- Compute per-ticker P&L from factor exposures ---
    ticker_pnl = {}
    for ticker, w in weights.items():
        betas = factor_betas.get(ticker, {})
        factor_pnl = 0.0
        for factor, shock_value in factor_shocks.items():
            beta = betas.get(factor, 0.0)
            factor_pnl += beta * shock_value

        # Add sector-specific shock
        sector = sector_map.get(ticker, None)
        sector_pnl = 0.0
        if sector and sector in sector_shocks:
            # Sector shock is additive on top of factor-driven P&L
            # Only apply if the sector shock is incremental to factor shocks
            sector_pnl = sector_shocks[sector]

        # Total P&L for this ticker as fraction of portfolio
        ticker_pnl[ticker] = w * (factor_pnl + sector_pnl)

    total_pnl = sum(ticker_pnl.values())

    # --- Attribution by sector ---
    sector_pnl_agg = {}
    for ticker, pnl in ticker_pnl.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_pnl_agg.setdefault(sector, 0.0)
        sector_pnl_agg[sector] += pnl

    by_sector = [
        {"sector": s, "pnl_pct": round(p, 8)}
        for s, p in sorted(sector_pnl_agg.items(), key=lambda x: x[0])
    ]

    # --- Attribution by factor ---
    factor_pnl_agg = {}
    for factor, shock_value in factor_shocks.items():
        factor_contrib = 0.0
        for ticker, w in weights.items():
            betas = factor_betas.get(ticker, {})
            beta = betas.get(factor, 0.0)
            factor_contrib += w * beta * shock_value
        factor_pnl_agg[factor] = factor_contrib

    # Sector shocks as a separate attribution item
    for sector, shock_value in sector_shocks.items():
        sector_contrib = 0.0
        for ticker, w in weights.items():
            if sector_map.get(ticker) == sector:
                sector_contrib += w * shock_value
        factor_pnl_agg[f"sector_{sector}"] = sector_contrib

    by_factor = [
        {"factor": f, "pnl_pct": round(p, 8)}
        for f, p in sorted(factor_pnl_agg.items(), key=lambda x: x[0])
    ]

    return {
        "pnl_pct": round(total_pnl, 8),
        "attribution": {
            "by_sector": by_sector,
            "by_factor": by_factor,
        },
    }


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        output = {
            "pnl_pct": 0.0,
            "attribution": {"by_sector": [], "by_factor": []},
            "error": f"Invalid JSON input: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)
    except FileNotFoundError as e:
        output = {
            "pnl_pct": 0.0,
            "attribution": {"by_sector": [], "by_factor": []},
            "error": f"Input file not found: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "weights" not in data:
        output = {
            "pnl_pct": 0.0,
            "attribution": {"by_sector": [], "by_factor": []},
            "error": "Missing required field: weights",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "scenario" not in data:
        output = {
            "pnl_pct": 0.0,
            "attribution": {"by_sector": [], "by_factor": []},
            "error": "Missing required field: scenario",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    result = compute_stress_pnl(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
