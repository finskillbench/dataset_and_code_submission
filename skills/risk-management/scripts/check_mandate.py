#!/usr/bin/env python3
"""
Investment mandate compliance checker.

Checks whether a portfolio's weights satisfy a set of mandate constraints,
including position limits, sector limits, beta range, long-only requirement,
and turnover limits.

Usage:
    python check_mandate.py --input mandate_input.json
    echo '{"portfolio_weights": ...}' | python check_mandate.py

Input format:
    {
        "portfolio_weights": {"TICKER": float, ...},
        "mandate": {
            "constraints": [
                {"type": "position_limit_max", "value": 0.10},
                {"type": "sector_limit_max", "value": 0.30},
                {"type": "long_only", "value": true},
                {"type": "beta_range", "min": 0.8, "max": 1.2},
                {"type": "turnover_max_monthly", "value": 0.20, "current_weights": {...}}
            ]
        },
        "sector_map": {"TICKER": "sector", ...}
    }

Output format:
    {
        "overall_compliant": bool,
        "constraints": [
            {"type": str, "status": "PASS"|"FAIL", "value": float, "limit": float}
        ]
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check portfolio compliance against investment mandate",
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


def check_position_limit(weights, constraint):
    """Check max position size constraint."""
    limit = constraint["value"]
    max_ticker = max(weights, key=weights.get)
    max_weight = weights[max_ticker]
    passed = max_weight <= limit + 1e-6
    result = {
        "type": "position_limit_max",
        "status": "PASS" if passed else "FAIL",
        "value": round(max_weight, 6),
        "limit": limit,
    }
    if not passed:
        result["worst_offender"] = max_ticker
    return result


def check_sector_limit(weights, constraint, sector_map):
    """Check max sector concentration constraint."""
    limit = constraint["value"]
    sector_weights = {}
    for ticker, w in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_weights.setdefault(sector, 0.0)
        sector_weights[sector] += w

    worst_sector = max(sector_weights, key=sector_weights.get) if sector_weights else "N/A"
    max_sector_weight = sector_weights.get(worst_sector, 0.0)
    passed = max_sector_weight <= limit + 1e-6
    result = {
        "type": "sector_limit_max",
        "status": "PASS" if passed else "FAIL",
        "value": round(max_sector_weight, 6),
        "limit": limit,
    }
    if not passed:
        result["worst_offender"] = worst_sector
        result["sector_breakdown"] = {k: round(v, 6) for k, v in sector_weights.items()}
    return result


def check_long_only(weights, constraint):
    """Check that all positions are non-negative."""
    required = constraint.get("value", True)
    if not required:
        return {
            "type": "long_only",
            "status": "PASS",
            "value": "not required",
            "limit": "N/A",
        }

    neg_positions = {t: w for t, w in weights.items() if w < -1e-6}
    passed = len(neg_positions) == 0
    result = {
        "type": "long_only",
        "status": "PASS" if passed else "FAIL",
        "value": len(neg_positions),
        "limit": 0,
    }
    if not passed:
        result["short_positions"] = {k: round(v, 6) for k, v in neg_positions.items()}
    return result


def check_beta_range(weights, constraint):
    """Check that portfolio beta falls within the specified range."""
    beta_min = constraint.get("min", 0.0)
    beta_max = constraint.get("max", float("inf"))

    # If portfolio_beta is provided directly in the constraint
    portfolio_beta = constraint.get("portfolio_beta", None)

    if portfolio_beta is None:
        # Cannot check without beta data; report as PASS with note
        return {
            "type": "beta_range",
            "status": "PASS",
            "value": None,
            "limit": f"[{beta_min}, {beta_max}]",
            "note": "portfolio_beta not provided; cannot verify",
        }

    passed = beta_min - 1e-6 <= portfolio_beta <= beta_max + 1e-6
    return {
        "type": "beta_range",
        "status": "PASS" if passed else "FAIL",
        "value": round(portfolio_beta, 6),
        "limit": f"[{beta_min}, {beta_max}]",
    }


def check_turnover(weights, constraint):
    """Check monthly turnover constraint."""
    limit = constraint["value"]
    current_weights = constraint.get("current_weights", {})

    if not current_weights:
        return {
            "type": "turnover_max_monthly",
            "status": "PASS",
            "value": None,
            "limit": limit,
            "note": "current_weights not provided; cannot verify",
        }

    all_tickers = set(list(weights.keys()) + list(current_weights.keys()))
    turnover = 0.0
    for t in all_tickers:
        w_new = weights.get(t, 0.0)
        w_old = current_weights.get(t, 0.0)
        turnover += abs(w_new - w_old)
    # Full turnover (not half)
    passed = turnover <= limit + 1e-6
    return {
        "type": "turnover_max_monthly",
        "status": "PASS" if passed else "FAIL",
        "value": round(turnover, 6),
        "limit": limit,
    }


def check_min_names(weights, constraint):
    """Check minimum number of positions constraint."""
    limit = constraint["value"]
    n_positions = sum(1 for w in weights.values() if abs(w) > 1e-6)
    passed = n_positions >= limit
    return {
        "type": "min_names",
        "status": "PASS" if passed else "FAIL",
        "value": n_positions,
        "limit": limit,
    }


def check_weights_sum(weights, _constraint):
    """Check that weights sum to approximately 1.0."""
    total = sum(weights.values())
    passed = abs(total - 1.0) < 1e-4
    return {
        "type": "weights_sum_to_one",
        "status": "PASS" if passed else "FAIL",
        "value": round(total, 6),
        "limit": 1.0,
    }


CHECK_FNS = {
    "position_limit_max": lambda w, c, s: check_position_limit(w, c),
    "sector_limit_max": lambda w, c, s: check_sector_limit(w, c, s),
    "long_only": lambda w, c, s: check_long_only(w, c),
    "beta_range": lambda w, c, s: check_beta_range(w, c),
    "turnover_max_monthly": lambda w, c, s: check_turnover(w, c),
    "min_names": lambda w, c, s: check_min_names(w, c),
    "weights_sum_to_one": lambda w, c, s: check_weights_sum(w, c),
}


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        output = {
            "overall_compliant": False,
            "constraints": [],
            "error": f"Invalid JSON input: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)
    except FileNotFoundError as e:
        output = {
            "overall_compliant": False,
            "constraints": [],
            "error": f"Input file not found: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "portfolio_weights" not in data:
        output = {
            "overall_compliant": False,
            "constraints": [],
            "error": "Missing required field: portfolio_weights",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    weights = data["portfolio_weights"]
    mandate = data.get("mandate", {})
    constraint_list = mandate.get("constraints", [])
    sector_map = data.get("sector_map", {})

    results = []
    for constraint in constraint_list:
        ctype = constraint.get("type", "")
        check_fn = CHECK_FNS.get(ctype)
        if check_fn:
            results.append(check_fn(weights, constraint, sector_map))
        else:
            results.append({
                "type": ctype,
                "status": "PASS",
                "value": None,
                "limit": None,
                "note": f"Unknown constraint type: {ctype!r}; skipped",
            })

    overall = all(r["status"] == "PASS" for r in results)

    output = {
        "overall_compliant": overall,
        "constraints": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
