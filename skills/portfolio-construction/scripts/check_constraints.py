#!/usr/bin/env python3
"""
Portfolio constraint satisfaction checker.

Checks whether a given set of portfolio weights satisfies specified constraints,
including position limits, sector limits, long-only, and weight normalization.

Usage:
    python check_constraints.py --input constraints_input.json
    echo '{"weights": ...}' | python check_constraints.py

Input format:
    {
        "weights": {"TICKER": float, ...},
        "constraints": {
            "max_weight": float,
            "min_weight": float,
            "long_only": bool,
            "sector_limits": {"sector": float},
            "min_names": int,
            "max_sector_concentration": float
        },
        "sector_mapping": {"TICKER": "sector", ...}
    }

Output format:
    {
        "satisfied": bool,
        "violations": [{"constraint": str, "value": float, "limit": float}],
        "constraint_satisfaction": {"max_weight": bool, ...}
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check portfolio constraint satisfaction",
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


def check_constraints(data):
    weights = data["weights"]
    constraints = data.get("constraints", {})
    sector_mapping = data.get("sector_mapping", {})

    violations = []
    satisfaction = {}

    # --- Weight sum check ---
    total_weight = sum(weights.values())
    weight_sum_ok = abs(total_weight - 1.0) < 1e-4
    satisfaction["weights_sum_to_one"] = weight_sum_ok
    if not weight_sum_ok:
        violations.append({
            "constraint": "weights_sum_to_one",
            "value": round(total_weight, 6),
            "limit": 1.0,
        })

    # --- Max weight ---
    max_weight_limit = constraints.get("max_weight")
    if max_weight_limit is not None:
        max_w = max(weights.values())
        max_ok = max_w <= max_weight_limit + 1e-6
        satisfaction["max_weight"] = max_ok
        if not max_ok:
            # Find the worst offender
            worst_ticker = max(weights, key=weights.get)
            violations.append({
                "constraint": f"max_weight ({worst_ticker})",
                "value": round(max_w, 6),
                "limit": max_weight_limit,
            })

    # --- Min weight ---
    min_weight_limit = constraints.get("min_weight")
    if min_weight_limit is not None:
        min_w = min(weights.values())
        min_ok = min_w >= min_weight_limit - 1e-6
        satisfaction["min_weight"] = min_ok
        if not min_ok:
            worst_ticker = min(weights, key=weights.get)
            violations.append({
                "constraint": f"min_weight ({worst_ticker})",
                "value": round(min_w, 6),
                "limit": min_weight_limit,
            })

    # --- Long-only ---
    long_only = constraints.get("long_only", False)
    if long_only:
        any_negative = any(w < -1e-6 for w in weights.values())
        satisfaction["long_only"] = not any_negative
        if any_negative:
            neg_tickers = [t for t, w in weights.items() if w < -1e-6]
            for t in neg_tickers:
                violations.append({
                    "constraint": f"long_only ({t})",
                    "value": round(weights[t], 6),
                    "limit": 0.0,
                })

    # --- Sector limits ---
    sector_limits = constraints.get("sector_limits", {})
    if sector_limits and sector_mapping:
        sector_weights = {}
        for ticker, w in weights.items():
            sector = sector_mapping.get(ticker, "Unknown")
            sector_weights.setdefault(sector, 0.0)
            sector_weights[sector] += w

        for sector, limit in sector_limits.items():
            actual = sector_weights.get(sector, 0.0)
            ok = actual <= limit + 1e-6
            satisfaction[f"sector_limit_{sector}"] = ok
            if not ok:
                violations.append({
                    "constraint": f"sector_limit_{sector}",
                    "value": round(actual, 6),
                    "limit": limit,
                })

    # --- Min names (minimum number of positions) ---
    min_names = constraints.get("min_names")
    if min_names is not None:
        n_positions = sum(1 for w in weights.values() if abs(w) > 1e-6)
        min_names_ok = n_positions >= min_names
        satisfaction["min_names"] = min_names_ok
        if not min_names_ok:
            violations.append({
                "constraint": "min_names",
                "value": n_positions,
                "limit": min_names,
            })

    # --- Max sector concentration ---
    max_sector_conc = constraints.get("max_sector_concentration")
    if max_sector_conc is not None and sector_mapping:
        sector_weights = {}
        for ticker, w in weights.items():
            sector = sector_mapping.get(ticker, "Unknown")
            sector_weights.setdefault(sector, 0.0)
            sector_weights[sector] += w

        max_sector_w = max(sector_weights.values()) if sector_weights else 0.0
        conc_ok = max_sector_w <= max_sector_conc + 1e-6
        satisfaction["max_sector_concentration"] = conc_ok
        if not conc_ok:
            worst_sector = max(sector_weights, key=sector_weights.get)
            violations.append({
                "constraint": f"max_sector_concentration ({worst_sector})",
                "value": round(max_sector_w, 6),
                "limit": max_sector_conc,
            })

    overall_satisfied = len(violations) == 0
    return {
        "satisfied": overall_satisfied,
        "violations": violations,
        "constraint_satisfaction": satisfaction,
    }


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        output = {
            "satisfied": False,
            "violations": [],
            "constraint_satisfaction": {},
            "error": f"Invalid JSON input: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)
    except FileNotFoundError as e:
        output = {
            "satisfied": False,
            "violations": [],
            "constraint_satisfaction": {},
            "error": f"Input file not found: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "weights" not in data:
        output = {
            "satisfied": False,
            "violations": [],
            "constraint_satisfaction": {},
            "error": "Missing required field: weights",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    result = check_constraints(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
