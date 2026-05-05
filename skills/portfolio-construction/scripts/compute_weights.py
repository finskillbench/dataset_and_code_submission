#!/usr/bin/env python3
"""
Portfolio weight normalization utility.

Normalizes a set of portfolio weights to sum to 1.0 and reports summary statistics.

Usage:
    python compute_weights.py --input weights_input.json
    echo '{"weights": {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.3}}' | python compute_weights.py

Input format:
    {
        "weights": {"TICKER": float, ...}
    }

Output format:
    {
        "weights": {"TICKER": float, ...},
        "sum": float,
        "n_positions": int,
        "max_weight": float,
        "min_weight": float
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize portfolio weights to sum to 1.0",
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


def normalize_weights(data):
    weights = data["weights"]

    if not weights:
        return {
            "weights": {},
            "sum": 0.0,
            "n_positions": 0,
            "max_weight": 0.0,
            "min_weight": 0.0,
        }

    raw_sum = sum(weights.values())

    if abs(raw_sum) < 1e-16:
        # Cannot normalize zero-sum weights; return equal weight
        n = len(weights)
        normalized = {k: round(1.0 / n, 10) for k in weights}
        return {
            "weights": normalized,
            "sum": 1.0,
            "n_positions": n,
            "max_weight": round(1.0 / n, 10),
            "min_weight": round(1.0 / n, 10),
        }

    normalized = {k: round(v / raw_sum, 10) for k, v in weights.items()}

    # Count non-zero positions
    n_positions = sum(1 for v in normalized.values() if abs(v) > 1e-8)
    max_w = max(normalized.values())
    min_w = min(normalized.values())
    final_sum = round(sum(normalized.values()), 10)

    return {
        "weights": normalized,
        "sum": final_sum,
        "n_positions": n_positions,
        "max_weight": round(max_w, 10),
        "min_weight": round(min_w, 10),
    }


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        output = {"error": f"Invalid JSON input: {e}"}
        print(json.dumps(output, indent=2))
        sys.exit(1)
    except FileNotFoundError as e:
        output = {"error": f"Input file not found: {e}"}
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "weights" not in data:
        output = {"error": "Missing required field: weights"}
        print(json.dumps(output, indent=2))
        sys.exit(1)

    result = normalize_weights(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
