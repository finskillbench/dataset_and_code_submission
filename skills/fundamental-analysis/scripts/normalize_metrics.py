#!/usr/bin/env python3
"""
GAAP metric normalization helpers.

Applies adjustments to raw financial metrics to produce normalized figures.
Handles non-recurring items such as restructuring charges, impairments,
gains/losses on asset sales, and other one-time items.

Usage:
    python normalize_metrics.py --input normalize_input.json
    echo '{"metrics": {"operating_income": 100}, "adjustments": [...]}' \
        | python normalize_metrics.py

Input format:
    {
        "metrics": {
            "revenue": float,
            "operating_income": float,
            "net_income": float,
            ...
        },
        "adjustments": [
            {"item": "restructuring_charge", "amount": float, "metric": "operating_income"},
            {"item": "impairment_loss", "amount": float, "metric": "operating_income"},
            ...
        ]
    }

Output format:
    {
        "normalized": {"metric": value, ...},
        "adjustments_applied": int
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply GAAP normalization adjustments to financial metrics",
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


# Common non-recurring item categories and their default sign conventions.
# Positive amounts mean the item reduced the metric (add back to normalize).
# Negative amounts mean the item increased the metric (subtract to normalize).
# The "amount" field in the adjustment specifies the magnitude; the sign of
# the amount determines direction: positive = add back, negative = subtract.
ADJUSTABLE_METRICS = {
    "operating_income",
    "net_income",
    "ebitda",
    "operating_margin",
}


def normalize_metrics(data):
    metrics = dict(data.get("metrics", {}))
    adjustments = data.get("adjustments", [])

    adjustments_applied = 0

    for adj in adjustments:
        item_name = adj.get("item", "unknown")
        amount = adj.get("amount", 0.0)
        target_metric = adj.get("metric", None)

        if target_metric is None:
            # Try to infer the target metric from common item names
            target_metric = infer_target_metric(item_name)

        if target_metric and target_metric in metrics:
            # Add back the adjustment amount to normalize
            # Convention: positive amount = expense to add back
            #             negative amount = gain to subtract
            metrics[target_metric] = metrics[target_metric] + amount
            adjustments_applied += 1

    # Recompute derived metrics if their inputs were adjusted
    metrics = recompute_derived(metrics)

    return {
        "normalized": metrics,
        "adjustments_applied": adjustments_applied,
    }


def infer_target_metric(item_name):
    """Infer which metric an adjustment should apply to based on item name."""
    item_lower = item_name.lower()

    # Items that affect operating income
    operating_items = [
        "restructuring",
        "impairment",
        "asset_sale",
        "asset_write_down",
        "write_down",
        "goodwill_impairment",
        "severance",
        "reorganization",
    ]
    for keyword in operating_items:
        if keyword in item_lower:
            return "operating_income"

    # Items that only affect net income (below the operating line)
    below_line_items = [
        "tax_settlement",
        "legal_settlement",
        "discontinued_operations",
        "extraordinary",
    ]
    for keyword in below_line_items:
        if keyword in item_lower:
            return "net_income"

    # Default to operating_income
    return "operating_income"


def recompute_derived(metrics):
    """Recompute derived metrics from their components if available."""
    # Operating margin
    if "operating_margin" not in metrics:
        if "operating_income" in metrics and "revenue" in metrics:
            rev = metrics["revenue"]
            if rev != 0:
                metrics["operating_margin"] = metrics["operating_income"] / rev

    # EBITDA (if D&A is available)
    if "ebitda" not in metrics:
        if "operating_income" in metrics and "depreciation_amortization" in metrics:
            metrics["ebitda"] = (
                metrics["operating_income"] + metrics["depreciation_amortization"]
            )

    # Net margin
    if "net_margin" not in metrics:
        if "net_income" in metrics and "revenue" in metrics:
            rev = metrics["revenue"]
            if rev != 0:
                metrics["net_margin"] = metrics["net_income"] / rev

    # ROE
    if "roe" not in metrics:
        if "net_income" in metrics and "stockholders_equity" in metrics:
            equity = metrics["stockholders_equity"]
            if equity != 0:
                metrics["roe"] = metrics["net_income"] / equity

    # ROA
    if "roa" not in metrics:
        if "net_income" in metrics and "total_assets" in metrics:
            assets = metrics["total_assets"]
            if assets != 0:
                metrics["roa"] = metrics["net_income"] / assets

    return metrics


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        output = {
            "normalized": {},
            "adjustments_applied": 0,
            "error": f"Invalid JSON input: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)
    except FileNotFoundError as e:
        output = {
            "normalized": {},
            "adjustments_applied": 0,
            "error": f"Input file not found: {e}",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "metrics" not in data:
        output = {
            "normalized": {},
            "adjustments_applied": 0,
            "error": "Missing required field: metrics",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    result = normalize_metrics(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
