#!/usr/bin/env python3
"""
Compute revenue driver decomposition between two periods.

Compares current vs prior period financials to identify top revenue drivers
and margin changes. Works with aggregate data when segment data is unavailable.

Usage:
    python driver_decomposition.py --input input.json

Input format:
    {
        "current": {
            "revenue": float,
            "operating_income": float,
            "net_income": float,
            "depreciation_amortization": float,
            ...
        },
        "prior": {
            "revenue": float,
            "operating_income": float,
            "net_income": float,
            "depreciation_amortization": float,
            ...
        },
        "segments": {
            "product": {"segment_name": {"current": float, "prior": float}, ...},
            "geographic": {"segment_name": {"current": float, "prior": float}, ...}
        }
    }

    Note: "segments" is optional. If not provided, only aggregate metrics
    and margin drivers will be computed.

Output format:
    {
        "revenue_delta": float,
        "top_revenue_drivers": [...],
        "margin_drivers": [...]
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Compute revenue driver decomposition")
    parser.add_argument("--input", type=str, default=None)
    return parser.parse_args()


def read_input(path):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    return json.load(sys.stdin)


def safe_div(a, b, default=0.0):
    if b is None or b == 0:
        return default
    return a / b


def compute_segment_drivers(segments, revenue_delta):
    """Extract top revenue drivers from segment data."""
    drivers = []

    for seg_type, seg_data in segments.items():
        type_label = f"{seg_type}_segment"
        for seg_name, values in seg_data.items():
            curr = values.get("current", 0)
            prior = values.get("prior", 0)
            delta = curr - prior
            if delta == 0:
                continue
            direction = "increase" if delta > 0 else "decrease"
            contribution = abs(delta) / abs(revenue_delta) * 100 if revenue_delta != 0 else 0
            drivers.append({
                "driver": seg_name,
                "type": type_label,
                "direction": direction,
                "magnitude_usd": delta,
                "contribution_pct": round(contribution, 2),
            })

    # Sort by absolute magnitude
    drivers.sort(key=lambda d: abs(d["magnitude_usd"]), reverse=True)
    return drivers[:5]


def compute_margin_drivers(current, prior):
    """Compute margin change drivers."""
    drivers = []
    rev_curr = current.get("revenue", 1)
    rev_prior = prior.get("revenue", 1)

    # Operating margin
    om_curr = safe_div(current.get("operating_income", 0), rev_curr) * 100
    om_prior = safe_div(prior.get("operating_income", 0), rev_prior) * 100
    delta_om = om_curr - om_prior
    if abs(delta_om) > 0.01:
        drivers.append({
            "driver": "operating_margin",
            "type": "margin",
            "direction": "increase" if delta_om > 0 else "decrease",
            "prior_pct": round(om_prior, 2),
            "current_pct": round(om_curr, 2),
            "delta_pp": round(delta_om, 2),
        })

    # Net margin
    nm_curr = safe_div(current.get("net_income", 0), rev_curr) * 100
    nm_prior = safe_div(prior.get("net_income", 0), rev_prior) * 100
    delta_nm = nm_curr - nm_prior
    if abs(delta_nm) > 0.01:
        drivers.append({
            "driver": "net_margin",
            "type": "margin",
            "direction": "increase" if delta_nm > 0 else "decrease",
            "prior_pct": round(nm_prior, 2),
            "current_pct": round(nm_curr, 2),
            "delta_pp": round(delta_nm, 2),
        })

    # D&A as % of revenue
    da_curr = current.get("depreciation_amortization", 0)
    da_prior = prior.get("depreciation_amortization", 0)
    da_pct_curr = safe_div(da_curr, rev_curr) * 100
    da_pct_prior = safe_div(da_prior, rev_prior) * 100
    delta_da = da_pct_curr - da_pct_prior
    if abs(delta_da) > 0.01:
        drivers.append({
            "driver": "depreciation_ratio",
            "type": "margin",
            "direction": "increase" if delta_da > 0 else "decrease",
            "prior_pct": round(da_pct_prior, 2),
            "current_pct": round(da_pct_curr, 2),
            "delta_pp": round(delta_da, 2),
        })

    # OpEx ratio (SGA proxy)
    opex_curr = rev_curr - current.get("operating_income", 0) - da_curr
    opex_prior = rev_prior - prior.get("operating_income", 0) - da_prior
    opex_pct_curr = safe_div(opex_curr, rev_curr) * 100
    opex_pct_prior = safe_div(opex_prior, rev_prior) * 100
    delta_opex = opex_pct_curr - opex_pct_prior
    if abs(delta_opex) > 0.01:
        drivers.append({
            "driver": "opex_ratio",
            "type": "margin",
            "direction": "increase" if delta_opex > 0 else "decrease",
            "prior_pct": round(opex_pct_prior, 2),
            "current_pct": round(opex_pct_curr, 2),
            "delta_pp": round(delta_opex, 2),
        })

    return drivers


def compute_decomposition(data):
    current = data.get("current", {})
    prior = data.get("prior", {})
    segments = data.get("segments", {})

    if not current or not prior:
        return {"error": "Both 'current' and 'prior' period data required"}

    rev_curr = current.get("revenue", 0)
    rev_prior = prior.get("revenue", 0)
    revenue_delta = rev_curr - rev_prior

    # Revenue drivers from segments (if available)
    if segments:
        top_drivers = compute_segment_drivers(segments, revenue_delta)
    else:
        # Without segment data, report aggregate revenue change
        direction = "increase" if revenue_delta > 0 else "decrease"
        top_drivers = [{
            "driver": "Total Revenue",
            "type": "aggregate",
            "direction": direction,
            "magnitude_usd": revenue_delta,
            "contribution_pct": 100.0,
        }]

    # Margin drivers
    margin_drivers = compute_margin_drivers(current, prior)

    return {
        "revenue_delta": revenue_delta,
        "top_revenue_drivers": top_drivers,
        "margin_drivers": margin_drivers,
    }


def main():
    args = parse_args()
    try:
        data = read_input(args.input)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    result = compute_decomposition(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
