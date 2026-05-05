#!/usr/bin/env python3
"""
Compute earnings quality metrics: Piotroski F-Score, Beneish M-Score,
accruals ratio, and income quality ratio.

Requires current and prior period financial data from the XBRL panel.

Usage:
    python earnings_quality.py --input input.json

Input format:
    {
        "current": {
            "revenue": float,
            "operating_income": float,
            "net_income": float,
            "total_assets": float,
            "total_liabilities": float,
            "stockholders_equity": float,
            "operating_cash_flow": float,
            "depreciation_amortization": float,
            "shares_outstanding": float
        },
        "prior": {
            "revenue": float,
            "operating_income": float,
            "net_income": float,
            "total_assets": float,
            "total_liabilities": float,
            "stockholders_equity": float,
            "operating_cash_flow": float,
            "depreciation_amortization": float,
            "shares_outstanding": float
        }
    }

Output format:
    {
        "piotroski_score": int,
        "piotroski_components": {...},
        "beneish_m_score": float,
        "beneish_flag": bool,
        "accruals_ratio": float,
        "income_quality_ratio": float,
        "flags": [...]
    }
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Compute earnings quality metrics")
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


def compute_piotroski(current, prior):
    """Compute Piotroski F-Score (9 binary components)."""
    components = {}

    # Current period metrics
    ta_curr = current.get("total_assets", 0)
    ta_prior = prior.get("total_assets", 0)
    avg_assets = (ta_curr + ta_prior) / 2 if (ta_curr and ta_prior) else ta_curr

    ni_curr = current.get("net_income", 0)
    ni_prior = prior.get("net_income", 0)
    cfo_curr = current.get("operating_cash_flow", 0)
    rev_curr = current.get("revenue", 0)
    rev_prior = prior.get("revenue", 0)
    tl_curr = current.get("total_liabilities", 0)
    tl_prior = prior.get("total_liabilities", 0)
    shares_curr = current.get("shares_outstanding", 1)
    shares_prior = prior.get("shares_outstanding", 1)

    # Profitability signals
    roa_curr = safe_div(ni_curr, avg_assets)
    roa_prior = safe_div(ni_prior, ta_prior)

    # 1. ROA positive
    components["roa_positive"] = 1 if roa_curr > 0 else 0

    # 2. CFO positive
    components["cfo_positive"] = 1 if cfo_curr > 0 else 0

    # 3. Delta ROA positive (improving)
    components["delta_roa_positive"] = 1 if roa_curr > roa_prior else 0

    # 4. Accruals negative (CFO > net income, quality earnings)
    components["accruals_negative"] = 1 if cfo_curr > ni_curr else 0

    # Leverage signals
    leverage_curr = safe_div(tl_curr, ta_curr)
    leverage_prior = safe_div(tl_prior, ta_prior)

    # 5. Delta leverage negative (decreasing leverage)
    components["delta_leverage_negative"] = 1 if leverage_curr < leverage_prior else 0

    # 6. Delta current ratio positive
    cl_curr = tl_curr * 0.4  # Approximate current liabilities
    cl_prior = tl_prior * 0.4
    ca_curr = ta_curr * 0.3  # Approximate current assets
    ca_prior = ta_prior * 0.3
    cr_curr = safe_div(ca_curr, cl_curr, 1.0)
    cr_prior = safe_div(ca_prior, cl_prior, 1.0)
    components["delta_current_ratio_positive"] = 1 if cr_curr > cr_prior else 0

    # 7. No dilution (shares not increased)
    components["no_dilution"] = 1 if shares_curr <= shares_prior else 0

    # Efficiency signals
    gm_curr = safe_div(rev_curr - current.get("cost_of_revenue", rev_curr * 0.6), rev_curr)
    gm_prior = safe_div(rev_prior - prior.get("cost_of_revenue", rev_prior * 0.6), rev_prior)

    # 8. Delta gross margin positive
    # Use operating margin as proxy if cost_of_revenue not available
    om_curr = safe_div(current.get("operating_income", 0), rev_curr)
    om_prior = safe_div(prior.get("operating_income", 0), rev_prior)
    components["delta_gross_margin_positive"] = 1 if om_curr > om_prior else 0

    # 9. Delta asset turnover positive
    at_curr = safe_div(rev_curr, avg_assets)
    at_prior = safe_div(rev_prior, ta_prior)
    components["delta_asset_turnover_positive"] = 1 if at_curr > at_prior else 0

    score = sum(components.values())
    return score, components


def compute_beneish(current, prior):
    """Compute Beneish M-Score (8-variable model).

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    """
    rev_curr = current.get("revenue", 1)
    rev_prior = prior.get("revenue", 1)
    ta_curr = current.get("total_assets", 1)
    ta_prior = prior.get("total_assets", 1)
    ni_curr = current.get("net_income", 0)
    cfo_curr = current.get("operating_cash_flow", 0)
    oi_curr = current.get("operating_income", 0)
    oi_prior = prior.get("operating_income", 0)
    da_curr = current.get("depreciation_amortization", 0)
    da_prior = prior.get("depreciation_amortization", 0)
    tl_curr = current.get("total_liabilities", 0)
    tl_prior = prior.get("total_liabilities", 0)

    # DSRI: Days Sales in Receivables Index
    # Approximate receivables as 10% of revenue (common proxy)
    rec_curr = rev_curr * 0.10
    rec_prior = rev_prior * 0.10
    dsri = safe_div(rec_curr / rev_curr, rec_prior / rev_prior, 1.0) if rev_prior else 1.0

    # GMI: Gross Margin Index
    gm_curr = safe_div(rev_curr - (rev_curr - oi_curr - da_curr), rev_curr, 0.5)
    gm_prior = safe_div(rev_prior - (rev_prior - oi_prior - da_prior), rev_prior, 0.5)
    gmi = safe_div(gm_prior, gm_curr, 1.0)

    # AQI: Asset Quality Index
    ppe_curr = ta_curr * 0.3  # Approximate
    ppe_prior = ta_prior * 0.3
    aqi_curr = safe_div(ta_curr - ppe_curr - (ta_curr * 0.3), ta_curr, 0.5)
    aqi_prior = safe_div(ta_prior - ppe_prior - (ta_prior * 0.3), ta_prior, 0.5)
    aqi = safe_div(aqi_curr, aqi_prior, 1.0)

    # SGI: Sales Growth Index
    sgi = safe_div(rev_curr, rev_prior, 1.0)

    # DEPI: Depreciation Index
    depi_curr = safe_div(da_curr, da_curr + ppe_curr, 0.1)
    depi_prior = safe_div(da_prior, da_prior + ppe_prior, 0.1)
    depi = safe_div(depi_prior, depi_curr, 1.0)

    # SGAI: SGA Expense Index
    sga_curr = rev_curr - oi_curr - da_curr  # Approximate SGA
    sga_prior = rev_prior - oi_prior - da_prior
    sgai_curr = safe_div(sga_curr, rev_curr, 0.3)
    sgai_prior = safe_div(sga_prior, rev_prior, 0.3)
    sgai = safe_div(sgai_curr, sgai_prior, 1.0)

    # TATA: Total Accruals to Total Assets
    tata = safe_div(ni_curr - cfo_curr, ta_curr, 0.0)

    # LVGI: Leverage Index
    lvgi_curr = safe_div(tl_curr, ta_curr, 0.5)
    lvgi_prior = safe_div(tl_prior, ta_prior, 0.5)
    lvgi = safe_div(lvgi_curr, lvgi_prior, 1.0)

    # M-Score formula
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    return round(m_score, 4)


def compute_earnings_quality(data):
    current = data.get("current", {})
    prior = data.get("prior", {})

    if not current:
        return {"error": "Missing 'current' period data"}

    # If no prior data, use current as prior (delta components will be 0)
    if not prior:
        prior = current

    # Piotroski F-Score
    piotroski_score, piotroski_components = compute_piotroski(current, prior)

    # Beneish M-Score
    beneish_m_score = compute_beneish(current, prior)
    beneish_flag = beneish_m_score > -1.78

    # Accruals ratio: (net_income - operating_cash_flow) / total_assets
    ni = current.get("net_income", 0)
    cfo = current.get("operating_cash_flow", 0)
    ta = current.get("total_assets", 1)
    accruals_ratio = round((ni - cfo) / ta, 6) if ta else 0.0

    # Income quality ratio: operating_cash_flow / net_income
    income_quality_ratio = round(safe_div(cfo, ni, 0.0), 4)

    # Flags
    flags = []
    if piotroski_score <= 3:
        flags.append("low_piotroski")
    if beneish_flag:
        flags.append("high_beneish_manipulation_risk")
    if abs(accruals_ratio) > 0.10:
        flags.append("high_accruals")
    if income_quality_ratio < 0:
        flags.append("negative_income_quality")

    return {
        "piotroski_score": piotroski_score,
        "piotroski_components": piotroski_components,
        "beneish_m_score": beneish_m_score,
        "beneish_flag": beneish_flag,
        "accruals_ratio": accruals_ratio,
        "income_quality_ratio": income_quality_ratio,
        "flags": flags,
    }


def main():
    args = parse_args()
    try:
        data = read_input(args.input)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    result = compute_earnings_quality(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
