#!/usr/bin/env python3
"""Compute VaR, CVaR, concentration, factor exposure, and mandate compliance."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from _common import numpy_to_python, portfolio_weights, resolve_factor_betas
from lib.engines.risk_engine import (
    check_constraints,
    compute_concentration_metrics,
    compute_factor_exposures,
    compute_portfolio_var,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute portfolio risk report")
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
    symbols = data.get("symbols") or (list(weights.keys()) if weights else None)
    returns_data = data.get("returns_data")
    confidence = data.get("confidence_level", data.get("confidence", 0.95))
    method = data.get("method", "historical")
    mandate = data.get("mandate")
    sector_map = data.get("sector_map")

    if weights is None:
        print(json.dumps({"error": "weights not provided"}))
        sys.exit(1)
    if symbols is None:
        print(json.dumps({"error": "symbols not provided"}))
        sys.exit(1)
    if returns_data is None:
        print(json.dumps({"error": "returns_data not provided"}))
        sys.exit(1)

    weights_vec = np.array([weights.get(s, 0.0) for s in symbols])
    returns_df = pd.DataFrame(returns_data, columns=symbols)
    var, cvar = compute_portfolio_var(weights_vec, returns_df, confidence, method)
    concentration = compute_concentration_metrics(weights_vec)

    factor_exp = {}
    factor_betas = resolve_factor_betas(data)
    if factor_betas:
        factor_df = pd.DataFrame(
            {sym: betas for sym, betas in factor_betas.items() if sym in symbols}
        ).T
        if not factor_df.empty:
            factor_df = factor_df.reindex(symbols)
            factor_exp = compute_factor_exposures(weights_vec, factor_df)

    mandate_results = []
    if mandate:
        mandate_with_context = dict(mandate)
        if sector_map:
            mandate_with_context["_sector_map"] = sector_map
        mandate_with_context["_portfolio_beta"] = factor_exp.get("mkt_rf", 1.0)
        mandate_with_context["_factor_exposures"] = factor_exp
        mandate_results = check_constraints(weights, mandate_with_context)

    output = {
        "var": var,
        "cvar": cvar,
        "concentration": concentration,
        "factor_exposures": factor_exp,
        "mandate_compliance": mandate_results,
    }
    print(json.dumps(numpy_to_python(output), indent=2))


if __name__ == "__main__":
    main()
