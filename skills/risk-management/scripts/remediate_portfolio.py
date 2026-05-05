#!/usr/bin/env python3
"""Generate low-turnover trades to restore mandate compliance."""

from __future__ import annotations

import argparse
import json
import sys

from _common import (
    extend_sector_map,
    numpy_to_python,
    portfolio_weights,
    resolve_covariance,
    resolve_factor_betas,
)
from lib.engines.risk_engine import compute_remediation


def parse_args():
    parser = argparse.ArgumentParser(description="Generate remediation trades")
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
    mandate = data.get("mandate")
    sector_map = data.get("sector_map")
    cov_matrix, cov_symbols = resolve_covariance(data)
    factor_betas = resolve_factor_betas(data)

    if weights is None:
        print(json.dumps({"error": "portfolio_weights not provided"}))
        sys.exit(1)
    if mandate is None:
        print(json.dumps({"error": "mandate not provided"}))
        sys.exit(1)
    if sector_map is None:
        print(json.dumps({"error": "sector_map not provided"}))
        sys.exit(1)
    if cov_matrix is None or cov_symbols is None:
        print(json.dumps({"error": "covariance matrix not provided or resolvable"}))
        sys.exit(1)

    sector_map = extend_sector_map(dict(sector_map), cov_symbols)
    result = compute_remediation(
        current_weights=weights,
        mandate=mandate,
        cov_matrix=cov_matrix,
        symbols=cov_symbols,
        sector_map=sector_map,
        factor_betas=factor_betas,
    )
    output = {
        "trades": result.get("trades", []),
        "total_turnover": result.get("turnover", 0.0),
        "post_trade_compliant": result.get("compliant", False),
        "num_trades": result.get("num_trades", 0),
    }
    print(json.dumps(numpy_to_python(output), indent=2))


if __name__ == "__main__":
    main()
