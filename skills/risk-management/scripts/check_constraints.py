#!/usr/bin/env python3
"""Check portfolio compliance against a mandate."""

from __future__ import annotations

import argparse
import json
import sys

from _common import numpy_to_python, portfolio_weights
from lib.engines.risk_engine import check_constraints


def parse_args():
    parser = argparse.ArgumentParser(description="Check mandate constraints")
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
    factor_exposures = data.get("factor_exposures") or data.get("factor_betas")

    if weights is None:
        print(json.dumps({"error": "portfolio_weights not provided"}))
        sys.exit(1)
    if mandate is None:
        print(json.dumps({"error": "mandate not provided"}))
        sys.exit(1)

    check_mandate = dict(mandate)
    if sector_map:
        check_mandate["_sector_map"] = sector_map
    if factor_exposures:
        check_mandate["_factor_exposures"] = factor_exposures

    results = check_constraints(weights, check_mandate)
    output = {
        "overall_compliant": all(r.get("status") == "PASS" for r in results),
        "violations_count": sum(1 for r in results if r.get("status") == "FAIL"),
        "constraints": results,
    }
    print(json.dumps(numpy_to_python(output), indent=2))


if __name__ == "__main__":
    main()
