#!/usr/bin/env python3
"""
Value-at-Risk (VaR) and Conditional VaR (CVaR) computation.

Computes portfolio VaR and CVaR using historical or parametric methods.

Usage:
    python compute_var.py --input var_input.json
    echo '{"weights": [0.5, 0.5], "returns": [[0.01, -0.02], ...], "confidence": 0.95}' \
        | python compute_var.py

Input format:
    {
        "weights": [float, ...],
        "returns": [[float, ...], ...],   (T x N matrix of asset returns)
        "confidence": 0.95,
        "method": "historical"|"parametric"
    }

Output format:
    {
        "var": float,
        "cvar": float,
        "method": str
    }
"""

import argparse
import json
import sys

import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute portfolio VaR and CVaR",
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


def compute_historical_var(portfolio_returns, confidence):
    """Compute VaR and CVaR using the historical simulation method.

    VaR is the quantile of the empirical return distribution at the given
    confidence level. CVaR (Expected Shortfall) is the mean of returns
    below the VaR threshold.
    """
    sorted_returns = np.sort(portfolio_returns)
    n = len(sorted_returns)
    if n == 0:
        return 0.0, 0.0

    # VaR at confidence level: the loss threshold such that P(L > VaR) = 1 - confidence
    # Using the lower quantile (loss convention: positive VaR means loss)
    alpha = 1.0 - confidence
    index = int(np.floor(alpha * n))
    index = max(0, min(index, n - 1))

    var_value = -sorted_returns[index]  # Negate so VaR is positive for losses
    cvar_value = -np.mean(sorted_returns[: index + 1]) if index > 0 else var_value

    return float(var_value), float(cvar_value)


def compute_parametric_var(portfolio_returns, confidence):
    """Compute VaR and CVaR using the parametric (variance-covariance) method.

    Assumes portfolio returns are normally distributed.
    """
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns, ddof=1)

    if sigma < 1e-16:
        return 0.0, 0.0

    z_alpha = stats.norm.ppf(1.0 - confidence)

    # VaR = -(mu + z_alpha * sigma), negated for loss convention
    var_value = -(mu + z_alpha * sigma)

    # CVaR for normal distribution
    # E[X | X < quantile] = mu - sigma * phi(z_alpha) / alpha
    alpha = 1.0 - confidence
    cvar_value = -(mu - sigma * stats.norm.pdf(z_alpha) / alpha)

    return float(var_value), float(cvar_value)


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        output = {"var": 0.0, "cvar": 0.0, "method": "error",
                  "error": f"Invalid JSON input: {e}"}
        print(json.dumps(output, indent=2))
        sys.exit(1)
    except FileNotFoundError as e:
        output = {"var": 0.0, "cvar": 0.0, "method": "error",
                  "error": f"Input file not found: {e}"}
        print(json.dumps(output, indent=2))
        sys.exit(1)

    # Validate inputs
    if "weights" not in data:
        output = {"var": 0.0, "cvar": 0.0, "method": "error",
                  "error": "Missing required field: weights"}
        print(json.dumps(output, indent=2))
        sys.exit(1)

    if "returns" not in data:
        output = {"var": 0.0, "cvar": 0.0, "method": "error",
                  "error": "Missing required field: returns"}
        print(json.dumps(output, indent=2))
        sys.exit(1)

    weights = np.array(data["weights"], dtype=np.float64)
    returns = np.array(data["returns"], dtype=np.float64)
    confidence = data.get("confidence", 0.95)
    method = data.get("method", "historical")

    if len(weights) == 0:
        output = {"var": 0.0, "cvar": 0.0, "method": method}
        print(json.dumps(output, indent=2))
        sys.exit(0)

    if returns.ndim == 1:
        # Single period: returns is a 1D vector, treat as portfolio returns directly
        portfolio_returns = returns
    else:
        # T x N matrix: compute portfolio returns as weighted sum
        if returns.shape[1] != len(weights):
            output = {
                "var": 0.0,
                "cvar": 0.0,
                "method": "error",
                "error": (
                    f"Dimension mismatch: returns has {returns.shape[1]} columns "
                    f"but weights has {len(weights)} elements"
                ),
            }
            print(json.dumps(output, indent=2))
            sys.exit(1)
        portfolio_returns = returns @ weights

    if method == "historical":
        var_val, cvar_val = compute_historical_var(portfolio_returns, confidence)
    elif method == "parametric":
        var_val, cvar_val = compute_parametric_var(portfolio_returns, confidence)
    else:
        output = {
            "var": 0.0,
            "cvar": 0.0,
            "method": "error",
            "error": f"Unknown method: {method!r}. Use 'historical' or 'parametric'.",
        }
        print(json.dumps(output, indent=2))
        sys.exit(1)

    output = {
        "var": round(var_val, 8),
        "cvar": round(cvar_val, 8),
        "method": method,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
