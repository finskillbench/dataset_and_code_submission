#!/usr/bin/env python3
"""
Black-Litterman model: posterior returns and optimal portfolio.

Combines market-implied equilibrium returns with analyst views to produce
posterior expected returns, then optimizes to get optimal weights.

Usage:
    python black_litterman.py --input bl_input.json

Input format:
    {
        "covariance_matrix": [[float, ...], ...],
        "symbols": ["TICKER", ...],
        "expected_returns": {"TICKER": float, ...},
        "market_cap_weights": {"TICKER": float, ...},
        "risk_free_rate": float,
        "analyst_views": [
            {
                "view": {
                    "type": "absolute"|"relative",
                    "symbols": ["TICKER", ...],
                    "return": float,
                    "confidence": float
                }
            },
            ...
        ],
        "risk_aversion": float,
        "tau": float
    }

Output format:
    {
        "status": "optimal"|"error",
        "posterior_returns": {"TICKER": float, ...},
        "optimal_weights": {"TICKER": float, ...}
    }
"""

import argparse
import json
import sys

import cvxpy as cp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Black-Litterman model")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to JSON input file. If omitted, reads from stdin.")
    return parser.parse_args()


def read_input(path):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    return json.load(sys.stdin)


def compute_equilibrium_returns(Sigma, w_mkt, risk_aversion):
    """Compute implied equilibrium returns: pi = delta * Sigma * w_mkt."""
    return risk_aversion * Sigma @ w_mkt


def build_view_matrices(views, symbols, Sigma):
    """Build the P (pick) matrix and Q (view returns) vector.

    Supports:
      - absolute views: P row has 1.0 for the single symbol
      - relative views: P row has +1 for first symbol, -1 for second symbol
        (meaning: first symbol outperforms second by Q amount)
    """
    sym_idx = {s: i for i, s in enumerate(symbols)}
    n = len(symbols)
    P_rows = []
    Q_vals = []
    omega_diags = []

    for v in views:
        view = v["view"]
        vtype = view["type"]
        vsymbols = view["symbols"]
        vreturn = view["return"]
        confidence = view.get("confidence", 0.5)

        row = np.zeros(n)
        if vtype == "absolute":
            if vsymbols[0] in sym_idx:
                row[sym_idx[vsymbols[0]]] = 1.0
        elif vtype == "relative":
            if len(vsymbols) >= 2:
                if vsymbols[0] in sym_idx:
                    row[sym_idx[vsymbols[0]]] = 1.0
                if vsymbols[1] in sym_idx:
                    row[sym_idx[vsymbols[1]]] = -1.0
        else:
            continue

        if np.any(row != 0):
            P_rows.append(row)
            Q_vals.append(vreturn)
            # Omega diagonal: variance of view, scaled by confidence
            # Lower confidence → higher variance → less weight on view
            view_var = float(row @ Sigma @ row)
            omega_diags.append(view_var * (1.0 - confidence) / max(confidence, 1e-6))

    if not P_rows:
        return None, None, None

    P = np.array(P_rows)
    Q = np.array(Q_vals)
    Omega = np.diag(omega_diags)
    return P, Q, Omega


def compute_posterior_returns(pi, Sigma, P, Q, Omega, tau):
    """Compute Black-Litterman posterior returns.

    Formula:
        posterior = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                    * [(tau*Sigma)^-1 * pi + P'*Omega^-1 * Q]
    """
    tau_Sigma = tau * Sigma
    tau_Sigma_inv = np.linalg.inv(tau_Sigma)
    Omega_inv = np.linalg.inv(Omega)

    # Posterior precision
    M = tau_Sigma_inv + P.T @ Omega_inv @ P
    M_inv = np.linalg.inv(M)

    # Posterior mean
    posterior = M_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
    return posterior


def optimize_weights(mu_posterior, Sigma, n, risk_aversion):
    """Optimize portfolio weights given posterior returns."""
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.min(eigvals) < 0:
        Sigma = Sigma + (abs(np.min(eigvals)) + 1e-8) * np.eye(n)

    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1]

    portfolio_return = mu_posterior @ w
    portfolio_risk = cp.quad_form(w, Sigma)
    prob_obj = cp.Maximize(portfolio_return - (risk_aversion / 2.0) * portfolio_risk)

    prob = cp.Problem(prob_obj, cons)
    try:
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-6)
    except Exception:
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000)
        except Exception as e:
            return None, f"Solver error: {e}"

    if w.value is None or prob.status in ("infeasible", "unbounded"):
        return None, prob.status

    weights = np.array(w.value).flatten()
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    return weights, "optimal"


def main():
    args = parse_args()

    try:
        data = read_input(args.input)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}))
        sys.exit(1)
    except FileNotFoundError as e:
        print(json.dumps({"status": "error", "message": f"File not found: {e}"}))
        sys.exit(1)

    # Validate
    if "covariance_matrix" not in data or "symbols" not in data:
        print(json.dumps({"status": "error",
                          "message": "Missing covariance_matrix or symbols"}))
        sys.exit(1)
    if "analyst_views" not in data:
        print(json.dumps({"status": "error",
                          "message": "Missing analyst_views"}))
        sys.exit(1)

    symbols = data["symbols"]
    n = len(symbols)
    Sigma = np.array(data["covariance_matrix"], dtype=np.float64)
    Sigma = (Sigma + Sigma.T) / 2.0

    risk_free_rate = data.get("risk_free_rate", 0.0)
    risk_aversion = data.get("risk_aversion", 2.5)
    tau = data.get("tau", 0.05)

    # Market-cap weights for equilibrium returns
    mcw = data.get("market_cap_weights", {})
    if mcw and len(mcw) > 0:
        w_mkt = np.array([mcw.get(s, 1.0 / n) for s in symbols], dtype=np.float64)
        w_mkt = w_mkt / w_mkt.sum()
    else:
        # If no market-cap weights, use equal weight as prior
        w_mkt = np.ones(n) / n

    # Compute equilibrium (implied) returns
    pi = compute_equilibrium_returns(Sigma, w_mkt, risk_aversion)

    # Build view matrices
    views = data["analyst_views"]
    P, Q, Omega = build_view_matrices(views, symbols, Sigma)

    if P is None:
        # No valid views — return equilibrium
        posterior = pi
    else:
        posterior = compute_posterior_returns(pi, Sigma, P, Q, Omega, tau)

    posterior_dict = {s: round(float(posterior[i]), 10) for i, s in enumerate(symbols)}

    # Optimize with posterior returns
    weights, status = optimize_weights(posterior, Sigma, n, risk_aversion)

    if weights is None:
        output = {
            "status": status,
            "posterior_returns": posterior_dict,
            "optimal_weights": {s: 0.0 for s in symbols},
        }
        print(json.dumps(output, indent=2))
        sys.exit(0)

    weight_dict = {s: round(float(weights[i]), 10) for i, s in enumerate(symbols)}

    output = {
        "status": "optimal",
        "posterior_returns": posterior_dict,
        "optimal_weights": weight_dict,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
