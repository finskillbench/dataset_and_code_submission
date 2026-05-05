#!/usr/bin/env python3
"""
Mean-variance portfolio optimizer using cvxpy.

Accepts JSON input specifying expected returns, covariance matrix, constraints,
and objective function. Returns optimal portfolio weights and risk/return metrics.

Usage:
    python optimize.py --input portfolio_input.json

Input format:
    {
        "expected_returns": {"TICKER": float, ...},
        "covariance_matrix": [[float, ...], ...],
        "symbols": ["TICKER", ...],
        "constraints": {"max_weight": float, "long_only": bool, ...},
        "objective": "max_sharpe"|"min_variance"|"max_return"|"risk_parity",
        "risk_free_rate": float,
        "risk_aversion": float
    }

Output format:
    {
        "status": "optimal"|"infeasible"|"error",
        "weights": {"TICKER": float},
        "expected_return": float,
        "expected_risk": float,
        "sharpe_ratio": float,
        "constraint_satisfaction": {}
    }
"""

import argparse
import json
import sys

import cvxpy as cp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Mean-variance portfolio optimizer")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to JSON input file. If omitted, reads from stdin.")
    return parser.parse_args()


def read_input(path):
    if path:
        with open(path, "r") as f:
            return json.load(f)
    return json.load(sys.stdin)


def validate_input(data):
    required = ["expected_returns", "covariance_matrix", "symbols"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    symbols = data["symbols"]
    n = len(symbols)
    if n == 0:
        raise ValueError("symbols list must not be empty")
    er = data["expected_returns"]
    if len(er) != n:
        raise ValueError(f"expected_returns has {len(er)} entries, expected {n}")
    for s in symbols:
        if s not in er:
            raise ValueError(f"Symbol {s!r} missing from expected_returns")
    cov = data["covariance_matrix"]
    if len(cov) != n:
        raise ValueError(f"covariance_matrix has {len(cov)} rows, expected {n}")
    for i, row in enumerate(cov):
        if len(row) != n:
            raise ValueError(f"covariance_matrix row {i} has {len(row)} columns, expected {n}")


def solve(mu, Sigma, n, constraints, objective, risk_aversion, risk_free_rate, symbols, sector_mapping):
    """Solve portfolio optimization using cvxpy."""
    sym_idx = {s: i for i, s in enumerate(symbols)}

    # Ensure PSD
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.min(eigvals) < 0:
        Sigma = Sigma + (abs(np.min(eigvals)) + 1e-8) * np.eye(n)

    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1]

    if constraints.get("max_weight"):
        cons.append(w <= constraints["max_weight"])
    if constraints.get("min_weight"):
        cons.append(w >= constraints["min_weight"])

    sector_limits = constraints.get("sector_limits", {})
    if sector_limits and sector_mapping:
        for sector, limit in sector_limits.items():
            idx = [sym_idx[s] for s in symbols if sector_mapping.get(s) == sector]
            if idx:
                cons.append(cp.sum(w[idx]) <= limit)

    if constraints.get("max_turnover") and constraints.get("current_portfolio"):
        w_current = np.array([constraints["current_portfolio"].get(s, 0.0) for s in symbols])
        turnover = cp.norm(w - w_current, 1) / 2
        cons.append(turnover <= constraints["max_turnover"])

    if constraints.get("max_tracking_error") and constraints.get("benchmark_weights"):
        w_bench = np.array([constraints["benchmark_weights"].get(s, 0.0) for s in symbols])
        active = w - w_bench
        te_var = cp.quad_form(active, Sigma)
        cons.append(te_var <= constraints["max_tracking_error"] ** 2)

    # Objective
    if objective == "max_sharpe":
        portfolio_return = mu @ w - risk_free_rate
        portfolio_risk = cp.quad_form(w, Sigma)
        prob_obj = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
    elif objective == "min_variance":
        prob_obj = cp.Minimize(cp.quad_form(w, Sigma))
    elif objective == "max_return":
        prob_obj = cp.Maximize(mu @ w)
    elif objective == "risk_parity":
        diag_var = np.diag(Sigma)
        inv_var = 1.0 / np.maximum(diag_var, 1e-10)
        target_w = inv_var / inv_var.sum()
        prob_obj = cp.Minimize(cp.sum_squares(w - target_w) + 0.5 * cp.quad_form(w, Sigma))
    else:
        return None, f"Unknown objective: {objective}"

    prob = cp.Problem(prob_obj, cons)
    try:
        try:
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-6)
        except (cp.SolverError, Exception):
            prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000,
                       eps_abs=1e-6, eps_rel=1e-6)
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
        print(json.dumps({"status": "error", "message": f"Invalid JSON input: {e}"}))
        sys.exit(1)
    except FileNotFoundError as e:
        print(json.dumps({"status": "error", "message": f"Input file not found: {e}"}))
        sys.exit(1)

    try:
        validate_input(data)
    except ValueError as e:
        print(json.dumps({"status": "error", "message": f"Input validation failed: {e}"}))
        sys.exit(1)

    symbols = data["symbols"]
    n = len(symbols)
    mu = np.array([data["expected_returns"][s] for s in symbols], dtype=np.float64)
    Sigma = np.array(data["covariance_matrix"], dtype=np.float64)
    constraints = data.get("constraints", {})
    objective = data.get("objective", "max_sharpe")
    risk_free_rate = data.get("risk_free_rate", 0.0)
    risk_aversion = data.get("risk_aversion", 1.0)
    sector_mapping = data.get("sector_mapping", None)

    # Symmetrize
    Sigma = (Sigma + Sigma.T) / 2.0

    weights, status = solve(mu, Sigma, n, constraints, objective,
                            risk_aversion, risk_free_rate, symbols, sector_mapping)

    if weights is None:
        output = {
            "status": status,
            "weights": {s: 0.0 for s in symbols},
            "expected_return": None,
            "expected_risk": None,
            "sharpe_ratio": None,
            "constraint_satisfaction": {},
        }
        print(json.dumps(output, indent=2))
        sys.exit(0)

    # Compute metrics
    port_return = float(mu @ weights)
    port_risk = float(np.sqrt(max(weights @ Sigma @ weights, 0.0)))
    sharpe = float((port_return - risk_free_rate) / port_risk) if port_risk > 1e-12 else 0.0

    weight_dict = {s: float(weights[i]) for i, s in enumerate(symbols)}

    # Constraint satisfaction
    cs = {}
    if constraints.get("max_weight"):
        cs["max_weight"] = all(v <= constraints["max_weight"] + 1e-6 for v in weight_dict.values())
    if constraints.get("long_only", True):
        cs["long_only"] = all(v >= -1e-6 for v in weight_dict.values())
    if constraints.get("min_names"):
        active = sum(1 for v in weight_dict.values() if v > 1e-4)
        cs["min_names"] = active >= constraints["min_names"]
    if constraints.get("max_names"):
        active = sum(1 for v in weight_dict.values() if v > 1e-4)
        cs["max_names"] = active <= constraints["max_names"]

    output = {
        "status": status,
        "weights": weight_dict,
        "expected_return": round(port_return, 6),
        "expected_risk": round(port_risk, 6),
        "sharpe_ratio": round(sharpe, 4),
        "constraint_satisfaction": cs,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
