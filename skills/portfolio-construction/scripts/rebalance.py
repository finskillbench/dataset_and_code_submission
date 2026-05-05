#!/usr/bin/env python3
"""
Portfolio rebalancing optimizer.

Given a current portfolio, expected returns, covariance matrix, and an objective,
computes optimal new weights, the trade list, and total turnover.

Usage:
    python rebalance.py --input rebalance_input.json

Input format:
    {
        "current_portfolio": {"TICKER": float, ...},
        "expected_returns": {"TICKER": float, ...},
        "covariance_matrix": [[float, ...], ...],
        "symbols": ["TICKER", ...],
        "objective": "max_sharpe"|"min_variance"|"max_return"|"risk_parity",
        "risk_free_rate": float,
        "constraints": {"long_only": true, "max_weight": float, ...}  (optional)
    }

Output format:
    {
        "status": "optimal"|"infeasible"|"error",
        "new_weights": {"TICKER": float, ...},
        "trade_list": {"TICKER": float, ...},
        "turnover": float,
        "weights": {"TICKER": float, ...},
        "expected_return": float,
        "expected_risk": float
    }
"""

import argparse
import json
import sys

import cvxpy as cp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio rebalancing optimizer")
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
    cov = data["covariance_matrix"]
    if len(cov) != n:
        raise ValueError(f"covariance_matrix has {len(cov)} rows, expected {n}")


def solve_rebalance(mu, Sigma, n, w_current, constraints, objective,
                    risk_free_rate, symbols):
    """Solve portfolio rebalancing using cvxpy."""
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
    if constraints.get("max_turnover"):
        turnover = cp.norm(w - w_current, 1) / 2
        cons.append(turnover <= constraints["max_turnover"])

    # Objective
    if objective == "max_sharpe":
        portfolio_return = mu @ w - risk_free_rate
        portfolio_risk = cp.quad_form(w, Sigma)
        prob_obj = cp.Maximize(portfolio_return - portfolio_risk)
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
    Sigma = (Sigma + Sigma.T) / 2.0

    current_portfolio = data.get("current_portfolio", {})
    w_current = np.array([current_portfolio.get(s, 0.0) for s in symbols], dtype=np.float64)

    constraints = data.get("constraints", {})
    objective = data.get("objective", "max_sharpe")
    risk_free_rate = data.get("risk_free_rate", 0.0)

    weights, status = solve_rebalance(mu, Sigma, n, w_current, constraints,
                                      objective, risk_free_rate, symbols)

    if weights is None:
        output = {
            "status": status,
            "new_weights": {s: 0.0 for s in symbols},
            "trade_list": {},
            "turnover": 0.0,
            "weights": {s: 0.0 for s in symbols},
        }
        print(json.dumps(output, indent=2))
        sys.exit(0)

    new_weight_dict = {s: round(float(weights[i]), 10) for i, s in enumerate(symbols)}

    # Compute trade list and turnover
    trade_list = {}
    total_turnover = 0.0
    for i, s in enumerate(symbols):
        trade = float(weights[i]) - w_current[i]
        if abs(trade) > 1e-8:
            trade_list[s] = round(trade, 10)
        total_turnover += abs(trade)
    turnover = round(total_turnover / 2.0, 10)  # one-way turnover

    # Compute metrics
    port_return = float(mu @ weights)
    port_risk = float(np.sqrt(max(weights @ Sigma @ weights, 0.0)))

    output = {
        "status": status,
        "new_weights": new_weight_dict,
        "trade_list": trade_list,
        "turnover": turnover,
        "weights": new_weight_dict,
        "expected_return": round(port_return, 6),
        "expected_risk": round(port_risk, 6),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
