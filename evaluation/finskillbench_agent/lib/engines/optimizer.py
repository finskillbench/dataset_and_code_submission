"""
Portfolio optimization engine.

Provides:
  - mean-variance optimization via cvxpy (max Sharpe, min variance, max return, risk parity)
  - Black-Litterman model
  - Constraint handling (long-only, sector limits, turnover, tracking error, factor neutrality)

Internalized from scripts/portfolio_construction/_optimizer.py for standalone use.
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None


def optimize_portfolio(
    expected_returns: dict[str, float],
    covariance_matrix: np.ndarray,
    symbols: list[str],
    constraints: dict,
    benchmark_weights: dict[str, float] | None = None,
    current_portfolio: dict[str, float] | None = None,
    objective: str = "max_sharpe",
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    sector_mapping: dict[str, str] | None = None,
    factor_exposures: dict[str, dict] | None = None,
) -> dict:
    """Optimize a portfolio allocation using cvxpy."""
    if cp is None:
        return _fallback_optimization(
            expected_returns, covariance_matrix, symbols, constraints,
            benchmark_weights, objective,
        )

    N = len(symbols)
    sym_idx = {s: i for i, s in enumerate(symbols)}

    mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
    Sigma = covariance_matrix.copy()

    # Ensure PSD
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.min(eigvals) < 0:
        Sigma = Sigma + (abs(np.min(eigvals)) + 1e-8) * np.eye(N)

    w = cp.Variable(N, nonneg=True)
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

    min_names = constraints.get("min_names", 0)
    max_names = constraints.get("max_names", N)

    if constraints.get("max_turnover") and current_portfolio:
        w_current = np.array([current_portfolio.get(s, 0.0) for s in symbols])
        turnover = cp.norm(w - w_current, 1) / 2
        cons.append(turnover <= constraints["max_turnover"])

    if constraints.get("max_tracking_error") and benchmark_weights:
        w_bench = np.array([benchmark_weights.get(s, 0.0) for s in symbols])
        active = w - w_bench
        te_var = cp.quad_form(active, Sigma)
        cons.append(te_var <= constraints["max_tracking_error"] ** 2)

    if constraints.get("factor_neutrality") and factor_exposures:
        factors = set()
        for sym in symbols:
            if sym in factor_exposures:
                factors.update(factor_exposures[sym].keys())
        for factor in sorted(factors):
            exposure = np.array([
                factor_exposures.get(s, {}).get(factor, 0.0) for s in symbols
            ])
            cons.append(w @ exposure == 0.0)

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
        return {
            "status": "error", "weights": {},
            "expected_return": 0.0, "expected_risk": 0.0, "sharpe_ratio": 0.0,
            "active_weights": {}, "tracking_error": 0.0, "turnover": 0.0,
            "constraint_satisfaction": {},
            "diagnostics": {"error": f"Unknown objective: {objective}"},
        }

    prob = cp.Problem(prob_obj, cons)
    try:
        try:
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-6)
        except (cp.SolverError, Exception):
            prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000,
                       eps_abs=1e-6, eps_rel=1e-6)
    except Exception as e:
        return {
            "status": "solver_error", "weights": {},
            "expected_return": 0.0, "expected_risk": 0.0, "sharpe_ratio": 0.0,
            "active_weights": {}, "tracking_error": 0.0, "turnover": 0.0,
            "constraint_satisfaction": {},
            "diagnostics": {"error": str(e)},
        }

    if w.value is None or prob.status in ("infeasible", "unbounded"):
        return {
            "status": "infeasible" if prob.status == "infeasible" else prob.status,
            "weights": {}, "expected_return": 0.0, "expected_risk": 0.0,
            "sharpe_ratio": 0.0, "active_weights": {}, "tracking_error": 0.0,
            "turnover": 0.0, "constraint_satisfaction": {},
            "diagnostics": {"status": prob.status},
        }

    weights_arr = np.array(w.value).flatten()
    weights_arr = np.maximum(weights_arr, 0.0)
    weights_arr = weights_arr / weights_arr.sum()

    weights_dict = {s: float(weights_arr[i]) for i, s in enumerate(symbols)}
    exp_ret = float(mu @ weights_arr)
    exp_risk = float(np.sqrt(weights_arr @ Sigma @ weights_arr))
    sharpe = (exp_ret - risk_free_rate) / exp_risk if exp_risk > 1e-10 else 0.0

    active_weights = {}
    tracking_error = 0.0
    if benchmark_weights:
        w_bench = np.array([benchmark_weights.get(s, 0.0) for s in symbols])
        active_arr = weights_arr - w_bench
        active_weights = {s: float(active_arr[i]) for i, s in enumerate(symbols)}
        te_var = active_arr @ Sigma @ active_arr
        tracking_error = float(np.sqrt(max(te_var, 0.0)))

    turnover = 0.0
    if current_portfolio:
        w_curr = np.array([current_portfolio.get(s, 0.0) for s in symbols])
        turnover = float(np.sum(np.abs(weights_arr - w_curr)) / 2)

    cs: dict[str, bool] = {}
    if constraints.get("max_weight"):
        cs["max_weight"] = all(v <= constraints["max_weight"] + 1e-6 for v in weights_dict.values())
    if constraints.get("long_only", True):
        cs["long_only"] = all(v >= -1e-6 for v in weights_dict.values())
    if constraints.get("min_names"):
        active_names = sum(1 for v in weights_dict.values() if v > 1e-4)
        cs["min_names"] = active_names >= constraints["min_names"]
    if constraints.get("max_names"):
        active_names = sum(1 for v in weights_dict.values() if v > 1e-4)
        cs["max_names"] = active_names <= constraints["max_names"]
    if constraints.get("sector_limits") and sector_mapping:
        for sector, limit in constraints["sector_limits"].items():
            sector_wt = sum(weights_dict.get(s, 0.0) for s in symbols
                            if sector_mapping.get(s) == sector)
            cs[f"sector_{sector}"] = sector_wt <= limit + 1e-6

    return {
        "status": "optimal", "weights": weights_dict,
        "expected_return": round(exp_ret, 6), "expected_risk": round(exp_risk, 6),
        "sharpe_ratio": round(sharpe, 4), "active_weights": active_weights,
        "tracking_error": round(tracking_error, 6), "turnover": round(turnover, 6),
        "constraint_satisfaction": cs,
        "diagnostics": {"solver_status": prob.status,
                        "objective_value": float(prob.value) if prob.value is not None else None},
    }


def black_litterman(
    market_cap_weights: dict[str, float],
    covariance_matrix: np.ndarray,
    symbols: list[str],
    views: list[dict],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> dict:
    """Black-Litterman model for combining market equilibrium with investor views."""
    N = len(symbols)
    sym_idx = {s: i for i, s in enumerate(symbols)}

    Sigma = covariance_matrix.copy()
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.min(eigvals) < 0:
        Sigma = Sigma + (abs(np.min(eigvals)) + 1e-8) * np.eye(N)

    w_mkt = np.array([market_cap_weights.get(s, 0.0) for s in symbols])
    w_mkt = w_mkt / max(w_mkt.sum(), 1e-10)
    pi = risk_aversion * Sigma @ w_mkt

    K = len(views)
    if K == 0:
        return {
            "status": "equilibrium",
            "posterior_returns": {s: float(pi[i]) for i, s in enumerate(symbols)},
            "posterior_weights": {s: float(w_mkt[i]) for i, s in enumerate(symbols)},
            "view_adjustments": [],
        }

    P = np.zeros((K, N))
    Q = np.zeros(K)
    omega_diag = np.zeros(K)

    for k, view in enumerate(views):
        view_syms = view.get("symbols", [])
        view_ret = view.get("return", 0.0)
        confidence = view.get("confidence", 0.5)

        if view.get("type") == "relative" and len(view_syms) == 2:
            P[k, sym_idx[view_syms[0]]] = 1.0
            P[k, sym_idx[view_syms[1]]] = -1.0
        else:
            for vs in view_syms:
                if vs in sym_idx:
                    P[k, sym_idx[vs]] = 1.0 / len(view_syms)

        Q[k] = view_ret
        omega_diag[k] = (1.0 - confidence) * (P[k] @ Sigma @ P[k]) * tau

    Omega = np.diag(omega_diag)
    tau_Sigma = tau * Sigma
    try:
        M = np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(Omega) @ P
        mu_bl = np.linalg.solve(M, np.linalg.inv(tau_Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    except np.linalg.LinAlgError:
        mu_bl = pi

    try:
        w_bl = np.linalg.solve(risk_aversion * Sigma, mu_bl)
        w_bl = np.maximum(w_bl, 0.0)
        if w_bl.sum() > 1e-10:
            w_bl = w_bl / w_bl.sum()
    except np.linalg.LinAlgError:
        w_bl = w_mkt

    posterior_returns = {s: float(mu_bl[i]) for i, s in enumerate(symbols)}
    posterior_weights = {s: float(w_bl[i]) for i, s in enumerate(symbols)}

    view_adjustments = []
    for k, view in enumerate(views):
        view_adjustments.append({
            "view": view,
            "prior_return": float(P[k] @ pi),
            "posterior_return": float(P[k] @ mu_bl),
        })

    return {
        "status": "optimal",
        "posterior_returns": posterior_returns,
        "posterior_weights": posterior_weights,
        "view_adjustments": view_adjustments,
    }


def _fallback_optimization(
    expected_returns: dict[str, float],
    covariance_matrix: np.ndarray,
    symbols: list[str],
    constraints: dict,
    benchmark_weights: dict[str, float] | None,
    objective: str,
) -> dict:
    """Simple heuristic fallback when cvxpy is unavailable."""
    N = len(symbols)
    mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
    max_w = constraints.get("max_weight", 1.0)

    if objective == "min_variance":
        diag_var = np.diag(covariance_matrix)
        inv_var = 1.0 / np.maximum(diag_var, 1e-10)
        raw_w = inv_var / inv_var.sum()
    elif objective == "max_return":
        raw_w = np.zeros(N)
        n_top = max(5, N // 5)
        top_idx = np.argsort(mu)[-n_top:]
        raw_w[top_idx] = 1.0 / n_top
    else:
        raw_w = np.ones(N) / N

    raw_w = np.minimum(raw_w, max_w)
    raw_w = raw_w / raw_w.sum()

    weights_dict = {s: float(raw_w[i]) for i, s in enumerate(symbols)}
    exp_ret = float(mu @ raw_w)
    exp_risk = float(np.sqrt(raw_w @ covariance_matrix @ raw_w))

    active_weights = {}
    tracking_error = 0.0
    if benchmark_weights:
        w_bench = np.array([benchmark_weights.get(s, 0.0) for s in symbols])
        active_arr = raw_w - w_bench
        active_weights = {s: float(active_arr[i]) for i, s in enumerate(symbols)}
        te_var = active_arr @ covariance_matrix @ active_arr
        tracking_error = float(np.sqrt(max(te_var, 0.0)))

    return {
        "status": "fallback", "weights": weights_dict,
        "expected_return": round(exp_ret, 6), "expected_risk": round(exp_risk, 6),
        "sharpe_ratio": round(exp_ret / exp_risk, 4) if exp_risk > 1e-10 else 0.0,
        "active_weights": active_weights, "tracking_error": round(tracking_error, 6),
        "turnover": 0.0, "constraint_satisfaction": {"fallback": True},
        "diagnostics": {"note": "cvxpy not available, using heuristic"},
    }
