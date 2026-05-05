"""
Core risk computation library.

Provides deterministic risk computations used as ground truth oracle.
Internalized from scripts/risk_management/_risk_engine.py for standalone use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


def compute_portfolio_var(
    weights_vec: np.ndarray,
    returns_df: pd.DataFrame,
    confidence: float = 0.95,
    method: str = "historical",
) -> Tuple[float, float]:
    """Compute portfolio VaR and CVaR (Expected Shortfall)."""
    portfolio_returns = returns_df.values @ weights_vec

    if method == "historical":
        sorted_returns = np.sort(portfolio_returns)
        idx = int((1 - confidence) * len(sorted_returns))
        var = sorted_returns[idx]
        cvar = sorted_returns[:idx + 1].mean()
    elif method == "parametric":
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std(ddof=1)
        z = stats.norm.ppf(1 - confidence)
        var = mu + z * sigma
        cvar = mu - sigma * stats.norm.pdf(z) / (1 - confidence)
    elif method == "cornish_fisher":
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std(ddof=1)
        skew = stats.skew(portfolio_returns)
        kurt = stats.kurtosis(portfolio_returns, fisher=True)
        z = stats.norm.ppf(1 - confidence)
        z_cf = (z + (z**2 - 1) * skew / 6
                + (z**3 - 3*z) * kurt / 24
                - (2*z**3 - 5*z) * skew**2 / 36)
        var = mu + z_cf * sigma
        cvar = mu - sigma * stats.norm.pdf(z) / (1 - confidence)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return float(var), float(cvar)


def compute_portfolio_volatility(
    weights_vec: np.ndarray, cov_matrix: np.ndarray,
) -> float:
    """Annualized portfolio volatility from daily covariance matrix."""
    port_var = weights_vec @ cov_matrix @ weights_vec
    return float(np.sqrt(port_var) * np.sqrt(252))


def compute_factor_exposures(
    weights_vec: np.ndarray, factor_betas_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute portfolio-level factor betas as weighted average."""
    port_betas = weights_vec @ factor_betas_df.values
    return {col: float(v) for col, v in zip(factor_betas_df.columns, port_betas)}


def compute_tracking_error(
    weights_vec: np.ndarray, bench_weights_vec: np.ndarray, cov_matrix: np.ndarray,
) -> float:
    """Annualized tracking error vs. benchmark."""
    active = weights_vec - bench_weights_vec
    te_var = active @ cov_matrix @ active
    return float(np.sqrt(te_var * 252))


def compute_max_drawdown(returns_series: pd.Series) -> Dict[str, Any]:
    """Compute maximum drawdown statistics."""
    cum = (1 + returns_series).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    end_idx = drawdown.idxmin()
    max_dd = float(drawdown.min())
    dd_to_end = drawdown[:end_idx]
    start_idx = dd_to_end[dd_to_end == 0].index[-1] if len(dd_to_end[dd_to_end == 0]) > 0 else drawdown.index[0]
    duration_days = (end_idx - start_idx).days if hasattr(end_idx - start_idx, 'days') else 0
    return {"max_dd": max_dd, "start": str(start_idx), "end": str(end_idx), "duration_days": duration_days}


def compute_concentration_metrics(weights_vec: np.ndarray) -> Dict[str, float]:
    """Compute portfolio concentration metrics."""
    w = np.abs(weights_vec)
    hhi = float(np.sum(w ** 2))
    sorted_w = np.sort(w)[::-1]
    top5 = float(sorted_w[:min(5, len(sorted_w))].sum())
    top10 = float(sorted_w[:min(10, len(sorted_w))].sum())
    effective_n = float(1.0 / hhi) if hhi > 0 else float(len(w))
    return {"hhi": hhi, "top5_weight": top5, "top10_weight": top10, "effective_n": effective_n}


def compute_sector_exposures(
    weights_dict: Dict[str, float], sector_map: Dict[str, str],
) -> Dict[str, float]:
    """Aggregate weights by sector."""
    sector_weights: Dict[str, float] = {}
    for sym, w in weights_dict.items():
        sector = sector_map.get(sym, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w
    return sector_weights


def check_constraints(
    portfolio_weights: Dict[str, float], mandate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check portfolio against mandate constraints."""
    results = []
    constraints = mandate.get("constraints", [])

    for constraint in constraints:
        ctype = constraint["type"]
        result: Dict[str, Any] = {"type": ctype, "status": "PASS"}

        if ctype == "position_limit_max":
            limit = constraint["value"]
            worst_sym = max(portfolio_weights, key=lambda s: abs(portfolio_weights.get(s, 0)))
            worst_val = abs(portfolio_weights.get(worst_sym, 0))
            result.update({"worst_symbol": worst_sym, "value": round(worst_val, 6), "limit": limit})
            if worst_val > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(worst_val - limit, 6)

        elif ctype == "position_limit_min":
            limit = constraint["value"]
            non_zero = {s: w for s, w in portfolio_weights.items() if abs(w) > 1e-10}
            if non_zero:
                smallest_sym = min(non_zero, key=lambda s: abs(non_zero[s]))
                smallest_val = abs(non_zero[smallest_sym])
                result.update({"worst_symbol": smallest_sym, "value": round(smallest_val, 6), "limit": limit})
                if smallest_val < limit:
                    result["status"] = "FAIL"
                    result["violation_magnitude"] = round(limit - smallest_val, 6)

        elif ctype == "sector_limit_max":
            limit = constraint["value"]
            sector_map = mandate.get("_sector_map", {})
            if sector_map:
                sector_w = compute_sector_exposures(portfolio_weights, sector_map)
                worst_sector = max(sector_w, key=sector_w.get)
                worst_val = sector_w[worst_sector]
                result.update({"worst_sector": worst_sector, "value": round(worst_val, 6), "limit": limit})
                if worst_val > limit:
                    result["status"] = "FAIL"
                    result["violation_magnitude"] = round(worst_val - limit, 6)
            else:
                result.update({"status": "SKIP", "reason": "No sector map provided"})

        elif ctype == "beta_range":
            beta_min = constraint.get("min", 0.0)
            beta_max = constraint.get("max", 2.0)
            beta_val = mandate.get("_portfolio_beta", 1.0)
            result.update({"value": round(beta_val, 4), "min": beta_min, "max": beta_max})
            if beta_val < beta_min:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(beta_min - beta_val, 4)
            elif beta_val > beta_max:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(beta_val - beta_max, 4)

        elif ctype == "tracking_error_max":
            limit = constraint["value"]
            te_val = mandate.get("_tracking_error", 0.0)
            result.update({"value": round(te_val, 6), "limit": limit})
            if te_val > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(te_val - limit, 6)

        elif ctype == "cash_max":
            limit = constraint["value"]
            cash_weight = portfolio_weights.get("CASH", 0.0)
            result.update({"value": round(cash_weight, 6), "limit": limit})
            if cash_weight > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(cash_weight - limit, 6)

        elif ctype == "long_only":
            has_short = any(w < -1e-8 for w in portfolio_weights.values())
            result.update({"value": not has_short, "limit": True})
            if has_short:
                result["status"] = "FAIL"
                result["short_positions"] = [s for s, w in portfolio_weights.items() if w < -1e-8]

        elif ctype == "turnover_max_monthly":
            limit = constraint["value"]
            turnover = mandate.get("_monthly_turnover", 0.0)
            result.update({"value": round(turnover, 6), "limit": limit})
            if turnover > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(turnover - limit, 6)

        elif ctype == "gross_exposure_max":
            limit = constraint.get("value", 1.0)
            gross = sum(abs(w) for w in portfolio_weights.values())
            result.update({"value": round(gross, 6), "limit": limit})
            if gross > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(gross - limit, 6)

        elif ctype == "hhi_max":
            limit = constraint["value"]
            weights_arr = np.array(list(portfolio_weights.values()))
            hhi = float(np.sum(weights_arr ** 2))
            result.update({"value": round(hhi, 6), "limit": limit})
            if hhi > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(hhi - limit, 6)

        elif ctype == "factor_exposure_max":
            factor_name = constraint.get("factor", "mkt_rf")
            limit = constraint.get("value", 1.0)
            factor_val = mandate.get("_factor_exposures", {}).get(factor_name, 0.0)
            result.update({"factor": factor_name, "value": round(factor_val, 4), "limit": limit})
            if abs(factor_val) > limit:
                result["status"] = "FAIL"
                result["violation_magnitude"] = round(abs(factor_val) - limit, 4)

        results.append(result)
    return results


def compute_stress_pnl(
    weights_dict: Dict[str, float],
    scenario: Dict[str, Any],
    factor_betas: Dict[str, Dict[str, float]],
    sector_map: Dict[str, str],
    cov_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute estimated P&L under a stress scenario."""
    scenario_type = scenario.get("type", "hypothetical")

    rate_sensitivity = {
        "Real Estate": -8.0, "Utilities": -5.0, "Technology": -3.0,
        "Healthcare": -2.0, "Consumer Staples": -1.5, "Financial Services": +2.0,
        "Energy": +0.5, "Materials": +1.0, "Industrials": -1.0,
        "Consumer Discretionary": -2.5, "Communication Services": -2.0,
    }

    if scenario_type == "historical_replay":
        shocks = scenario.get("market_shocks", {})
        equity_dd = shocks.get("equity_drawdown_pct", -10.0) / 100.0
        rate_change = shocks.get("rate_change_bp", 0.0) / 10000.0
        credit_change = shocks.get("credit_spread_widening_bp", 0.0) / 10000.0

        total_pnl = 0.0
        sector_pnl: Dict[str, float] = {}
        factor_pnl: Dict[str, float] = {}

        for sym, weight in weights_dict.items():
            sector = sector_map.get(sym, "Unknown")
            betas = factor_betas.get(sym, {})
            sym_mkt_beta = betas.get("mkt_rf", 1.0)
            equity_pnl = weight * equity_dd * sym_mkt_beta
            sector_rate_mult = rate_sensitivity.get(sector, -1.0)
            rate_pnl = weight * rate_change * sector_rate_mult
            sym_cma_beta = abs(betas.get("cma", 0.0))
            credit_pnl = weight * credit_change * (1 + sym_cma_beta)
            sym_pnl = equity_pnl + rate_pnl + credit_pnl
            total_pnl += sym_pnl
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) + sym_pnl
            factor_pnl["market"] = factor_pnl.get("market", 0.0) + equity_pnl
            factor_pnl["interest_rate"] = factor_pnl.get("interest_rate", 0.0) + rate_pnl
            factor_pnl["credit"] = factor_pnl.get("credit", 0.0) + credit_pnl
    else:
        shocks = scenario.get("shocks", [])
        sector_sensitivities = scenario.get("sector_sensitivities", {
            "Real Estate": -0.15, "Utilities": -0.10, "Technology": -0.08,
            "Healthcare": -0.03, "Financial Services": +0.03, "Energy": 0.00,
            "Consumer Staples": -0.02, "Consumer Discretionary": -0.06,
            "Industrials": -0.04, "Materials": -0.03, "Communication Services": -0.05,
        })
        total_pnl = 0.0
        sector_pnl = {}
        factor_pnl = {}
        for sym, weight in weights_dict.items():
            sector = sector_map.get(sym, "Unknown")
            betas = factor_betas.get(sym, {})
            sym_pnl = 0.0
            for shock in shocks:
                factor = shock["factor"]
                if factor == "equity_market":
                    mag = shock.get("magnitude_pct", 0.0) / 100.0
                    contrib = weight * mag * betas.get("mkt_rf", 1.0)
                    sym_pnl += contrib
                    factor_pnl["market"] = factor_pnl.get("market", 0.0) + contrib
                elif factor == "interest_rate":
                    mag_bp = shock.get("magnitude_bp", 0.0)
                    sec_sens = sector_sensitivities.get(sector, -0.05)
                    contrib = weight * (mag_bp / 10000.0) * sec_sens * 10
                    sym_pnl += contrib
                    factor_pnl["interest_rate"] = factor_pnl.get("interest_rate", 0.0) + contrib
                elif factor == "credit_spread":
                    mag_bp = shock.get("magnitude_bp", 0.0)
                    cma_beta = abs(betas.get("cma", 0.0))
                    contrib = weight * (mag_bp / 10000.0) * (1 + cma_beta) * -1
                    sym_pnl += contrib
                    factor_pnl["credit"] = factor_pnl.get("credit", 0.0) + contrib
            total_pnl += sym_pnl
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) + sym_pnl

    attribution = {
        "by_sector": [{"sector": s, "contribution_pct": round(v, 6)}
                      for s, v in sorted(sector_pnl.items(), key=lambda x: x[1])],
        "by_factor": [{"factor": f, "contribution_pct": round(v, 6)}
                      for f, v in sorted(factor_pnl.items(), key=lambda x: x[1])],
    }
    return {"pnl_pct": round(float(total_pnl), 6), "attribution": attribution}


def compute_remediation(
    current_weights: Dict[str, float],
    mandate: Dict[str, Any],
    cov_matrix: np.ndarray,
    symbols: List[str],
    sector_map: Dict[str, str],
    factor_betas: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Find minimal-trade adjustment to restore mandate compliance."""
    n = len(symbols)
    w_current = np.array([current_weights.get(s, 0.0) for s in symbols])

    def objective(w_new):
        return np.sum((w_new - w_current) ** 2)

    constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    constraints_specs = mandate.get("constraints", [])

    for cspec in constraints_specs:
        ctype = cspec["type"]
        if ctype == "long_only" and cspec.get("value", True):
            bounds = [(0.0, 1.0)] * n
        elif ctype == "position_limit_max":
            limit = cspec["value"]
            for i in range(n):
                bounds[i] = (bounds[i][0], min(bounds[i][1], limit))
        elif ctype == "position_limit_min":
            limit = cspec.get("value", 0.001)
            for i in range(n):
                if w_current[i] > 1e-8:
                    bounds[i] = (max(bounds[i][0], limit), bounds[i][1])
        elif ctype == "sector_limit_max":
            limit = cspec["value"]
            for sector in set(sector_map.values()):
                idxs = [i for i, s in enumerate(symbols) if sector_map.get(s) == sector]
                if idxs:
                    constraints_list.append({
                        "type": "ineq",
                        "fun": lambda w, idxs=idxs, lim=limit: lim - np.sum(w[idxs]),
                    })
        elif ctype == "beta_range" and factor_betas:
            beta_min = cspec.get("min", 0.0)
            beta_max = cspec.get("max", 2.0)
            mkt_betas = np.array([factor_betas.get(s, {}).get("mkt_rf", 1.0) for s in symbols])
            constraints_list.append({"type": "ineq", "fun": lambda w, b=mkt_betas, lo=beta_min: w @ b - lo})
            constraints_list.append({"type": "ineq", "fun": lambda w, b=mkt_betas, hi=beta_max: hi - w @ b})
        elif ctype == "gross_exposure_max":
            limit = cspec.get("value", 1.0)
            constraints_list.append({"type": "ineq", "fun": lambda w, lim=limit: lim - np.sum(np.abs(w))})
        elif ctype == "hhi_max":
            limit = cspec["value"]
            constraints_list.append({"type": "ineq", "fun": lambda w, lim=limit: lim - np.sum(w ** 2)})

    w0 = np.ones(n) / n
    try:
        result = optimize.minimize(objective, w0, method="SLSQP", bounds=bounds,
                                   constraints=constraints_list, options={"maxiter": 2000, "ftol": 1e-12})
        w_optimal = result.x
    except Exception:
        w_optimal = np.clip(w_current, [b[0] for b in bounds], [b[1] for b in bounds])
        wsum = w_optimal.sum()
        w_optimal = w_optimal / wsum if wsum > 1e-10 else np.ones(n) / n

    w_optimal[np.abs(w_optimal) < 1e-8] = 0.0
    wsum = w_optimal.sum()
    if wsum > 1e-10:
        w_optimal = w_optimal / wsum

    new_weights = {s: float(w_optimal[i]) for i, s in enumerate(symbols)}
    check_mandate = dict(mandate)
    check_mandate["_sector_map"] = sector_map
    check = check_constraints(new_weights, check_mandate)
    actually_compliant = all(c.get("status") == "PASS" for c in check)

    trades = []
    total_turnover = 0.0
    for i, sym in enumerate(symbols):
        diff = w_optimal[i] - w_current[i]
        if abs(diff) > 1e-6:
            trades.append({
                "symbol": sym, "action": "buy" if diff > 0 else "sell",
                "current_weight": round(float(w_current[i]), 6),
                "target_weight": round(float(w_optimal[i]), 6),
                "trade_pct": round(float(diff), 6),
            })
            total_turnover += abs(diff)
    total_turnover /= 2.0

    return {
        "trades": trades, "turnover": round(float(total_turnover), 6),
        "compliant": actually_compliant, "num_trades": len(trades),
    }
