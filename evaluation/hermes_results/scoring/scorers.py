"""Scorers for FinSkillBench tasks (RM + PC + FA).

Base layer forked from experiments/zqbok_experiment02/inspect_tasks/scorers.py.
PC + FA scorers subsequently ported from experiments/zqbok_experiment05/lib/scorers.py
with the review-mandated P1 and P2 fixes from
experiments/zqbok_experiment05/implementation_plans/scoring_methodology_review.md
applied.

RM fixes (relative to zq_exp02):
  - parse_json_response: single-line fenced JSON now parses
  - _score_ranked_recall: NDCG@k + multiset recall/precision (P1 #3)
  - _score_absolute_error: tol=0 handled, sector attribution MAE composite (P2 #4)
  - _score_constraint_satisfaction_plus_cost: F1 + action/symbol normalization
  - _score_exact_match: overall_compliant credit gated on constraint coverage;
    contradictory duplicate pred types penalized (no silent last-wins)

PC scorers ported from zq_exp05 (review P1 + P2):
  - _score_constraint_satisfaction: hard gate on exp_sat, then L2 on weights
  - _score_parameter_match: recursive leaf-level comparison (P1 #1)
  - _score_view_specification: accepts optimal_weights or posterior_weights (P1 #2)
  - _score_turnover_compliance: composite w_L2 + turnover + trade_list (P2 #6)

FA scorers ported from zq_exp05 (review P2):
  - _score_metric_absolute_error: wrapper-unwrap for {"metrics": {...}} containers
  - _score_driver_f1_and_direction: numeric fidelity on magnitude / contribution / delta_pp
  - _score_earnings_quality_composite: numeric_ratio_weight default raised to 0.3
    so Beneish / accruals / income-quality ratios are actually scored (P2 #5).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np


_FENCE_OPEN = re.compile(r"^```[a-zA-Z0-9_+-]*\s*")


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_OPEN.sub("", text, count=1).rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def weight_l2(predicted: dict, expected: dict) -> float:
    all_keys = set(predicted.keys()) | set(expected.keys())
    diff_sq = sum((predicted.get(k, 0.0) - expected.get(k, 0.0)) ** 2 for k in all_keys)
    return float(np.sqrt(diff_sq))


def score_task(task: dict, response: str) -> dict:
    parsed = parse_json_response(response)
    if parsed is None:
        return {"score": 0.0, "valid_json": False, "method": "parse_failure",
                "details": {"error": "Could not parse JSON from response"}}

    method = task.get("verification", {}).get("method", "unknown")
    expected = task.get("expected_output", {})

    scorers = {
        "l2_distance_and_objective": _score_l2_distance,
        "constraint_satisfaction_and_objective": _score_constraint_satisfaction,
        "turnover_compliance_and_objective": _score_turnover_compliance,
        "parameter_match": _score_parameter_match,
        "infeasibility_detection": _score_infeasibility,
        "view_specification_and_weights": _score_view_specification,
        "exact_match": _score_exact_match,
        "absolute_error": _score_absolute_error,
        "ranked_list_recall": _score_ranked_recall,
        "constraint_satisfaction_plus_cost": _score_constraint_satisfaction_plus_cost,
        "metric_absolute_error": _score_metric_absolute_error,
        "driver_f1_and_direction": _score_driver_f1_and_direction,
        "earnings_quality_composite": _score_earnings_quality_composite,
    }

    scorer_fn = scorers.get(method)
    if scorer_fn is None:
        return {"score": 0.0, "valid_json": True, "method": method,
                "details": {"error": f"Unknown scoring method: {method}"}}

    try:
        return scorer_fn(parsed, expected, task.get("verification", {}))
    except Exception as e:
        return {"score": 0.0, "valid_json": True, "method": method,
                "details": {"error": f"Scoring error: {e}"}}


# ---------------------------------------------------------------------------
# Portfolio Construction scorers
# ---------------------------------------------------------------------------

def _score_l2_distance(parsed: dict, expected: dict, verification: dict) -> dict:
    pred_weights = parsed.get("weights", {})
    exp_weights = expected.get("weights", {})
    if not pred_weights or not exp_weights:
        return {"score": 0.0, "valid_json": True, "method": "l2_distance_and_objective",
                "details": {"error": "Missing weights in response"}}
    l2 = weight_l2(pred_weights, exp_weights)
    threshold = verification.get("weight_l2_threshold", 0.05)
    score = max(0.0, min(1.0, 1.0 - l2 / (4 * threshold)))
    return {"score": score, "valid_json": True, "method": "l2_distance_and_objective",
            "details": {"l2_distance": l2, "threshold": threshold}}


def _score_constraint_satisfaction(parsed: dict, expected: dict, verification: dict) -> dict:
    """Constrained optimization (review §1.2).

    Hard gate on expected.constraint_satisfaction: any missing / mismatched
    key ⇒ score 0. When the gate passes, score the portfolio proximity using
    the same L2 normalization as l2_distance_and_objective.

    When expected.constraint_satisfaction is absent, fall back to L2 on
    weights (legacy behavior; layer_a episodes always carry the dict).
    """
    pred_sat = parsed.get("constraint_satisfaction", {})
    exp_sat = expected.get("constraint_satisfaction", {})
    if exp_sat:
        if not pred_sat:
            return {"score": 0.0, "valid_json": True,
                    "method": "constraint_satisfaction_and_objective",
                    "details": {"error": "Missing constraint_satisfaction",
                                "gate_passed": False}}
        for k, v in exp_sat.items():
            if k not in pred_sat or pred_sat[k] != v:
                return {"score": 0.0, "valid_json": True,
                        "method": "constraint_satisfaction_and_objective",
                        "details": {"gate_passed": False, "failed_key": k}}
        pred_w = parsed.get("weights", {})
        exp_w = expected.get("weights", {})
        if not pred_w or not exp_w:
            return {"score": 0.0, "valid_json": True,
                    "method": "constraint_satisfaction_and_objective",
                    "details": {"error": "Gate passed but missing weights",
                                "gate_passed": True}}
        l2 = weight_l2(pred_w, exp_w)
        threshold = verification.get("weight_l2_threshold", 0.05)
        score = max(0.0, min(1.0, 1.0 - l2 / (4 * threshold)))
        return {"score": score, "valid_json": True,
                "method": "constraint_satisfaction_and_objective",
                "details": {"l2_distance": l2, "threshold": threshold,
                            "gate_passed": True}}
    pred_w = parsed.get("weights", {})
    exp_w = expected.get("weights", {})
    if pred_w and exp_w:
        l2 = weight_l2(pred_w, exp_w)
        score = max(0.0, min(1.0, 1.0 - l2 / 0.2))
        return {"score": score, "valid_json": True,
                "method": "constraint_satisfaction_and_objective",
                "details": {"l2_distance": l2,
                            "note": "No expected constraint_satisfaction; L2 fallback"}}
    return {"score": 0.0, "valid_json": True,
            "method": "constraint_satisfaction_and_objective",
            "details": {"error": "No weights or constraint_satisfaction in response"}}


def _weight_vector_for_rebal(parsed: dict, expected: dict) -> tuple[dict, dict]:
    pred = parsed.get("new_weights") or parsed.get("weights") or {}
    exp = expected.get("new_weights") or expected.get("weights") or {}
    return pred, exp


def _turnover_ramp(pred_turn: Any, exp_turn: Any) -> float:
    if exp_turn is None:
        return 1.0
    if pred_turn is None:
        return 0.0
    rel = abs(float(pred_turn) - float(exp_turn)) / max(abs(float(exp_turn)), 1e-9)
    return max(0.0, 1.0 - min(1.0, rel))


def _trade_dict_consistency(pred_tl: dict, exp_tl: dict) -> float:
    if not exp_tl:
        return 1.0
    if not pred_tl:
        return 0.0
    acc = 0.0
    n = 0
    for k, ev in exp_tl.items():
        pv = pred_tl.get(k)
        n += 1
        if pv is None:
            continue
        rel = abs(float(pv) - float(ev)) / max(abs(float(ev)), 1e-9)
        acc += max(0.0, 1.0 - min(1.0, rel))
    return acc / max(n, 1)


def _score_turnover_compliance(parsed: dict, expected: dict, verification: dict) -> dict:
    """Rebalancing composite (review §1.4, P2 #6): weights L2 + turnover + trade list."""
    pred_w, exp_w = _weight_vector_for_rebal(parsed, expected)
    if not pred_w or not exp_w:
        return {"score": 0.0, "valid_json": True,
                "method": "turnover_compliance_and_objective",
                "details": {"error": "Missing new_weights/weights"}}
    l2 = weight_l2(pred_w, exp_w)
    w_score = max(0.0, min(1.0, 1.0 - l2 / 0.2))
    t_score = _turnover_ramp(parsed.get("turnover"), expected.get("turnover"))
    pred_tl = parsed.get("trade_list") or {}
    exp_tl = expected.get("trade_list") or {}
    tr_score = _trade_dict_consistency(
        pred_tl if isinstance(pred_tl, dict) else {},
        exp_tl if isinstance(exp_tl, dict) else {},
    )
    w_w = float(verification.get("weight_l2_component_weight", 0.5))
    t_w = float(verification.get("turnover_component_weight", 0.25))
    tr_w = float(verification.get("trade_list_component_weight", 0.25))
    tot = w_w + t_w + tr_w
    score = (w_w * w_score + t_w * t_score + tr_w * tr_score) / max(tot, 1e-9)
    return {"score": score, "valid_json": True,
            "method": "turnover_compliance_and_objective",
            "details": {"l2_distance": l2,
                        "weight_score": round(w_score, 4),
                        "turnover_score": round(t_score, 4),
                        "trade_consistency_score": round(tr_score, 4)}}


def _recursive_parameter_leaf_scores(
    pred: Any, exp: Any, rel_tol: float
) -> tuple[int, int]:
    if isinstance(exp, dict) and isinstance(pred, dict):
        keys = set(exp.keys()) | set(pred.keys())
        if not keys:
            return 1, 1
        matches = total = 0
        for k in keys:
            m, t = _recursive_parameter_leaf_scores(pred.get(k), exp.get(k), rel_tol)
            matches += m
            total += t
        return matches, total
    if isinstance(exp, (int, float)) and isinstance(pred, (int, float)):
        denom = max(abs(float(exp)), 1e-6)
        ok = abs(float(pred) - float(exp)) / denom < rel_tol
        return (1 if ok else 0), 1
    if exp == pred:
        return 1, 1
    return 0, 1


def _score_parameter_match(parsed: dict, expected: dict, verification: dict) -> dict:
    """Tool-use parameterization (review §1.3, P1 #1): recursive leaf-level match."""
    rel_tol = verification.get("relative_tolerance", 0.2)
    matches, total = _recursive_parameter_leaf_scores(parsed, expected, rel_tol)
    score = matches / total if total > 0 else 0.0
    return {"score": score, "valid_json": True, "method": "parameter_match",
            "details": {"matches": matches, "total": total}}


def _score_infeasibility(parsed: dict, expected: dict, verification: dict) -> dict:
    pred_feasible = parsed.get("feasible", parsed.get("status") != "infeasible")
    exp_feasible = expected.get("feasible", expected.get("status") != "infeasible")
    score = 1.0 if pred_feasible == exp_feasible else 0.0
    return {"score": score, "valid_json": True, "method": "infeasibility_detection",
            "details": {"predicted_feasible": pred_feasible,
                        "expected_feasible": exp_feasible}}


def _score_view_specification(parsed: dict, expected: dict, verification: dict) -> dict:
    """Black-Litterman (review §1.5, P1 #2): posterior_returns MAE + weights L2.

    Reads optimal_weights first, falling back to posterior_weights for
    backward compatibility.
    """
    score_components = []
    if "posterior_returns" in expected and "posterior_returns" in parsed:
        pred_r = parsed["posterior_returns"]
        exp_r = expected["posterior_returns"]
        if isinstance(pred_r, dict) and isinstance(exp_r, dict):
            all_keys = set(pred_r.keys()) | set(exp_r.keys())
            if all_keys:
                errors = [abs(pred_r.get(k, 0) - exp_r.get(k, 0)) for k in all_keys]
                mae = np.mean(errors)
                score_components.append(max(0.0, 1.0 - mae / 0.05))
    exp_w_key = None
    if "optimal_weights" in expected:
        exp_w_key = "optimal_weights"
    elif "posterior_weights" in expected:
        exp_w_key = "posterior_weights"
    pred_w_key = None
    if exp_w_key and exp_w_key in parsed:
        pred_w_key = exp_w_key
    elif "optimal_weights" in parsed:
        pred_w_key = "optimal_weights"
    elif "posterior_weights" in parsed:
        pred_w_key = "posterior_weights"
    if exp_w_key and pred_w_key:
        pred_w = parsed[pred_w_key]
        exp_w = expected[exp_w_key]
        if isinstance(pred_w, dict) and isinstance(exp_w, dict):
            l2 = weight_l2(pred_w, exp_w)
            threshold = verification.get("weight_l2_threshold", 0.10)
            score_components.append(max(0.0, min(1.0, 1.0 - l2 / (4 * threshold))))
    score = float(np.mean(score_components)) if score_components else 0.0
    return {"score": score, "valid_json": True, "method": "view_specification_and_weights",
            "details": {"n_components": len(score_components)}}


# ---------------------------------------------------------------------------
# Risk Management scorers (retained from jb fork; stricter than zq_exp05)
# ---------------------------------------------------------------------------

def _normalize_status(status: str) -> str:
    s = status.lower().strip()
    if s in ("pass", "true", "compliant", "ok", "satisfied"):
        return "PASS"
    if s in ("fail", "false", "violation", "violated", "breach", "breached", "non-compliant"):
        return "FAIL"
    return s.upper()


def _normalize_constraint_type(ctype: str) -> str:
    c = ctype.lower().replace("-", "_").replace(" ", "_")
    mappings = {
        "position_limit_max": "position_limit",
        "max_position": "position_limit",
        "max_single_position_weight": "position_limit",
        "max_weight": "position_limit",
        "max_position_weight": "position_limit",
        "sector_limit_max": "sector_limit",
        "max_sector": "sector_limit",
        "max_sector_concentration": "sector_limit",
        "sector_concentration": "sector_limit",
        "beta_range": "beta_range",
        "beta": "beta_range",
        "portfolio_beta": "beta_range",
        "long_only": "long_only",
        "no_short": "long_only",
        "turnover_max_monthly": "turnover",
        "turnover_max": "turnover",
        "max_turnover": "turnover",
        "turnover": "turnover",
    }
    return mappings.get(c, c)


def _score_exact_match(parsed: dict, expected: dict, verification: dict) -> dict:
    """Fix 6+8: require constraint coverage for the overall_compliant credit and
    penalize contradictory duplicate pred entries (dict last-wins was silent)."""
    pred_compliant = parsed.get("overall_compliant")
    exp_compliant = expected.get("overall_compliant")
    compliant_match = pred_compliant == exp_compliant

    pred_constraints = parsed.get("constraints", [])
    exp_constraints = expected.get("constraints", [])

    duplicate_pred_types = False
    if exp_constraints:
        exp_by_type: dict[str, dict] = {}
        for c in exp_constraints:
            ntype = _normalize_constraint_type(c.get("type", ""))
            exp_by_type[ntype] = c
        pred_by_type: dict[str, list[dict]] = {}
        for c in pred_constraints:
            ntype = _normalize_constraint_type(c.get("type", ""))
            pred_by_type.setdefault(ntype, []).append(c)
        matches = 0
        total = len(exp_by_type)
        for ntype, exp_c in exp_by_type.items():
            exp_status = _normalize_status(exp_c.get("status", ""))
            pred_list = pred_by_type.get(ntype, [])
            if not pred_list:
                continue
            pred_statuses = {_normalize_status(p.get("status", "")) for p in pred_list}
            if len(pred_statuses) > 1:
                duplicate_pred_types = True
                continue
            if next(iter(pred_statuses)) == exp_status:
                matches += 1
        constraint_acc = matches / total if total > 0 else 0.0
        has_coverage = bool(pred_constraints)
    else:
        constraint_acc = 0.0
        has_coverage = True

    compliant_credit = 1.0 if (compliant_match and has_coverage) else 0.0
    score = 0.4 * compliant_credit + 0.6 * constraint_acc
    return {"score": score, "valid_json": True, "method": "exact_match",
            "details": {"compliant_match": compliant_match,
                        "constraint_accuracy": constraint_acc,
                        "predicted_compliant": pred_compliant,
                        "expected_compliant": exp_compliant,
                        "has_constraint_coverage": has_coverage,
                        "duplicate_pred_types": duplicate_pred_types}}


def _triangular_tolerance(abs_err: float, tol: float) -> float:
    """Shared triangular-ramp scoring: full credit within 0.1·tol, zero past ~3.1·tol."""
    if tol <= 0:
        return 1.0 if abs_err == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - (abs_err - tol * 0.1) / (3 * tol)))


def _sector_attribution_mae(
    parsed_attr: Any, expected_attr: Any
) -> tuple[float | None, int]:
    """Return (mae, n_sectors) over the union of sectors in by_sector lists.

    Accepts both list-of-dicts and dict-keyed-by-sector forms. Missing keys
    count as 0.0 contribution (so hallucinated sectors are penalized).
    Returns (None, 0) when attribution cannot be evaluated.
    """
    def _coerce_sectors(attr: Any) -> dict[str, float]:
        if attr is None:
            return {}
        if isinstance(attr, dict):
            by_sector = attr.get("by_sector", attr)
        else:
            by_sector = attr
        out: dict[str, float] = {}
        if isinstance(by_sector, list):
            for item in by_sector:
                if not isinstance(item, dict):
                    continue
                sector = item.get("sector") or item.get("name")
                if sector is None:
                    continue
                val = item.get("contribution_pct", item.get("contribution", item.get("value")))
                try:
                    out[str(sector)] = float(val)
                except (TypeError, ValueError):
                    continue
        elif isinstance(by_sector, dict):
            for sector, val in by_sector.items():
                try:
                    out[str(sector)] = float(val)
                except (TypeError, ValueError):
                    continue
        return out

    exp_map = _coerce_sectors(expected_attr)
    pred_map = _coerce_sectors(parsed_attr)
    if not exp_map or not pred_map:
        return None, 0
    keys = set(exp_map) | set(pred_map)
    if not keys:
        return None, 0
    mae = float(np.mean([abs(pred_map.get(k, 0.0) - exp_map.get(k, 0.0)) for k in keys]))
    return mae, len(keys)


def _score_absolute_error(parsed: dict, expected: dict, verification: dict) -> dict:
    """Absolute error on estimated_pnl_pct, with optional sector-attribution MAE.

    Fix 7.2: when verification lists `sector_attribution_mae` in metrics and
    both sides contain `attribution.by_sector`, the composite score becomes
      0.7 · pnl_score + 0.3 · attribution_score
    Otherwise the score is the pnl_score alone (backwards-compatible with the
    ZQ scorer).
    """
    pred_pnl = parsed.get("estimated_pnl_pct")
    exp_pnl = expected.get("estimated_pnl_pct")
    if pred_pnl is None:
        return {"score": 0.0, "valid_json": True, "method": "absolute_error",
                "details": {"error": "Missing estimated_pnl_pct"}}

    tolerance = verification.get("tolerance_pct", 1.0) / 100.0
    pnl_err = abs(float(pred_pnl) - float(exp_pnl))
    pnl_score = _triangular_tolerance(pnl_err, tolerance)

    details: dict[str, Any] = {
        "pnl_abs_error": pnl_err,
        "pnl_tolerance": tolerance,
        "pnl_score": pnl_score,
        "predicted_pnl": pred_pnl,
        "expected_pnl": exp_pnl,
    }

    metrics = set(verification.get("metrics") or [])
    attribution_requested = "sector_attribution_mae" in metrics
    attr_tolerance = verification.get("attribution_tolerance_pct",
                                      verification.get("tolerance_pct", 1.0)) / 100.0

    attr_score: float | None = None
    if attribution_requested:
        mae, n_sectors = _sector_attribution_mae(
            parsed.get("attribution"), expected.get("attribution")
        )
        if mae is not None:
            attr_score = _triangular_tolerance(mae, attr_tolerance)
            details["attribution_mae"] = mae
            details["attribution_tolerance"] = attr_tolerance
            details["attribution_n_sectors"] = n_sectors
            details["attribution_score"] = attr_score
        else:
            details["attribution_mae"] = None
            details["attribution_note"] = "attribution.by_sector missing on pred or exp"

    if attr_score is not None:
        score = 0.7 * pnl_score + 0.3 * attr_score
    else:
        score = pnl_score

    return {"score": score, "valid_json": True, "method": "absolute_error", "details": details}


def _ndcg_at_k(
    pred_items: list[dict], exp_items: list[dict], k: int
) -> tuple[float, float, float]:
    """True NDCG@k with graded relevance by expected magnitude.

    Matching rule: items are keyed by `type`. Duplicates are allowed — each
    predicted item is matched greedily to the highest-magnitude unmatched
    expected item of the same type. A predicted item whose type does not
    appear in the top-k expected set contributes 0 relevance.

    Returns (ndcg, dcg, idcg). NDCG is 0.0 when IDCG is 0 (all expected
    magnitudes zero or no expected items).
    """
    exp_sorted = sorted(
        exp_items[:k],
        key=lambda r: -float(r.get("magnitude") or 0.0),
    )
    exp_mags = [float(r.get("magnitude") or 0.0) for r in exp_sorted]
    idcg = sum(m / math.log2(i + 2) for i, m in enumerate(exp_mags))
    if idcg <= 0.0:
        return 0.0, 0.0, 0.0

    available: dict[str, list[float]] = {}
    for r in exp_sorted:
        t = str(r.get("type", ""))
        available.setdefault(t, []).append(float(r.get("magnitude") or 0.0))

    dcg = 0.0
    for i, r in enumerate(pred_items[:k]):
        t = str(r.get("type", ""))
        pool = available.get(t)
        if pool:
            rel = pool.pop(0)
            dcg += rel / math.log2(i + 2)

    return dcg / idcg, dcg, idcg


def _multiset_overlap(pred_types: list[str], exp_types: list[str]) -> int:
    """Greedy multiset intersection size (duplicates counted with multiplicity)."""
    remaining = list(exp_types)
    matches = 0
    for t in pred_types:
        if t in remaining:
            remaining.remove(t)
            matches += 1
    return matches


def _score_ranked_recall(parsed: dict, expected: dict, verification: dict) -> dict:
    """Composite: 0.4·recall@k + 0.3·precision@k + 0.3·NDCG@k.

    Fix 7.1 + bug 2: NDCG@k with graded relevance + multiset recall/precision
    (the previous impl used set() which collapsed duplicate expected types).
    """
    pred_risks = parsed.get("top_risk_exposures", parsed.get("risks", []))
    exp_risks = expected.get("top_risk_exposures", [])
    if not pred_risks or not exp_risks:
        return {"score": 0.0, "valid_json": True, "method": "ranked_list_recall",
                "details": {"error": "Missing risk list"}}

    top_k = verification.get("top_k", 5)
    exp_types = [r.get("type", "") for r in exp_risks[:top_k]]
    pred_types = [r.get("type", "") for r in pred_risks[:top_k]]
    if not exp_types:
        return {"score": 0.0, "valid_json": True, "method": "ranked_list_recall",
                "details": {"error": "No types in expected output"}}

    overlap = _multiset_overlap(pred_types, exp_types)
    recall = overlap / len(exp_types)
    precision = overlap / max(len(pred_types), 1)
    ndcg, dcg, idcg = _ndcg_at_k(pred_risks, exp_risks, top_k)

    score = 0.4 * recall + 0.3 * precision + 0.3 * ndcg
    return {"score": score, "valid_json": True, "method": "ranked_list_recall",
            "details": {
                "recall_at_k": recall,
                "precision_at_k": precision,
                "ndcg_at_k": ndcg,
                "dcg": dcg,
                "idcg": idcg,
                "top_k": top_k,
            }}


def _score_constraint_satisfaction_plus_cost(parsed: dict, expected: dict, verification: dict) -> dict:
    """Fix 4+5: F1 on (symbol, action) pairs with case/whitespace normalization
    (prior impl was recall-only and case-sensitive)."""
    pred_trades = parsed.get("trades", [])
    exp_trades = expected.get("trades", [])
    pred_compliant = parsed.get("post_trade_compliant")
    exp_compliant = expected.get("post_trade_compliant", True)
    compliant_match = pred_compliant == exp_compliant
    pred_turnover = parsed.get("total_turnover", 0)
    exp_turnover = expected.get("total_turnover", 0)
    turnover_tolerance = verification.get("tolerance", {}).get("turnover_excess_pct", 50) / 100.0
    turnover_ok = abs(pred_turnover - exp_turnover) <= turnover_tolerance * max(exp_turnover, 0.01)

    def _norm(t: dict) -> tuple[str, str]:
        return (str(t.get("symbol", "")).strip().upper(),
                str(t.get("action", "")).strip().lower())

    precision = recall = direction_match = 0.0
    if exp_trades:
        exp_set = {_norm(t) for t in exp_trades}
        pred_set = {_norm(t) for t in pred_trades}
        tp = len(exp_set & pred_set)
        recall = tp / max(len(exp_set), 1)
        precision = tp / max(len(pred_set), 1)
        direction_match = (2 * precision * recall / (precision + recall)) \
            if (precision + recall) > 0 else 0.0

    score = 0.3 * (1.0 if compliant_match else 0.0) + \
            0.3 * (1.0 if turnover_ok else 0.0) + \
            0.4 * direction_match
    return {"score": score, "valid_json": True, "method": "constraint_satisfaction_plus_cost",
            "details": {"compliant_match": compliant_match, "turnover_ok": turnover_ok,
                        "direction_match": direction_match,
                        "direction_precision": precision, "direction_recall": recall}}


# ---------------------------------------------------------------------------
# Fundamental Analysis scorers (ported from zq_exp05 with P2 #5 fix)
# ---------------------------------------------------------------------------

def _score_metric_absolute_error(parsed: dict, expected: dict, verification: dict) -> dict:
    """FA normalization (review §3.1). Unwraps a single-key {'metrics': {...}}
    container when both sides use it, then scores per-metric relative error.
    """
    tolerances = verification.get("metric_tolerances", {})
    if not expected:
        return {"score": 0.0, "valid_json": True, "method": "metric_absolute_error",
                "details": {"error": "No expected output"}}
    if len(expected) == 1:
        wrapper_key = next(iter(expected))
        if isinstance(expected[wrapper_key], dict) and isinstance(parsed.get(wrapper_key), dict):
            expected = expected[wrapper_key]
            parsed = parsed[wrapper_key]
    matches = total = 0
    tol_pct = verification.get("tolerance_pct", 5.0) / 100.0
    for key, exp_val in expected.items():
        if key in ("source", "confidence", "tier", "xbrl_fmp_agreement"):
            continue
        pred_val = parsed.get(key)
        if pred_val is None or exp_val is None:
            total += 1
            continue
        total += 1
        if isinstance(exp_val, (int, float)) and isinstance(pred_val, (int, float)):
            tol = tolerances.get(key, tol_pct)
            rel_err = abs(pred_val - exp_val) / max(abs(exp_val), 1e-8)
            if rel_err <= tol:
                matches += 1
        elif pred_val == exp_val:
            matches += 1
    score = matches / total if total > 0 else 0.0
    return {"score": score, "valid_json": True, "method": "metric_absolute_error",
            "details": {"matches": matches, "total": total}}


def _norm_dir(d: Any) -> str:
    return str(d or "").strip().lower()


def _numeric_tol_score(pred_val: Any, exp_val: Any, rel_tol: float) -> float:
    if pred_val is None or exp_val is None:
        return 0.0
    rel_err = abs(float(pred_val) - float(exp_val)) / max(abs(float(exp_val)), 1e-9)
    return max(0.0, 1.0 - min(1.0, rel_err / max(rel_tol, 1e-9)))


def _score_driver_f1_and_direction(parsed: dict, expected: dict, verification: dict) -> dict:
    """FA driver decomposition (review §3.3, P3 #9 numeric expansion).

    Combines:
      - name recall/precision on top_revenue_drivers,
      - direction accuracy on matched revenue drivers,
      - magnitude_usd and contribution_pct tolerance scores on matched rows,
      - margin_drivers name/direction/delta_pp,
      - revenue_delta relative error.
    """
    top_k = verification.get("top_k", 5)
    scores: list[float] = []
    pred_drivers = parsed.get("top_revenue_drivers", [])
    exp_drivers = expected.get("top_revenue_drivers", [])
    if exp_drivers:
        exp_names = [d.get("driver", "").lower().strip() for d in exp_drivers[:top_k]]
        pred_names = [d.get("driver", "").lower().strip() for d in pred_drivers[:top_k]]
        exp_set, pred_set = set(exp_names), set(pred_names)
        recall = len(exp_set & pred_set) / max(len(exp_set), 1)
        precision = len(exp_set & pred_set) / max(len(pred_set), 1)
        scores.append(0.5 * recall + 0.5 * precision)

        exp_dir_map = {d.get("driver", "").lower().strip(): d.get("direction", "")
                       for d in exp_drivers}
        exp_row_map = {d.get("driver", "").lower().strip(): d for d in exp_drivers}
        dir_matches = dir_total = 0
        num_rev: list[float] = []
        m_tol = float(verification.get("revenue_magnitude_rel_tol", 0.15))
        c_tol = float(verification.get("revenue_contribution_rel_tol", 0.15))
        for d in pred_drivers[:top_k]:
            name = d.get("driver", "").lower().strip()
            if name in exp_dir_map:
                dir_total += 1
                if _norm_dir(d.get("direction", "")) == _norm_dir(exp_dir_map[name]):
                    dir_matches += 1
                er = exp_row_map.get(name, {})
                if "magnitude_usd" in er:
                    num_rev.append(_numeric_tol_score(
                        d.get("magnitude_usd"), er.get("magnitude_usd"), m_tol))
                if "contribution_pct" in er:
                    num_rev.append(_numeric_tol_score(
                        d.get("contribution_pct"), er.get("contribution_pct"), c_tol))
        if dir_total > 0:
            scores.append(dir_matches / dir_total)
        if num_rev and verification.get("include_driver_numeric_terms", True):
            scores.append(float(np.mean(num_rev)))
    else:
        scores.append(0.0)

    pred_margins = parsed.get("margin_drivers", [])
    exp_margins = expected.get("margin_drivers", [])
    if exp_margins:
        exp_mn = {d.get("driver", "").lower().strip() for d in exp_margins}
        pred_mn = {d.get("driver", "").lower().strip() for d in pred_margins}
        margin_recall = len(exp_mn & pred_mn) / max(len(exp_mn), 1)
        exp_md = {d.get("driver", "").lower().strip(): d.get("direction", "")
                  for d in exp_margins}
        exp_mrow = {d.get("driver", "").lower().strip(): d for d in exp_margins}
        m_dm = m_dt = 0
        num_mar: list[float] = []
        dp_tol = float(verification.get("margin_delta_pp_tol", 0.15))
        for d in pred_margins:
            name = d.get("driver", "").lower().strip()
            if name in exp_md:
                m_dt += 1
                if _norm_dir(d.get("direction", "")) == _norm_dir(exp_md[name]):
                    m_dm += 1
                er = exp_mrow.get(name, {})
                if er.get("delta_pp") is not None:
                    num_mar.append(_numeric_tol_score(
                        d.get("delta_pp"), er.get("delta_pp"), dp_tol))
        margin_dir_acc = m_dm / max(m_dt, 1)
        scores.append(0.5 * margin_recall + 0.5 * margin_dir_acc)
        if num_mar and verification.get("include_driver_numeric_terms", True):
            scores.append(float(np.mean(num_mar)))

    pred_delta = parsed.get("revenue_delta")
    exp_delta = expected.get("revenue_delta")
    if pred_delta is not None and exp_delta is not None and exp_delta != 0:
        rel_err = abs(float(pred_delta) - float(exp_delta)) / max(abs(float(exp_delta)), 1)
        scores.append(max(0.0, 1.0 - rel_err))

    score = float(np.mean(scores)) if scores else 0.0
    return {"score": score, "valid_json": True, "method": "driver_f1_and_direction",
            "details": {"n_components": len(scores),
                        "component_scores": [round(s, 4) for s in scores]}}


def _score_earnings_quality_composite(
    parsed: dict, expected: dict, verification: dict
) -> dict:
    """FA earnings quality (review §3.2, P2 #5).

    Composite over Piotroski components, Beneish flag, concern-flag F1, and
    numeric ratios (beneish_m_score, accruals_ratio, income_quality_ratio).

    Default weights follow the review's recommended shape (0.3 / 0.2 / 0.2 / 0.3),
    which is the key P2 fix: prior zq_exp05 default set numeric_ratio_weight to
    0.0, so Beneish M-score / accruals / income-quality ratios were never
    scored on episodes that didn't explicitly enable them. Episodes that set
    the weights explicitly override these defaults.
    """
    pw = float(verification.get(
        "piotroski_weight", verification.get("component_weight", 0.3)))
    bw = float(verification.get("beneish_flag_weight", 0.2))
    fw = float(verification.get("flag_f1_weight", 0.2))
    nw = float(verification.get("numeric_ratio_weight", 0.3))
    numeric_tolerances = verification.get(
        "numeric_tolerances",
        {"beneish_m_score": 0.05, "accruals_ratio": 0.25, "income_quality_ratio": 0.05},
    )
    scores: list[tuple[str, float, float]] = []

    pred_components = parsed.get("piotroski_components", {})
    exp_components = expected.get("piotroski_components", {})
    if exp_components:
        matches = sum(1 for k in exp_components
                      if pred_components.get(k) == exp_components[k])
        scores.append(("component_accuracy", matches / max(len(exp_components), 1), pw))
    else:
        pred_score = parsed.get("piotroski_score")
        exp_score = expected.get("piotroski_score")
        if pred_score is not None and exp_score is not None:
            diff = abs(int(pred_score) - int(exp_score))
            scores.append(("piotroski_score", max(0.0, 1.0 - diff / 9.0), pw))

    pred_flag = parsed.get("beneish_flag")
    exp_flag = expected.get("beneish_flag")
    if pred_flag is not None and exp_flag is not None:
        scores.append(("beneish_flag", 1.0 if pred_flag == exp_flag else 0.0, bw))

    pred_flags = set(parsed.get("flags", []))
    exp_flags = set(expected.get("flags", []))
    if exp_flags or pred_flags:
        tp = len(pred_flags & exp_flags)
        precision = tp / max(len(pred_flags), 1)
        recall = tp / max(len(exp_flags), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        scores.append(("flag_f1", f1, fw))
    else:
        scores.append(("flag_f1", 1.0, fw))

    numeric_fields = ["beneish_m_score", "accruals_ratio", "income_quality_ratio"]
    num_scores: list[float] = []
    for fld in numeric_fields:
        if fld in expected and expected.get(fld) is not None and parsed.get(fld) is not None:
            tol = float(numeric_tolerances.get(fld, 0.05))
            num_scores.append(_numeric_tol_score(parsed.get(fld), expected.get(fld), tol))
    if num_scores and nw > 0:
        scores.append(("numeric_ratio", float(np.mean(num_scores)), nw))

    if scores:
        total_weight = sum(w for _, _, w in scores)
        score = sum(s * w for _, s, w in scores) / max(total_weight, 1e-9)
    else:
        score = 0.0
    details = {name: round(s, 4) for name, s, _ in scores}
    return {"score": round(score, 4), "valid_json": True,
            "method": "earnings_quality_composite", "details": details}
