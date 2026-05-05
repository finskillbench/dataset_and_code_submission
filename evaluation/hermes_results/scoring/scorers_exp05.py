"""Scorers: verification-method scorers for evaluation.

Internalized from inspect_tasks/scorers.py for standalone use.

**``valid_json`` semantics**

- ``True`` for outcomes that are **not** unparseable final text: normal scoring,
  unknown method, :func:`score_incomplete_eval` (no ``submit_answer`` — see
  ``max_turns_exhausted`` vs ``incomplete_submission``).
- ``False`` only when the agent supplied a final string that is **not valid JSON**
  for scoring (:func:`score_task` → ``invalid_submission``, formerly ``parse_failure``).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def score_incomplete_eval(agent_error: str | None = None) -> dict:
    """Score a run with no usable ``submit_answer`` (empty ``final_answer``).

    Distinguishes:

    - **max_turns_exhausted** — agent hit the turn limit without submitting (from
      ``fc_loop`` when ``error`` is ``max_turns_exhausted``).
    - **incomplete_submission** — all other cases (API abort mid-run, empty error,
      etc.): no valid final answer string to parse.

    Both are score 0.0 with ``valid_json: True`` (classified “no submission,” not
    “invalid JSON”). Unparseable *text* in ``final_answer`` is ``invalid_submission``.
    """
    detail = agent_error if agent_error else "empty answer"
    if agent_error == "max_turns_exhausted":
        method = "max_turns_exhausted"
    else:
        method = "incomplete_submission"
    return {
        "score": 0.0,
        "valid_json": True,
        "method": method,
        "details": {"error": detail},
    }


def score_no_answer(error: str | None = None) -> dict:
    """Deprecated alias for :func:`score_incomplete_eval`."""
    return score_incomplete_eval(error)


def weight_l2(predicted: dict, expected: dict) -> float:
    all_keys = set(predicted.keys()) | set(expected.keys())
    diff_sq = sum((predicted.get(k, 0.0) - expected.get(k, 0.0)) ** 2 for k in all_keys)
    return float(np.sqrt(diff_sq))


def _coerce_response_to_str(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, dict):
        return json.dumps(response)
    if isinstance(response, str):
        return response
    return str(response)


def score_task(task: dict, response: Any) -> dict:
    text = _coerce_response_to_str(response)
    if text is None:
        return score_incomplete_eval("empty answer")

    parsed = parse_json_response(text)
    if parsed is None:
        return {
            "score": 0.0,
            "valid_json": False,
            "method": "invalid_submission",
            "details": {"error": "Could not parse JSON from response"},
        }

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
        return {
            "score": 0.0,
            "valid_json": True,
            "method": method,
            "details": {"error": f"Unknown scoring method: {method}"},
        }
    try:
        return scorer_fn(parsed, expected, task.get("verification", {}))
    except Exception as e:
        return {
            "score": 0.0,
            "valid_json": True,
            "method": method,
            "details": {"error": f"Scoring error: {e}"},
        }


def _score_l2_distance(parsed, expected, verification):
    pred_weights = parsed.get("weights", {})
    exp_weights = expected.get("weights", {})
    if not pred_weights or not exp_weights:
        return {
            "score": 0.0,
            "valid_json": True,
            "method": "l2_distance_and_objective",
            "details": {"error": "Missing weights in response"},
        }
    l2 = weight_l2(pred_weights, exp_weights)
    threshold = verification.get("weight_l2_threshold", 0.05)
    score = max(0.0, min(1.0, 1.0 - l2 / (4 * threshold)))
    return {
        "score": score,
        "valid_json": True,
        "method": "l2_distance_and_objective",
        "details": {"l2_distance": l2, "threshold": threshold},
    }


def _score_constraint_satisfaction(parsed, expected, verification):
    pred_sat = parsed.get("constraint_satisfaction", {})
    exp_sat = expected.get("constraint_satisfaction", {})
    if exp_sat:
        if not pred_sat:
            return {
                "score": 0.0,
                "valid_json": True,
                "method": "constraint_satisfaction_and_objective",
                "details": {"error": "Missing constraint_satisfaction", "gate_passed": False},
            }
        for k, v in exp_sat.items():
            if k not in pred_sat or pred_sat[k] != v:
                return {
                    "score": 0.0,
                    "valid_json": True,
                    "method": "constraint_satisfaction_and_objective",
                    "details": {"gate_passed": False, "failed_key": k},
                }
        pred_w = parsed.get("weights", {})
        exp_w = expected.get("weights", {})
        if not pred_w or not exp_w:
            return {
                "score": 0.0,
                "valid_json": True,
                "method": "constraint_satisfaction_and_objective",
                "details": {"error": "Gate passed but missing weights", "gate_passed": True},
            }
        l2 = weight_l2(pred_w, exp_w)
        threshold = verification.get("weight_l2_threshold", 0.05)
        score = max(0.0, min(1.0, 1.0 - l2 / (4 * threshold)))
        return {
            "score": score,
            "valid_json": True,
            "method": "constraint_satisfaction_and_objective",
            "details": {"l2_distance": l2, "threshold": threshold, "gate_passed": True},
        }
    pred_w = parsed.get("weights", {})
    exp_w = expected.get("weights", {})
    if pred_w and exp_w:
        l2 = weight_l2(pred_w, exp_w)
        score = max(0.0, min(1.0, 1.0 - l2 / 0.2))
        return {
            "score": score,
            "valid_json": True,
            "method": "constraint_satisfaction_and_objective",
            "details": {"l2_distance": l2, "note": "No expected constraint_satisfaction; L2 fallback"},
        }
    return {
        "score": 0.0,
        "valid_json": True,
        "method": "constraint_satisfaction_and_objective",
        "details": {"error": "No weights or constraint_satisfaction in response"},
    }


def _weight_vector_for_rebal(parsed, expected):
    pred = parsed.get("new_weights") or parsed.get("weights") or {}
    exp = expected.get("new_weights") or expected.get("weights") or {}
    return pred, exp


def _turnover_ramp(pred_turn, exp_turn):
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


def _score_turnover_compliance(parsed, expected, verification):
    pred_w, exp_w = _weight_vector_for_rebal(parsed, expected)
    if not pred_w or not exp_w:
        return {
            "score": 0.0,
            "valid_json": True,
            "method": "turnover_compliance_and_objective",
            "details": {"error": "Missing new_weights/weights"},
        }
    l2 = weight_l2(pred_w, exp_w)
    w_score = max(0.0, min(1.0, 1.0 - l2 / 0.2))
    pred_turn = parsed.get("turnover")
    exp_turn = expected.get("turnover")
    t_score = _turnover_ramp(pred_turn, exp_turn)
    pred_tl = parsed.get("trade_list") or {}
    exp_tl = expected.get("trade_list") or {}
    tr_score = _trade_dict_consistency(pred_tl if isinstance(pred_tl, dict) else {}, exp_tl if isinstance(exp_tl, dict) else {})
    w_w = float(verification.get("weight_l2_component_weight", 0.5))
    t_w = float(verification.get("turnover_component_weight", 0.25))
    tr_w = float(verification.get("trade_list_component_weight", 0.25))
    tot = w_w + t_w + tr_w
    score = (w_w * w_score + t_w * t_score + tr_w * tr_score) / max(tot, 1e-9)
    return {
        "score": score,
        "valid_json": True,
        "method": "turnover_compliance_and_objective",
        "details": {
            "l2_distance": l2,
            "weight_score": round(w_score, 4),
            "turnover_score": round(t_score, 4),
            "trade_consistency_score": round(tr_score, 4),
        },
    }


def _recursive_parameter_leaf_scores(pred: Any, exp: Any, rel_tol: float) -> tuple[int, int]:
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
    if exp is None or pred is None:
        return 0, 1
    return 0, 1


def _score_parameter_match(parsed, expected, verification):
    rel_tol = verification.get("relative_tolerance", 0.2)
    matches, total = _recursive_parameter_leaf_scores(parsed, expected, rel_tol)
    score = matches / total if total > 0 else 0.0
    return {
        "score": score,
        "valid_json": True,
        "method": "parameter_match",
        "details": {"matches": matches, "total": total},
    }


def _score_infeasibility(parsed, expected, verification):
    pred_feasible = parsed.get("feasible", parsed.get("status") != "infeasible")
    exp_feasible = expected.get("feasible", expected.get("status") != "infeasible")
    score = 1.0 if pred_feasible == exp_feasible else 0.0
    return {
        "score": score,
        "valid_json": True,
        "method": "infeasibility_detection",
        "details": {"predicted_feasible": pred_feasible, "expected_feasible": exp_feasible},
    }


def _score_view_specification(parsed, expected, verification):
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
    return {
        "score": score,
        "valid_json": True,
        "method": "view_specification_and_weights",
        "details": {"n_components": len(score_components)},
    }


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


def _score_exact_match(parsed, expected, verification):
    pred_compliant = parsed.get("overall_compliant")
    exp_compliant = expected.get("overall_compliant")
    compliant_match = pred_compliant == exp_compliant
    pred_constraints = parsed.get("constraints", [])
    exp_constraints = expected.get("constraints", [])
    if exp_constraints:
        exp_by_type = {_normalize_constraint_type(c.get("type", "")): c for c in exp_constraints}
        pred_by_type = {_normalize_constraint_type(c.get("type", "")): c for c in pred_constraints}
        matches = sum(
            1
            for ntype, exp_c in exp_by_type.items()
            if _normalize_status(pred_by_type.get(ntype, {}).get("status", ""))
            == _normalize_status(exp_c.get("status", ""))
        )
        constraint_acc = matches / max(len(exp_by_type), 1)
    else:
        constraint_acc = 0.0
    score = 0.4 * (1.0 if compliant_match else 0.0) + 0.6 * constraint_acc
    return {
        "score": score,
        "valid_json": True,
        "method": "exact_match",
        "details": {
            "compliant_match": compliant_match,
            "constraint_accuracy": constraint_acc,
            "predicted_compliant": pred_compliant,
            "expected_compliant": exp_compliant,
        },
    }


def _score_absolute_error(parsed, expected, verification):
    pred_pnl = parsed.get("estimated_pnl_pct")
    exp_pnl = expected.get("estimated_pnl_pct")
    if pred_pnl is None or exp_pnl is None:
        return {
            "score": 0.0,
            "valid_json": True,
            "method": "absolute_error",
            "details": {"error": "Missing estimated_pnl_pct"},
        }
    tolerance = verification.get("tolerance_pct", 1.0) / 100.0
    abs_err = abs(float(pred_pnl) - float(exp_pnl))
    pnl_score = max(0.0, min(1.0, 1.0 - (abs_err - tolerance * 0.1) / (3 * tolerance)))

    exp_attr = expected.get("attribution") or {}
    pred_attr = parsed.get("attribution") or {}
    if exp_attr.get("by_sector"):
        attr_score = _attribution_sector_mae_score(pred_attr, exp_attr, verification)
        pnl_w = float(verification.get("pnl_score_weight", 0.7))
        attr_w = float(verification.get("attribution_score_weight", 0.3))
        s = pnl_w + attr_w
        score = (pnl_w * pnl_score + attr_w * attr_score) / max(s, 1e-9)
    else:
        attr_score = 1.0
        score = pnl_score
    return {
        "score": score,
        "valid_json": True,
        "method": "absolute_error",
        "details": {
            "abs_error": abs_err,
            "tolerance": tolerance,
            "pnl_score": round(pnl_score, 4),
            "attribution_score": round(attr_score, 4),
        },
    }


def _attribution_sector_mae_score(pred_attr: dict, exp_attr: dict, verification: dict) -> float:
    exp_list = exp_attr.get("by_sector") or []
    pred_list = pred_attr.get("by_sector") or []
    exp_map = {x.get("sector"): float(x.get("contribution_pct", 0)) for x in exp_list if x.get("sector")}
    pred_map = {x.get("sector"): float(x.get("contribution_pct", 0)) for x in pred_list if x.get("sector")}
    if not exp_map:
        return 1.0
    tol = verification.get("attribution_tolerance_pct", 0.05)
    scores = []
    for sec, ev in exp_map.items():
        pv = pred_map.get(sec)
        if pv is None:
            scores.append(0.0)
            continue
        rel_err = abs(pv - ev) / max(abs(ev), 1e-8)
        scores.append(max(0.0, 1.0 - min(1.0, rel_err / max(tol, 1e-9))))
    return float(np.mean(scores)) if scores else 0.0


def _multiset_match_count(exp_list: list, pred_list: list, key: str, top_k: int) -> int:
    exp_bag = Counter(r.get(key, "") for r in exp_list[:top_k])
    pred_bag = Counter(r.get(key, "") for r in pred_list[:top_k])
    return sum(min(exp_bag[t], pred_bag[t]) for t in set(exp_bag) | set(pred_bag))


def _ndcg_at_k(pred_list: list, exp_list: list, top_k: int) -> float:
    rel_by_type: dict[str, float] = {}
    for r in exp_list[:top_k]:
        t = r.get("type", "")
        mag = float(r.get("magnitude", 0) or 0)
        rel_by_type[t] = max(rel_by_type.get(t, 0.0), mag)

    def dcg(ranked: list) -> float:
        s = 0.0
        for i, r in enumerate(ranked[:top_k]):
            t = r.get("type", "")
            rel = rel_by_type.get(t, 0.0)
            if rel > 0:
                s += rel / math.log2(i + 2)
        return s

    ideal_rels = sorted(rel_by_type.values(), reverse=True)

    def ideal_dcg() -> float:
        s = 0.0
        for i, rel in enumerate(ideal_rels[:top_k]):
            if rel > 0:
                s += rel / math.log2(i + 2)
        return s

    idcg = ideal_dcg()
    if idcg <= 0:
        return 1.0 if not pred_list else 0.0
    return dcg(pred_list) / idcg


def _score_ranked_recall(parsed, expected, verification):
    pred_risks = parsed.get("top_risk_exposures", parsed.get("risks", []))
    exp_risks = expected.get("top_risk_exposures", [])
    if not pred_risks or not exp_risks:
        return {
            "score": 0.0,
            "valid_json": True,
            "method": "ranked_list_recall",
            "details": {"error": "Missing risk list"},
        }
    top_k = verification.get("top_k", 5)
    exp_total = max(len(exp_risks[:top_k]), 1)
    pred_total = max(len(pred_risks[:top_k]), 1)
    inter = _multiset_match_count(exp_risks, pred_risks, "type", top_k)
    recall = inter / exp_total
    precision = inter / pred_total
    ndcg = _ndcg_at_k(pred_risks, exp_risks, top_k)
    score = 0.4 * recall + 0.3 * precision + 0.3 * ndcg
    return {
        "score": score,
        "valid_json": True,
        "method": "ranked_list_recall",
        "details": {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "ndcg_at_k": ndcg,
        },
    }


def _trade_pair_multiset_f1(pred_trades: list, exp_trades: list) -> float:
    def norm_pair(t: dict) -> tuple[str, str]:
        return (str(t.get("symbol", "")).upper().strip(), str(t.get("action", "")).lower().strip())

    pred_bag = Counter(norm_pair(t) for t in pred_trades)
    exp_bag = Counter(norm_pair(t) for t in exp_trades)
    tp = sum(min(exp_bag[k], pred_bag[k]) for k in set(exp_bag) | set(pred_bag))
    pred_sum = sum(pred_bag.values())
    exp_sum = sum(exp_bag.values())
    prec = tp / max(pred_sum, 1)
    rec = tp / max(exp_sum, 1)
    if prec + rec <= 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _score_constraint_satisfaction_plus_cost(parsed, expected, verification):
    pred_compliant = parsed.get("post_trade_compliant")
    exp_compliant = expected.get("post_trade_compliant", True)
    compliant_match = pred_compliant == exp_compliant
    pred_turnover = parsed.get("total_turnover", 0)
    exp_turnover = expected.get("total_turnover", 0)
    turnover_tolerance = verification.get("tolerance", {}).get("turnover_excess_pct", 50) / 100.0
    turnover_ok = abs(pred_turnover - exp_turnover) <= turnover_tolerance * max(exp_turnover, 0.01)
    pred_trades = parsed.get("trades", [])
    exp_trades = expected.get("trades", [])
    if exp_trades:
        trade_f1 = _trade_pair_multiset_f1(pred_trades, exp_trades)
    else:
        trade_f1 = 0.0
    score = 0.3 * (1.0 if compliant_match else 0.0) + 0.3 * (1.0 if turnover_ok else 0.0) + 0.4 * trade_f1
    return {
        "score": score,
        "valid_json": True,
        "method": "constraint_satisfaction_plus_cost",
        "details": {
            "compliant_match": compliant_match,
            "turnover_ok": turnover_ok,
            "trade_f1": trade_f1,
        },
    }


def _score_metric_absolute_error(parsed, expected, verification):
    tolerances = verification.get("metric_tolerances", {})
    if not expected:
        return {
            "score": 0.0,
            "valid_json": True,
            "method": "metric_absolute_error",
            "details": {"error": "No expected output"},
        }
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
    return {
        "score": score,
        "valid_json": True,
        "method": "metric_absolute_error",
        "details": {"matches": matches, "total": total},
    }


def _norm_dir(d: str) -> str:
    return str(d or "").strip().lower()


def _numeric_tol_score(pred_val, exp_val, rel_tol: float) -> float:
    if pred_val is None or exp_val is None:
        return 0.0
    rel_err = abs(float(pred_val) - float(exp_val)) / max(abs(float(exp_val)), 1e-9)
    return max(0.0, 1.0 - min(1.0, rel_err / max(rel_tol, 1e-9)))


def _score_driver_f1_and_direction(parsed, expected, verification):
    top_k = verification.get("top_k", 5)
    scores = []
    pred_drivers = parsed.get("top_revenue_drivers", [])
    exp_drivers = expected.get("top_revenue_drivers", [])
    if exp_drivers:
        exp_names = [d.get("driver", "").lower().strip() for d in exp_drivers[:top_k]]
        pred_names = [d.get("driver", "").lower().strip() for d in pred_drivers[:top_k]]
        exp_set, pred_set = set(exp_names), set(pred_names)
        recall = len(exp_set & pred_set) / max(len(exp_set), 1)
        precision = len(exp_set & pred_set) / max(len(pred_set), 1)
        scores.append(0.5 * recall + 0.5 * precision)
        exp_dir_map = {d.get("driver", "").lower().strip(): d.get("direction", "") for d in exp_drivers}
        exp_row_map = {d.get("driver", "").lower().strip(): d for d in exp_drivers}
        dir_matches = dir_total = 0
        num_rev = []
        for d in pred_drivers[:top_k]:
            name = d.get("driver", "").lower().strip()
            if name in exp_dir_map:
                dir_total += 1
                if _norm_dir(d.get("direction", "")) == _norm_dir(exp_dir_map[name]):
                    dir_matches += 1
                er = exp_row_map.get(name, {})
                m_tol = float(verification.get("revenue_magnitude_rel_tol", 0.15))
                c_tol = float(verification.get("revenue_contribution_rel_tol", 0.15))
                if "magnitude_usd" in er:
                    num_rev.append(
                        _numeric_tol_score(d.get("magnitude_usd"), er.get("magnitude_usd"), m_tol)
                    )
                if "contribution_pct" in er:
                    num_rev.append(
                        _numeric_tol_score(d.get("contribution_pct"), er.get("contribution_pct"), c_tol)
                    )
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
        exp_md = {d.get("driver", "").lower().strip(): d.get("direction", "") for d in exp_margins}
        exp_mrow = {d.get("driver", "").lower().strip(): d for d in exp_margins}
        m_dm = m_dt = 0
        num_mar = []
        for d in pred_margins:
            name = d.get("driver", "").lower().strip()
            if name in exp_md:
                m_dt += 1
                if _norm_dir(d.get("direction", "")) == _norm_dir(exp_md[name]):
                    m_dm += 1
                er = exp_mrow.get(name, {})
                dp_tol = float(verification.get("margin_delta_pp_tol", 0.15))
                if er.get("delta_pp") is not None:
                    num_mar.append(_numeric_tol_score(d.get("delta_pp"), er.get("delta_pp"), dp_tol))
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
    return {
        "score": score,
        "valid_json": True,
        "method": "driver_f1_and_direction",
        "details": {"n_components": len(scores), "component_scores": [round(s, 4) for s in scores]},
    }


def _score_earnings_quality_composite(parsed, expected, verification):
    pw = float(verification.get("piotroski_weight", verification.get("component_weight", 0.4)))
    bw = float(verification.get("beneish_flag_weight", 0.3))
    fw = float(verification.get("flag_f1_weight", 0.3))
    nw = float(verification.get("numeric_ratio_weight", 0.0))
    numeric_tolerances = verification.get(
        "numeric_tolerances",
        {"beneish_m_score": 0.05, "accruals_ratio": 0.25, "income_quality_ratio": 0.05},
    )
    scores: list[tuple[str, float, float]] = []
    pred_components = parsed.get("piotroski_components", {})
    exp_components = expected.get("piotroski_components", {})
    if exp_components:
        matches = sum(1 for k in exp_components if pred_components.get(k) == exp_components[k])
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
    num_scores = []
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
    return {
        "score": round(score, 4),
        "valid_json": True,
        "method": "earnings_quality_composite",
        "details": details,
    }
