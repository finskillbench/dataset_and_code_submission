"""Final smoke: every scoring method used in exp05 executes without raising."""

from __future__ import annotations

import json

import pytest

from lib.scorers import score_task


def _minimal_task(method: str, expected: dict | None = None) -> dict:
    return {
        "task_id": f"smoke_{method}",
        "sub_task": "smoke",
        "verification": {"method": method},
        "expected_output": expected or {},
    }


@pytest.mark.parametrize(
    "method,parsed,expected",
    [
        (
            "l2_distance_and_objective",
            {"weights": {"A": 1.0}},
            {"weights": {"A": 1.0}},
        ),
        (
            "constraint_satisfaction_and_objective",
            {"constraint_satisfaction": {"x": True}, "weights": {"A": 1.0}},
            {"constraint_satisfaction": {"x": True}, "weights": {"A": 1.0}},
        ),
        (
            "turnover_compliance_and_objective",
            {"new_weights": {"A": 1.0}, "turnover": 0.0, "trade_list": {}},
            {"new_weights": {"A": 1.0}, "weights": {"A": 1.0}, "turnover": 0.0, "trade_list": {}},
        ),
        ("parameter_match", {"a": 1}, {"a": 1}),
        ("infeasibility_detection", {"feasible": True}, {"feasible": True}),
        (
            "view_specification_and_weights",
            {"posterior_returns": {"A": 0.1}, "optimal_weights": {"A": 1.0}},
            {"posterior_returns": {"A": 0.1}, "optimal_weights": {"A": 1.0}},
        ),
        (
            "exact_match",
            {"overall_compliant": True, "constraints": []},
            {"overall_compliant": True, "constraints": []},
        ),
        (
            "absolute_error",
            {"estimated_pnl_pct": 0.01},
            {"estimated_pnl_pct": 0.01},
        ),
        (
            "ranked_list_recall",
            {"top_risk_exposures": [{"type": "t", "magnitude": 1.0}]},
            {"top_risk_exposures": [{"type": "t", "magnitude": 1.0}]},
        ),
        (
            "constraint_satisfaction_plus_cost",
            {"post_trade_compliant": True, "total_turnover": 0.1, "trades": []},
            {"post_trade_compliant": True, "total_turnover": 0.1, "trades": []},
        ),
        ("metric_absolute_error", {"metrics": {"x": 1}}, {"metrics": {"x": 1}}),
        (
            "driver_f1_and_direction",
            {"revenue_delta": 1, "top_revenue_drivers": [], "margin_drivers": []},
            {"revenue_delta": 1, "top_revenue_drivers": [], "margin_drivers": []},
        ),
        (
            "earnings_quality_composite",
            {
                "piotroski_components": {"a": 1},
                "beneish_flag": False,
                "flags": [],
            },
            {
                "piotroski_components": {"a": 1},
                "beneish_flag": False,
                "flags": [],
            },
        ),
    ],
)
def test_scorer_executes(method: str, parsed: dict, expected: dict):
    task = _minimal_task(method, expected)
    r = score_task(task, json.dumps(parsed))
    assert "score" in r
    assert "valid_json" in r
    assert "method" in r
    assert r["method"] == method
