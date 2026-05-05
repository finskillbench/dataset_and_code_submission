"""Regression tests for experiment 05 scorers (TDD harness)."""

from __future__ import annotations

import json

import pytest

from lib.scorers import score_task


def _task(verification: dict, expected_output: dict, sub_task: str = "test") -> dict:
    return {
        "task_id": "test_task",
        "sub_task": sub_task,
        "verification": verification,
        "expected_output": expected_output,
    }


class TestBlackLittermanWeights:
    def test_optimal_weights_used_not_posterior_weights_only(self):
        task = _task(
            {"method": "view_specification_and_weights", "weight_l2_threshold": 0.1},
            {
                "posterior_returns": {"A": 0.1, "B": 0.2},
                "optimal_weights": {"A": 0.5, "B": 0.5},
            },
        )
        pred = {
            "posterior_returns": {"A": 0.1, "B": 0.2},
            "optimal_weights": {"A": 0.5, "B": 0.5},
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.9
        assert r["details"].get("n_components") == 2

    def test_posterior_weights_backward_compat(self):
        task = _task(
            {"method": "view_specification_and_weights", "weight_l2_threshold": 0.1},
            {
                "posterior_returns": {"A": 0.1},
                "posterior_weights": {"A": 1.0},
            },
        )
        pred = {"posterior_returns": {"A": 0.1}, "posterior_weights": {"A": 1.0}}
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.9


class TestRebalancing:
    def test_new_weights_without_weights_alias(self):
        task = _task(
            {"method": "turnover_compliance_and_objective"},
            {
                "new_weights": {"X": 1.0},
                "weights": {"X": 1.0},
                "turnover": 0.1,
                "trade_list": {"X": 0.5},
            },
        )
        pred = {"new_weights": {"X": 1.0}, "turnover": 0.1, "trade_list": {"X": 0.5}}
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.9

    def test_score_task_accepts_dict_response(self):
        task = _task(
            {"method": "turnover_compliance_and_objective"},
            {"new_weights": {"X": 1.0}, "weights": {"X": 1.0}},
        )
        r = score_task(task, {"new_weights": {"X": 1.0}})
        assert r["valid_json"] is True
        assert r["score"] > 0.9


class TestConstrainedOptimizationGate:
    def test_gate_fails_wrong_constraint(self):
        task = _task(
            {"method": "constraint_satisfaction_and_objective", "weight_l2_threshold": 0.05},
            {
                "constraint_satisfaction": {"a": True, "b": False},
                "weights": {"A": 0.5, "B": 0.5},
            },
        )
        pred = {
            "constraint_satisfaction": {"a": True, "b": True},
            "weights": {"A": 0.5, "B": 0.5},
        }
        r = score_task(task, json.dumps(pred))
        assert r["score"] == 0.0

    def test_gate_passes_weights_scored(self):
        task = _task(
            {"method": "constraint_satisfaction_and_objective", "weight_l2_threshold": 0.05},
            {
                "constraint_satisfaction": {"a": True},
                "weights": {"A": 1.0},
            },
        )
        pred = {"constraint_satisfaction": {"a": True}, "weights": {"A": 1.0}}
        r = score_task(task, json.dumps(pred))
        assert r["score"] > 0.99


class TestParameterMatchRecursive:
    def test_nested_optimizer_call(self):
        task = _task(
            {"method": "parameter_match"},
            {
                "optimizer_call": {
                    "constraints": {"long_only": True, "max_weight_per_name": 0.1},
                    "objective": "max_return",
                }
            },
        )
        pred = {
            "optimizer_call": {
                "constraints": {"long_only": True, "max_weight_per_name": 0.1},
                "objective": "max_return",
            }
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] == 1.0


class TestRiskIdentificationNDCG:
    def test_multiset_and_order(self):
        task = _task(
            {"method": "ranked_list_recall", "top_k": 3},
            {
                "top_risk_exposures": [
                    {"type": "a", "magnitude": 1.0},
                    {"type": "b", "magnitude": 0.5},
                    {"type": "a", "magnitude": 0.3},
                ]
            },
            sub_task="risk_identification",
        )
        pred = {
            "top_risk_exposures": [
                {"type": "a", "magnitude": 1.0},
                {"type": "a", "magnitude": 0.3},
                {"type": "b", "magnitude": 0.5},
            ]
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.5


class TestStressAttribution:
    def test_pnl_plus_attribution(self):
        task = _task(
            {"method": "absolute_error", "tolerance_pct": 1.0},
            {
                "estimated_pnl_pct": -0.1,
                "attribution": {
                    "by_sector": [
                        {"sector": "S1", "contribution_pct": -0.05},
                        {"sector": "S2", "contribution_pct": -0.02},
                    ]
                },
            },
        )
        pred = {
            "estimated_pnl_pct": -0.1,
            "attribution": {
                "by_sector": [
                    {"sector": "S1", "contribution_pct": -0.05},
                    {"sector": "S2", "contribution_pct": -0.02},
                ]
            },
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.95


class TestRemediationTradeF1:
    def test_normalized_f1(self):
        task = _task(
            {"method": "constraint_satisfaction_plus_cost",
             "tolerance": {"turnover_excess_pct": 50}},
            {
                "post_trade_compliant": True,
                "total_turnover": 0.2,
                "trades": [
                    {"symbol": "aig", "action": "SELL"},
                    {"symbol": "c", "action": "buy"},
                ],
            },
        )
        pred = {
            "post_trade_compliant": True,
            "total_turnover": 0.2,
            "trades": [
                {"symbol": "AIG", "action": "sell"},
                {"symbol": "C", "action": "Buy"},
            ],
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["details"]["trade_f1"] == pytest.approx(1.0, abs=0.01)


class TestEarningsQualityNumeric:
    def test_numeric_fields_included(self):
        task = _task(
            {
                "method": "earnings_quality_composite",
                "piotroski_weight": 0.3,
                "beneish_flag_weight": 0.2,
                "flag_f1_weight": 0.2,
                "numeric_ratio_weight": 0.3,
                "numeric_tolerances": {
                    "beneish_m_score": 0.05,
                    "accruals_ratio": 0.2,
                    "income_quality_ratio": 0.05,
                },
            },
            {
                "piotroski_components": {"a": 1},
                "beneish_flag": False,
                "flags": [],
                "beneish_m_score": -2.0,
                "accruals_ratio": -0.02,
                "income_quality_ratio": 1.2,
            },
        )
        pred = {
            "piotroski_components": {"a": 1},
            "beneish_flag": False,
            "flags": [],
            "beneish_m_score": -2.0,
            "accruals_ratio": -0.02,
            "income_quality_ratio": 1.2,
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.95


class TestDriverDirection:
    def test_direction_case_insensitive(self):
        task = _task(
            {"method": "driver_f1_and_direction", "top_k": 2},
            {
                "revenue_delta": 100,
                "top_revenue_drivers": [
                    {"driver": "A", "direction": "increase"},
                ],
                "margin_drivers": [],
            },
        )
        pred = {
            "revenue_delta": 100,
            "top_revenue_drivers": [{"driver": "A", "direction": "Increase"}],
            "margin_drivers": [],
        }
        r = score_task(task, json.dumps(pred))
        assert r["valid_json"] is True
        assert r["score"] > 0.9
