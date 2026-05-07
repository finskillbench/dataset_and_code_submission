"""Regression probes for the RM scorers.

Originally a bug-demonstrator for the Exp02 scorer. After the fixes landed,
each probe now asserts the CORRECTED behavior — the summary still describes
the old buggy behavior for context, but `is_bug` flips when the fix is present.

Can target either packaged scorer via --scorer:
    python3.12 evaluation/hermes_results/scoring/tests/test_exp02_bugs.py
    python3.12 evaluation/hermes_results/scoring/tests/test_exp02_bugs.py --scorer jb
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parents[1]

SCORER_TARGETS = {
    "exp02": {
        "path": SCORING_DIR,
        "module": "scorers_exp05",
        "label": "evaluation/hermes_results/scoring/scorers_exp05.py",
    },
    "jb": {
        "path": SCORING_DIR,
        "module": "scorers",
        "label": "evaluation/hermes_results/scoring/scorers.py",
    },
}


def _load_scorer(target: str):
    cfg = SCORER_TARGETS[target]
    sys.path.insert(0, str(cfg["path"]))
    mod = importlib.import_module(cfg["module"])
    return mod.parse_json_response, mod.score_task, cfg["label"]


_cli = argparse.ArgumentParser()
_cli.add_argument("--scorer", choices=list(SCORER_TARGETS), default="exp02")
_args, _ = _cli.parse_known_args()
parse_json_response, score_task, _SCORER_LABEL = _load_scorer(_args.scorer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED = "\033[91m"
GREEN = "\033[92m"
YEL = "\033[93m"
RST = "\033[0m"

_results: list[tuple[str, bool, str]] = []


def probe(name: str, is_bug: bool, summary: str) -> None:
    tag = f"{RED}BUG{RST}" if is_bug else f"{GREEN}OK {RST}"
    print(f"{tag}  {name}")
    print(f"      {summary}")
    print()
    _results.append((name, is_bug, summary))


def run(task: dict, response_obj) -> dict:
    import json
    response = response_obj if isinstance(response_obj, str) else json.dumps(response_obj)
    return score_task(task, response)


# ---------------------------------------------------------------------------
# 1. ranked_list_recall — single expected item, perfect prediction
# ---------------------------------------------------------------------------

def probe_ranked_single_expected():
    task = {
        "verification": {"method": "ranked_list_recall", "top_k": 5},
        "expected_output": {"top_risk_exposures": [{"type": "A", "magnitude": 0.5}]},
    }
    # Perfect prediction — identical single item.
    res = run(task, {"top_risk_exposures": [{"type": "A", "magnitude": 0.5}]})
    got = res["score"]
    # Expected behavior: perfect match → 1.0. Actual: order_score hard-coded
    # to 0 when len(exp_risks) < 2, so perfect pred scores 0.4+0.3+0 = 0.7.
    probe(
        "ranked_list_recall: single expected item, perfect prediction",
        is_bug=(got < 1.0 - 1e-9),
        summary=f"perfect match -> score={got:.3f} (expected 1.0). "
                f"order_score skipped because len(exp_risks) < 2.",
    )


# ---------------------------------------------------------------------------
# 2. ranked_list_recall — duplicate types collapse via set()
# ---------------------------------------------------------------------------

def probe_ranked_duplicate_types():
    # Expected has THREE risks, two share the same type. Model predicts only
    # ONE of the two factor_tilts and the drawdown. Intuition: recall should
    # be ~2/3, not 1.0.
    task = {
        "verification": {"method": "ranked_list_recall", "top_k": 5},
        "expected_output": {"top_risk_exposures": [
            {"type": "factor_tilt", "magnitude": 0.5},
            {"type": "factor_tilt", "magnitude": 0.3},
            {"type": "drawdown", "magnitude": 0.2},
        ]},
    }
    res = run(task, {"top_risk_exposures": [
        {"type": "factor_tilt", "magnitude": 0.5},
        {"type": "drawdown", "magnitude": 0.2},
    ]})
    d = res["details"]
    probe(
        "ranked_list_recall: duplicate types collapsed to a set",
        is_bug=(d["recall_at_k"] >= 1.0 - 1e-9),
        summary=f"pred covers 2 of 3 items but recall_at_k={d['recall_at_k']:.2f} "
                f"(expected ~0.67). Exp02 uses set(types), collapsing duplicates.",
    )


# ---------------------------------------------------------------------------
# 3. absolute_error — tolerance_pct = 0 divides by zero
# ---------------------------------------------------------------------------

def probe_absolute_error_zero_tolerance():
    task = {
        "verification": {"method": "absolute_error", "tolerance_pct": 0.0},
        "expected_output": {"estimated_pnl_pct": 1.0},
    }
    # Predict the exact value — should be a perfect score.
    res = run(task, {"estimated_pnl_pct": 1.0})
    got = res["score"]
    err = res.get("details", {}).get("error", "")
    probe(
        "absolute_error: tolerance_pct=0 (exact match requested)",
        is_bug=(got < 1.0 - 1e-9),
        summary=f"exact match on tol=0 -> score={got:.3f}, detail='{err}'. "
                f"ZeroDivisionError swallowed by outer try/except.",
    )


# ---------------------------------------------------------------------------
# 4. constraint_satisfaction_plus_cost — hallucinated trades aren't penalized
# ---------------------------------------------------------------------------

def probe_csc_hallucinated_trades():
    task = {
        "verification": {"method": "constraint_satisfaction_plus_cost",
                         "tolerance": {"turnover_excess_pct": 50}},
        "expected_output": {
            "post_trade_compliant": True,
            "total_turnover": 0.1,
            "trades": [{"symbol": "AAPL", "action": "buy"}],
        },
    }
    # Correct trade + 9 hallucinated ones.
    hallucinated = [{"symbol": f"BAD{i}", "action": "sell"} for i in range(9)]
    res = run(task, {
        "post_trade_compliant": True,
        "total_turnover": 0.1,
        "trades": [{"symbol": "AAPL", "action": "buy"}, *hallucinated],
    })
    d = res["details"]
    # Post-fix: F1 with precision=0.1, recall=1.0 -> ~0.18.
    probe(
        "constraint_satisfaction_plus_cost: hallucinated trades penalized (F1)",
        is_bug=(d["direction_match"] >= 0.5),
        summary=f"1 correct trade + 9 wrong -> direction_match={d['direction_match']:.2f} "
                f"(F1 = 2·P·R/(P+R)). score={res['score']:.3f}.",
    )


# ---------------------------------------------------------------------------
# 5. constraint_satisfaction_plus_cost — case-sensitive action comparison
# ---------------------------------------------------------------------------

def probe_csc_action_case():
    task = {
        "verification": {"method": "constraint_satisfaction_plus_cost"},
        "expected_output": {
            "post_trade_compliant": True,
            "total_turnover": 0.1,
            "trades": [{"symbol": "AAPL", "action": "buy"}],
        },
    }
    # Same trade but upper-cased action.
    res = run(task, {
        "post_trade_compliant": True,
        "total_turnover": 0.1,
        "trades": [{"symbol": "AAPL", "action": "BUY"}],
    })
    d = res["details"]
    probe(
        "constraint_satisfaction_plus_cost: action string is case-sensitive",
        is_bug=(d["direction_match"] < 1.0 - 1e-9),
        summary=f"action='BUY' vs 'buy' -> direction_match={d['direction_match']:.2f}. "
                f"No .lower()/.strip() normalization.",
    )


# ---------------------------------------------------------------------------
# 6. exact_match — empty-string status can silently match
# ---------------------------------------------------------------------------

def probe_exact_match_free_overall():
    # Agent returns ONLY overall_compliant, omits per-constraint detail.
    task = {
        "verification": {"method": "exact_match"},
        "expected_output": {
            "overall_compliant": True,
            "constraints": [
                {"type": "position_limit", "status": "PASS"},
                {"type": "sector_limit", "status": "PASS"},
            ],
        },
    }
    res = run(task, {"overall_compliant": True})
    probe(
        "exact_match: full credit for just the overall flag",
        is_bug=(res["score"] > 0.4 - 1e-9 and res["score"] <= 0.4 + 1e-9),
        summary=f"pred={{'overall_compliant': True}} -> score={res['score']:.3f}. "
                f"40% guaranteed for getting the boolean right without any per-constraint work.",
    )


# ---------------------------------------------------------------------------
# 7. parse_json_response — single-line fenced payload fails
# ---------------------------------------------------------------------------

def probe_parse_single_line_fence():
    # Some models emit ```json {"x":1} ``` on one line; some emit multi-line.
    # Exp02 only handles multi-line fences.
    single_line = '```json {"x": 1}```'
    got = parse_json_response(single_line)
    probe(
        "parse_json_response: single-line fenced JSON",
        is_bug=(got is None),
        summary=f"input: {single_line!r} -> parsed={got!r}. "
                f"One-line fence stripper consumes the content line too.",
    )


# ---------------------------------------------------------------------------
# 8. exact_match — duplicate predicted constraint types silently overwritten
# ---------------------------------------------------------------------------

def probe_exact_match_duplicate_pred_types():
    task = {
        "verification": {"method": "exact_match"},
        "expected_output": {
            "overall_compliant": False,
            "constraints": [{"type": "position_limit", "status": "FAIL"}],
        },
    }
    # Pred emits two position_limit entries; the LAST one wins silently.
    # A correct FAIL followed by an incorrect PASS → scored as PASS (no match).
    res = run(task, {
        "overall_compliant": False,
        "constraints": [
            {"type": "position_limit", "status": "FAIL"},   # correct, ignored
            {"type": "position_limit", "status": "PASS"},   # wrong, wins
        ],
    })
    d = res["details"]
    # Post-fix: contradictory duplicates are flagged, not silently last-wins.
    probe(
        "exact_match: duplicate pred types flagged explicitly",
        is_bug=(not d.get("duplicate_pred_types", False)),
        summary=f"first FAIL, second PASS -> constraint_accuracy={d['constraint_accuracy']:.2f}, "
                f"duplicate_pred_types={d.get('duplicate_pred_types')} (no silent overwrite).",
    )


# ---------------------------------------------------------------------------
# 9. absolute_error — missing sector attribution is silently ignored
# ---------------------------------------------------------------------------

def probe_absolute_error_ignores_attribution():
    task = {
        "verification": {
            "method": "absolute_error",
            "tolerance_pct": 1.0,
            "metrics": ["estimated_pnl_pct", "sector_attribution_mae"],
        },
        "expected_output": {
            "estimated_pnl_pct": 1.0,
            "attribution": {"by_sector": {"Tech": 0.6, "Finance": 0.4}},
        },
    }
    # Correct P&L, totally wrong sector attribution.
    res = run(task, {
        "estimated_pnl_pct": 1.0,
        "attribution": {"by_sector": {"Tech": -0.5, "Finance": -0.5}},
    })
    probe(
        "absolute_error: sector_attribution_mae in metrics is silently ignored",
        is_bug=(res["score"] >= 1.0 - 1e-9),
        summary=f"P&L exact but attribution off by 1.0 -> score={res['score']:.3f}. "
                f"verification.metrics advertises sector_attribution_mae; scorer doesn't use it.",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

PROBES = [
    probe_ranked_single_expected,
    probe_ranked_duplicate_types,
    probe_absolute_error_zero_tolerance,
    probe_csc_hallucinated_trades,
    probe_csc_action_case,
    probe_exact_match_free_overall,
    probe_parse_single_line_fence,
    probe_exact_match_duplicate_pred_types,
    probe_absolute_error_ignores_attribution,
]


def main() -> None:
    print(f"{YEL}Probing scorer at {_SCORER_LABEL}{RST}\n")
    for fn in PROBES:
        fn()
    bugs = sum(1 for _, is_bug, _ in _results if is_bug)
    total = len(_results)
    color = RED if bugs else GREEN
    print(f"{color}Summary: {bugs}/{total} probes surfaced a bug.{RST}")


if __name__ == "__main__":
    main()
