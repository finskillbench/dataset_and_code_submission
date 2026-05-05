"""FinSkillBench scoring package.

Covers RM, PC, and FA verification methods. RM scorers were originally
forked from `experiments/zqbok_experiment02/inspect_tasks/scorers.py` with
fixes 7.1 and 7.2 (see experiment_plan.md §7). PC and FA scorers were later
ported from `experiments/zqbok_experiment05/lib/scorers.py` with the
P1 + P2 fixes from its `scoring_methodology_review.md` applied:

  P1 #1  parameter_match — recursive leaf-level comparison.
  P1 #2  view_specification_and_weights — accepts `optimal_weights`.
  P1 #3  ranked_list_recall — multiset recall/precision + graded NDCG@k.
  P2 #4  absolute_error — 0.7·pnl + 0.3·sector-attribution MAE composite.
  P2 #5  earnings_quality_composite — numeric_ratio_weight default 0.3 so
         Beneish / accruals / income-quality ratios are actually scored.
  P2 #6  turnover_compliance_and_objective — weights + turnover + trade-list
         composite; reads `new_weights` with `weights` fallback.
"""

from .scorers import parse_json_response, score_task

__all__ = ["parse_json_response", "score_task"]
