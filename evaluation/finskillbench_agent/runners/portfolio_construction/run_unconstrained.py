#!/usr/bin/env python3
"""Run unconstrained_optimization tasks across no_skill / curated / self_generated.

Usage:
    python experiments/zqbok_experiment05/runners/portfolio_construction/run_unconstrained.py \
        --model gpt-4.1 --condition curated --limit 5 --workers 8

    # All models, all conditions
    python experiments/zqbok_experiment05/runners/portfolio_construction/run_unconstrained.py \
        --model all --condition all

    # Resume a previous run
    python experiments/zqbok_experiment05/runners/portfolio_construction/run_unconstrained.py \
        --resume runs/unconstrained_optimization/20260420_120000
"""
from _base import build_parser, run_subtask_experiment

SUB_TASK = "unconstrained_optimization"


def main() -> None:
    parser = build_parser(SUB_TASK)
    args = parser.parse_args()
    run_subtask_experiment(SUB_TASK, args)


if __name__ == "__main__":
    main()
