#!/usr/bin/env python3
"""Run driver_decomposition tasks across no_skill / curated / self_generated.

Usage:
    python experiments/zqbok_experiment05/runners/fundamental_analysis/run_driver_decomposition.py \
        --model gpt-4.1 --condition curated --limit 5 --workers 8

    # All models, all conditions
    python experiments/zqbok_experiment05/runners/fundamental_analysis/run_driver_decomposition.py \
        --model all --condition all

    # Resume a previous run
    python experiments/zqbok_experiment05/runners/fundamental_analysis/run_driver_decomposition.py \
        --resume runs/driver_decomposition/20260420_120000
"""
from _base import build_parser, run_subtask_experiment

SUB_TASK = "driver_decomposition"


def main() -> None:
    parser = build_parser(SUB_TASK)
    args = parser.parse_args()
    run_subtask_experiment(SUB_TASK, args)


if __name__ == "__main__":
    main()
