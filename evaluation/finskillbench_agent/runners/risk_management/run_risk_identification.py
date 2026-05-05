#!/usr/bin/env python3
"""Run risk_identification tasks across no_skill / curated / self_generated.

Usage:
    python experiments/zqbok_experiment05/runners/risk_management/run_risk_identification.py \
        --model gpt-4.1 --condition curated --limit 5 --workers 8

    # All models, all conditions
    python experiments/zqbok_experiment05/runners/risk_management/run_risk_identification.py \
        --model all --condition all

    # Resume a previous run
    python experiments/zqbok_experiment05/runners/risk_management/run_risk_identification.py \
        --resume runs/risk_identification/20260420_120000
"""
from _base import build_parser, run_subtask_experiment

SUB_TASK = "risk_identification"


def main() -> None:
    parser = build_parser(SUB_TASK)
    args = parser.parse_args()
    run_subtask_experiment(SUB_TASK, args)


if __name__ == "__main__":
    main()
