#!/usr/bin/env python3
"""Merge per-subtask result files into results_all.jsonl.

Usage:
    python3.12 scripts/merge_results.py

Reads from results/finskillbench_agent/by_subtask/*.jsonl and writes
results/finskillbench_agent/results_all.jsonl.
"""
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
BY_SUBTASK = SUBMISSION_ROOT / "results" / "finskillbench_agent" / "by_subtask"
OUTPUT = SUBMISSION_ROOT / "results" / "finskillbench_agent" / "results_all.jsonl"


def main():
    lines = []
    for f in sorted(BY_SUBTASK.glob("*.jsonl")):
        with open(f) as fh:
            file_lines = fh.readlines()
            lines.extend(file_lines)
            print(f"  {f.name}: {len(file_lines)} rows")

    OUTPUT.write_text("".join(lines))
    print(f"\nWrote {len(lines)} rows to {OUTPUT.relative_to(SUBMISSION_ROOT)}")


if __name__ == "__main__":
    main()
