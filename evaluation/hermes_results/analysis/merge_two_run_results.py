"""Merge selected models from a source run's results.jsonl into a destination run's results.jsonl.

Usage (from analysis/ or any directory):
    python merge_two_run_results.py <src_run> <dst_run> [model1 model2 ...]

    src_run / dst_run: run folder names inside ../runs/ (relative to this script)
    models: optional list of model names to copy; omit to copy ALL models not already in dst

Example:
    python merge_two_run_results.py \\
        2026-04-27_skills_tools_all_fin_full_fa100 \\
        2026-04-27_skills_tools_all_fin_full_fa100_v2 \\
        claude-sonnet-4.6 gpt-5.4
"""
import json
import sys
from collections import Counter
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs"


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_run, dst_run = sys.argv[1], sys.argv[2]
    requested_models = set(sys.argv[3:]) if len(sys.argv) > 3 else None

    src = RUNS_DIR / src_run / "results.jsonl"
    dst = RUNS_DIR / dst_run / "results.jsonl"

    for p in (src, dst):
        if not p.exists():
            print(f"ERROR: not found: {p}")
            sys.exit(1)

    with open(dst) as f:
        existing_models = {json.loads(l).get("model") for l in f}

    with open(src) as f:
        src_rows = [l for l in f]

    if requested_models:
        to_add = [l for l in src_rows if json.loads(l).get("model") in requested_models]
    else:
        to_add = [l for l in src_rows if json.loads(l).get("model") not in existing_models]

    if not to_add:
        print("Nothing to append — all requested models already present in dst.")
        return

    print(f"Appending {len(to_add)} rows to {dst_run}/results.jsonl ...")
    with open(dst, "a") as f:
        for line in to_add:
            f.write(line if line.endswith("\n") else line + "\n")

    with open(dst) as f:
        rows = [json.loads(l) for l in f]

    print("Models after merge:", dict(Counter(r.get("model") for r in rows)))
    print("Total rows:", len(rows))


if __name__ == "__main__":
    main()
