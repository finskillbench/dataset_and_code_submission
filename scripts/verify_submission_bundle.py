#!/usr/bin/env python3
"""Verify the submission bundle reproduces all paper results.

Usage:
    python scripts/verify_submission_bundle.py
    python scripts/verify_submission_bundle.py --assert-match

Runs inside the submission directory only. Does not read dev-repo files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


def check_manifest() -> list[str]:
    """Verify MANIFEST.sha256 hashes match packaged files."""
    manifest = SUBMISSION_ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        return ["MANIFEST.sha256 not found — run scripts/build_submission.py first"]
    errors = []
    for line in manifest.read_text().strip().split("\n"):
        if not line.strip():
            continue
        expected_hash, rel_path = line.split("  ", 1)
        fpath = SUBMISSION_ROOT / rel_path
        if not fpath.exists():
            errors.append(f"Missing file: {rel_path}")
            continue
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        if h.hexdigest() != expected_hash:
            errors.append(f"Hash mismatch: {rel_path}")
    return errors


def check_episode_count() -> list[str]:
    """Verify 2,603 evaluated episodes and no infeasibility_detection."""
    errors = []
    data_dir = SUBMISSION_ROOT / "data"

    # FA
    fa_count = 0
    fa_dir = data_dir / "fundamentals" / "episodes" / "layer_a"
    if fa_dir.exists():
        for st_dir in fa_dir.iterdir():
            if st_dir.is_dir():
                fa_count += len(list(st_dir.glob("*.json")))

    # PC
    pc_count = 0
    pc_dir = data_dir / "portfolio_construction" / "episodes" / "layer_a"
    if pc_dir.exists():
        for st_dir in pc_dir.iterdir():
            if st_dir.is_dir():
                if st_dir.name == "infeasibility_detection":
                    errors.append(f"infeasibility_detection found in submission: {st_dir}")
                else:
                    pc_count += len(list(st_dir.glob("*.json")))

    # RM
    rm_count = 0
    rm_dir = data_dir / "risk_management" / "episodes" / "layer_a"
    if rm_dir.exists():
        for st_dir in rm_dir.iterdir():
            if st_dir.is_dir():
                rm_count += len(list(st_dir.glob("*.json")))

    total = fa_count + pc_count + rm_count
    if total != 2603:
        errors.append(f"Episode count: expected 2603, got {total} (FA={fa_count}, PC={pc_count}, RM={rm_count})")
    return errors


def check_results_coverage() -> list[str]:
    """Verify results_all.jsonl has 17,820 rows with correct coverage."""
    errors = []
    path = SUBMISSION_ROOT / "results" / "finskillbench_agent" / "results_all.jsonl"
    if not path.exists():
        return ["results_all.jsonl not found"]

    rows = [json.loads(l) for l in open(path)]
    if len(rows) != 17820:
        errors.append(f"results_all.jsonl: expected 17820 rows, got {len(rows)}")

    models = set(r["model"] for r in rows)
    if len(models) != 9:
        errors.append(f"Expected 9 models, got {len(models)}: {sorted(models)}")

    conditions = set(r["condition"] for r in rows)
    if conditions != {"no_skill", "curated", "self_generated"}:
        errors.append(f"Expected 3 conditions, got: {sorted(conditions)}")

    subtasks = set(r["sub_task"] for r in rows)
    if len(subtasks) != 12:
        errors.append(f"Expected 12 subtasks, got {len(subtasks)}: {sorted(subtasks)}")

    return errors


def check_hermes_coverage() -> list[str]:
    """Verify Hermes result files have correct row counts."""
    errors = []
    ns_path = SUBMISSION_ROOT / "results" / "hermes_agent" / "hermes_no_skill.jsonl"
    cur_path = SUBMISSION_ROOT / "results" / "hermes_agent" / "hermes_curated.jsonl"

    if ns_path.exists():
        ns_count = sum(1 for _ in open(ns_path))
        if ns_count != 1920:
            errors.append(f"hermes_no_skill.jsonl: expected 1920 rows, got {ns_count}")
    else:
        errors.append("hermes_no_skill.jsonl not found")

    if cur_path.exists():
        cur_count = sum(1 for _ in open(cur_path))
        if cur_count != 5280:
            errors.append(f"hermes_curated.jsonl: expected 5280 rows, got {cur_count}")
    else:
        errors.append("hermes_curated.jsonl not found")

    return errors


def check_reproduce_tables() -> list[str]:
    """Run reproduce_tables.py and verify it completes without error."""
    errors = []
    script = SUBMISSION_ROOT / "analysis" / "reproduce_tables.py"
    result = subprocess.run(
        [sys.executable, str(script), "--table", "all", "--format", "json"],
        capture_output=True, text=True, cwd=str(SUBMISSION_ROOT),
    )
    if result.returncode != 0:
        errors.append(f"reproduce_tables.py failed: {result.stderr[:500]}")
    return errors


def check_verify_claims() -> list[str]:
    """Run verify_claims.py and check all claims pass."""
    errors = []
    script = SUBMISSION_ROOT / "analysis" / "verify_claims.py"
    result = subprocess.run(
        [sys.executable, str(script), "--claim", "all", "--format", "json"],
        capture_output=True, text=True, cwd=str(SUBMISSION_ROOT),
    )
    if result.returncode != 0:
        errors.append(f"verify_claims.py failed (exit {result.returncode})")
        return errors

    try:
        claims = json.loads(result.stdout)
        for claim in claims:
            if not claim.get("pass"):
                errors.append(f"Claim failed: {claim['claim']} — "
                              f"expected {claim['paper_value']}, got {claim['computed_value']}")
    except json.JSONDecodeError:
        errors.append("Could not parse verify_claims.py JSON output")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify submission bundle.")
    parser.add_argument("--assert-match", action="store_true",
                        help="Exit nonzero on any failure")
    args = parser.parse_args()

    checks = [
        ("MANIFEST.sha256 integrity", check_manifest),
        ("Episode count (2,603)", check_episode_count),
        ("Results coverage (17,820 rows)", check_results_coverage),
        ("Hermes coverage (1,920 + 5,280)", check_hermes_coverage),
        ("reproduce_tables.py", check_reproduce_tables),
        ("verify_claims.py", check_verify_claims),
    ]

    all_errors = []
    report_lines = ["# Submission Verification Report", ""]

    for name, check_fn in checks:
        errors = check_fn()
        status = "✓ PASS" if not errors else "✗ FAIL"
        print(f"  {status}  {name}")
        report_lines.append(f"## {name}")
        report_lines.append(f"**{status}**")
        if errors:
            for e in errors:
                print(f"         {e}")
                report_lines.append(f"- {e}")
            all_errors.extend(errors)
        report_lines.append("")

    # Write report
    report_path = SUBMISSION_ROOT / "submission_verification_report.md"
    report_path.write_text("\n".join(report_lines))

    report_json = {
        "all_pass": len(all_errors) == 0,
        "total_checks": len(checks),
        "errors": all_errors,
    }
    json_path = SUBMISSION_ROOT / "submission_verification_report.json"
    json_path.write_text(json.dumps(report_json, indent=2))

    print(f"\nReport written to {report_path}")

    if all_errors:
        print(f"\n{len(all_errors)} error(s) found.")
        if args.assert_match:
            sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
