#!/usr/bin/env python3
"""Generate MANIFEST.sha256 for all files in the submission bundle.

Usage:
    python3.12 scripts/build_submission.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = SUBMISSION_ROOT / "MANIFEST.sha256"

EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "runs", ".venv"}
EXCLUDE_FILES = {"MANIFEST.sha256"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = []
    for f in sorted(SUBMISSION_ROOT.rglob("*")):
        if not f.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in f.parts):
            continue
        if f.name in EXCLUDE_FILES:
            continue
        rel = f.relative_to(SUBMISSION_ROOT)
        digest = sha256_file(f)
        entries.append(f"{digest}  {rel}")

    MANIFEST_PATH.write_text("\n".join(entries) + "\n")
    print(f"Wrote {len(entries)} entries to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
