#!/usr/bin/env python3
"""Sanitize submission data to remove source-identifying references.

Replaces vendor-specific field values with neutral labels in:
- Ground-truth JSON `source` fields
- Episode JSON `xbrl_ref` fields
- Validation summary files
- Documentation files

Usage:
    python scripts/sanitize_submission_data.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = SUBMISSION_ROOT / "data"

SOURCE_MAP = {
    "xbrl+fmp": "regulatory+provider",
    "xbrl": "regulatory",
    "fmp": "provider",
    "computed": "computed",
}

XBRL_REF_REPLACEMENT = "data/fundamentals/processed/feature_store"


def sanitize_json_file(path: Path, modified_count: list) -> None:
    """Sanitize a single JSON file in-place."""
    try:
        text = path.read_text()
        data = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    changed = False

    # Replace source fields in ground truth
    if isinstance(data, dict):
        changed |= _sanitize_dict(data)

    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        modified_count[0] += 1


def _sanitize_dict(d: dict) -> bool:
    changed = False
    for key, val in list(d.items()):
        if key == "source" and isinstance(val, str) and val.lower() in SOURCE_MAP:
            d[key] = SOURCE_MAP[val.lower()]
            changed = True
        elif key == "xbrl_ref" and isinstance(val, str):
            d[key] = XBRL_REF_REPLACEMENT
            changed = True
        elif isinstance(val, dict):
            changed |= _sanitize_dict(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    changed |= _sanitize_dict(item)

    # Rename source-specific keys
    renames = {
        "edgar_coverage": "regulatory_coverage",
        "xbrl_coverage": "structured_data_coverage",
        "xbrl_fmp_agreement": "cross_source_agreement",
    }
    for old_key, new_key in renames.items():
        if old_key in d:
            d[new_key] = d.pop(old_key)
            changed = True

    return changed


def sanitize_markdown(path: Path) -> None:
    """Replace source references in markdown files."""
    text = path.read_text()
    original = text
    text = re.sub(r'\bXBRL\+FMP\b', 'regulatory+provider', text)
    text = re.sub(r'\bFMP statements\b', 'provider statements', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFMP segments\b', 'provider segments', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFMP API\b', 'financial data API', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFMP\b', 'provider', text)
    text = re.sub(r'\bEDGAR\b', 'regulatory filings', text)
    text = re.sub(r'\bSEC EDGAR\b', 'regulatory filings', text, flags=re.IGNORECASE)
    text = re.sub(r'\bxbrl_ref\b', 'feature_store_ref', text)
    if text != original:
        path.write_text(text)
        print(f"  Sanitized: {path.relative_to(SUBMISSION_ROOT)}")


def main():
    modified = [0]

    # Sanitize all JSON files in data/
    print("Sanitizing JSON files...")
    for f in DATA_DIR.rglob("*.json"):
        sanitize_json_file(f, modified)
    print(f"  Modified {modified[0]} JSON files")

    # Sanitize markdown documentation
    print("\nSanitizing documentation...")
    for md in DATA_DIR.rglob("*.md"):
        sanitize_markdown(md)

    # Also sanitize the task_registry README
    tr_readme = DATA_DIR / "task_registry" / "README.md"
    if tr_readme.exists():
        sanitize_markdown(tr_readme)

    # Sanitize the top-level data README
    data_readme = DATA_DIR / "README.md"
    if data_readme.exists():
        sanitize_markdown(data_readme)

    # Sanitize the submission README (remove any FMP references)
    sub_readme = SUBMISSION_ROOT / "README.md"
    if sub_readme.exists():
        sanitize_markdown(sub_readme)

    print("\nDone.")


if __name__ == "__main__":
    main()
