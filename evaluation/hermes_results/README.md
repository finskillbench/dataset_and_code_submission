# Hermes Cross-Harness Results

This directory contains offline verification assets for the cross-harness validation (paper §7).

## Scope

The external Hermes Agent package and live rerun path are **not included** in this submission. Only the following are shipped:

1. **Precomputed result JSONL files** (`results/hermes_agent/hermes_no_skill.jsonl` and `hermes_curated.jsonl`)
2. **Task manifests** (`datasets/`) used to interpret the result files
3. **Offline analysis** scripts for reproducing Table 7

## Verification

```bash
python analysis/reproduce_tables.py --table 7
```

This reads the shipped Hermes JSONL files and reproduces Table 7 from the paper.
