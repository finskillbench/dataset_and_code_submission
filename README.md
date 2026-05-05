# FinSkillBench: Benchmarking Reliable Agent Skills in Financial Markets

A point-in-time investment-management benchmark with **2,603 task episodes** across 12 subtasks and 3 domains (portfolio construction, risk management, fundamental analysis). Each episode provides point-in-time inputs, hidden ground truth, and a task-specific verifier. The benchmark evaluates three conditions: no skill, curated skill packages, and self-generated skills.

**Prerequisites:** Python ≥ 3.12 is required. Python 3.11 and earlier will not work (`pandas>=3.0.2` and `scipy>=1.17.1` require 3.12+). Commands below use `python3.12` explicitly so local `pyenv` shims or an uninstalled `.python-version` do not accidentally select an older interpreter.

---

## Quick Start: Verify Paper Results (2 minutes, no API keys)

```bash
python3.12 -m pip install -r analysis/requirements.txt   # pandas, numpy, scipy only

# Reproduce all paper tables (Tables 2–7)
python3.12 analysis/reproduce_tables.py --table all

# Verify every inline numeric claim
python3.12 analysis/verify_claims.py --claim all

# Verify golden paper values, coverage, and checksums
python3.12 analysis/verify_expected_results.py
python3.12 scripts/verify_submission_bundle.py --assert-match
shasum -a 256 -c MANIFEST.sha256

# Explore tasks interactively
python3.12 demo.py
```

Expected verification summary:

```text
verify_expected_results.py: all expected paper values match.
verify_claims.py: All checks passed.
verify_submission_bundle.py: All checks passed.
MANIFEST.sha256: all files OK
```

## Run a Live Evaluation (~10 minutes, 1 API key, ~$2)

```bash
python3.12 -m pip install -r requirements.txt            # Full eval deps
export OPENAI_API_KEY=...                  # or AZURE_AI_API_KEY, etc.

# Smoke test: 1 task per subtask × 1 model × 1 condition
python3.12 run_benchmark.py --smoke

# Specific slice
python3.12 run_benchmark.py --model gpt-4.1 --condition curated \
    --subtask unconstrained_optimization --limit 3
```

## Directory Layout

```
├── README.md                    This file
├── PAPER_MAPPING.md             Paper table/claim → source file mapping
├── LICENSE                      Apache 2.0 (code) + CC-BY-4.0 (data)
├── DATA_LICENSES.md             Per-source data license inventory
├── MANIFEST.sha256              Checksums for all packaged artifacts
│
├── data/                        Benchmark data (2,603 episodes)
│   ├── fundamentals/            2,243 FA episodes + processed XBRL panel
│   ├── portfolio_construction/  200 PC episodes + covariance matrices
│   ├── risk_management/         160 RM episodes + constructed objects
│   └── universe/                S&P 500 security master (latest.json)
│
├── skills/                      Curated skill packages
│   ├── fundamental-analysis/    SKILL.md + references/ + scripts/
│   ├── portfolio-construction/  SKILL.md + references/ + scripts/
│   └── risk-management/         SKILL.md + references/ + scripts/
│
├── results/                     Raw experiment outputs (17,820 + 7,200 rows)
│   ├── finskillbench_agent/     Main results (§5–§6)
│   └── hermes_agent/            Cross-harness results (§7)
│
├── analysis/                    Offline reproduction scripts
│   ├── reproduce_tables.py      --table 2|3|4|5|6|7|all
│   ├── verify_claims.py         --claim <name>|all
│   └── requirements.txt         Tier 1: pandas, numpy, scipy
│
├── run_benchmark.py             Unified CLI for live evaluations
├── demo.py                      Task explorer (no API keys)
├── requirements.txt             Tier 2: full eval dependencies
│
├── evaluation/                  Evaluation harness code
│   ├── finskillbench_agent/     Main harness (agent, lib, runners, tasks)
│   └── hermes_results/          Cross-harness verification assets
│
└── scripts/                     Build and verification utilities
```

## API Key Setup

| Model | Provider | Environment Variable |
|---|---|---|
| gpt-5.4, gpt-4.1 | OpenAI / Azure | `OPENAI_API_KEY` or `AZURE_AI_API_KEY` |
| claude-sonnet-4.6 | Anthropic / Azure | `OPENAI_API_KEY` or `AZURE_AI_API_KEY` |
| gemini-2.5-pro, gemini-3.1-flash-lite | Google Vertex AI | `VERTEX_API_KEY` |
| grok-4, DeepSeek-V3.2, Phi-4, glm-5.1 | OpenRouter | `OPENROUTER_API_KEY` |

## Results Schema

Each row in `results_all.jsonl` contains:

| Field | Type | Description |
|---|---|---|
| `model` | string | Model short name |
| `condition` | string | no_skill, curated, or self_generated |
| `task_id` | string | Unique task identifier |
| `sub_task` | string | One of 12 subtasks |
| `score` | float | 0.0–1.0, task-specific scorer output |
| `valid_json` | bool | Whether the agent returned valid JSON |
| `scoring_method` | string | Scorer used (e.g., l2_distance, metric_absolute_error) |
| `episodes` | int | Number of agent turns used |
| `total_input_tokens` | int | Total input tokens consumed |
| `total_output_tokens` | int | Total output tokens generated |
| `latency_seconds` | float | Wall-clock time for the evaluation |

See `results/README.md` for the full schema and loading examples.

## Known Exclusions

- **Ground truth directories**: Expected outputs are embedded in each episode JSON. Separate `ground_truth/` directories are omitted to avoid redundancy.
- **Data pipelines**: Scripts for regenerating data from raw sources are not included. The benchmark data is self-contained.
- **Validation files**: Build-time data quality checks are omitted. The shipped data has been validated.
- **infeasibility_detection**: This PC subtask was not part of the paper's 12-subtask evaluation.
- **Hermes Agent rerun**: The external Hermes Agent package is not included. Only precomputed result JSONL files and offline Table 7 verification are shipped.
- **Full trajectories**: Agent conversation logs exceed the submission size budget. They will be hosted externally.
- **Raw data sources**: Raw regulatory filings and financial data API responses are excluded.

## Citation

The paper and technical appendix are submitted separately to NeurIPS. The appendix (Sections A–C) documents:

- **Section A** — Data processing pipelines and ground truth derivation for all three domains, including processing logic, optimizer formulations, ground truth validation, and abbreviated input/output examples for all 12 subtasks.
- **Section B** — System prompts and task prompt templates for both the FinSkillBench and Hermes Agent harnesses, condition-specific modifications, and tool schemas.
- **Section C** — Detailed subtask descriptions with difficulty analysis, cognitive demand taxonomy, and complete scorer formula table (verified against `evaluation/finskillbench_agent/lib/scorers.py`).

```bibtex
@inproceedings{finskillbench2026,
  title={FinSkillBench: Benchmarking Reliable Agent Skills in Financial Markets},
  author={Anonymous},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track},
  year={2026}
}
```
