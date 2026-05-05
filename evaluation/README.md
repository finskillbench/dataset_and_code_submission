# Evaluation Harness

## FinSkillBench Agent (`finskillbench_agent/`)

The main evaluation harness used for paper §5–§6. Contains:

- `agent/` — Function-calling agent loop (`fc_loop.py`), skill document loading (`skill_docs.py`), tool registry (`tools.py`)
- `lib/` — Scorers (`scorers.py`), task loader (`task_loader.py`), financial engines (`engines/optimizer.py`, `engines/risk_engine.py`)
- `tasks/` — Task loading with XBRL panel filtering, instruction building, output schema derivation
- `runners/` — 12 subtask runners (3 domains × 4–5 subtasks each), each with parallel execution support
- `scripts/` — Aggregation, rescoring, and statistics computation
- `tests/` — Scorer unit tests

### Running evaluations

Use the top-level `run_benchmark.py` CLI, which dispatches to these runners:

```bash
python run_benchmark.py --smoke                    # Quick test
python run_benchmark.py --model gpt-4.1 --subtask normalization --limit 5
```

### Model registry

Models are routed through LiteLLM. The mapping from short names to API routes is in `run_benchmark.py`.

## Hermes Results (`hermes_results/`)

Offline verification assets for the cross-harness validation (paper §7). Contains:

- `datasets/` — Task manifests used to interpret the shipped Hermes result JSONL files
- `analysis/` — Offline analysis scripts for Table 7
- `scoring/` — Scorer wrappers used by the Hermes harness

The external Hermes Agent package is not included. Only precomputed results and verification tools are shipped.
