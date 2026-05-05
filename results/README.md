# Results

Pre-computed experiment outputs that enable offline verification of every paper table and claim.

## Files

| File | Paper section | Rows | Description |
|---|---|---|---|
| `finskillbench_agent/results_all.jsonl` | §5–§6, Tables 2–6 | 17,820 | Main results: 9 models × 3 conditions × 12 subtasks |
| `finskillbench_agent/by_subtask/*.jsonl` | Same (split by subtask) | 17,820 | Same data, one file per subtask |
| `hermes_agent/hermes_no_skill.jsonl` | §7, Table 7 | 1,920 | Cross-harness no-skill: 8 models × 240 episodes |
| `hermes_agent/hermes_curated.jsonl` | §7, Table 7 | 5,280 | Cross-harness curated: 8 models × 660 episodes |

## JSONL Schema

Each line is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Model short name (e.g., "gpt-4.1") |
| `litellm_model` | string | LiteLLM route string used for API calls |
| `model_tier` | string | "frontier" or "open_weight" |
| `domain` | string | "portfolio_construction", "risk_management", or "fundamental_analysis" |
| `condition` | string | "no_skill", "curated", or "self_generated" |
| `task_id` | string | Unique task identifier |
| `sub_task` | string | One of 12 subtask names |
| `difficulty` | string | "easy", "medium", or "hard" |
| `as_of_date` | string | Point-in-time date for the task |
| `run_idx` | int | Run index (0 for single-run experiments) |
| `score` | float | 0.0–1.0, task-specific scorer output |
| `valid_json` | bool | Whether the agent returned parseable JSON |
| `scoring_method` | string | Scorer name (e.g., "l2_distance", "metric_absolute_error") |
| `scoring_details` | object | Scorer-specific breakdown |
| `episodes` | int | Number of agent turns used |
| `total_input_tokens` | int | Total input tokens consumed |
| `total_output_tokens` | int | Total output tokens generated |
| `skills_loaded` | list | Skill files loaded during the episode |
| `tool_calls_log` | list | Tool call trace (name, args, result, duration) |
| `error` | string/null | Error message if the evaluation failed |
| `latency_seconds` | float | Wall-clock time for the evaluation |

## Loading Examples

```python
import pandas as pd

# Load main results
df = pd.read_json("finskillbench_agent/results_all.jsonl", lines=True)

# Exclude Phi-4 for condition analysis
d = df[df["model"] != "Phi-4"]

# Per-condition means
d.groupby("condition")["score"].mean()

# Per-subtask means
d.groupby(["sub_task", "condition"])["score"].mean().unstack()

# Load Hermes results
ns = pd.read_json("hermes_agent/hermes_no_skill.jsonl", lines=True)
cur = pd.read_json("hermes_agent/hermes_curated.jsonl", lines=True)
```
