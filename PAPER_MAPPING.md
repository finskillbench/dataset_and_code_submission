# Paper-to-Artifact Traceability Map

Every table and inline claim in the paper maps to a specific results file and a script that reproduces it.

## Paper Tables → Source Data

| Paper ref | Content | Source file | Rows | Reproduction script |
|---|---|---|---|---|
| **Table 1** (§3) | Scorer-to-subtask mapping | Static (no data) | — | — |
| **Table 2** (§6.1) | Aggregate condition means (excl. Phi-4) | `results/finskillbench_agent/results_all.jsonl` | 15,840 | `python3.12 analysis/reproduce_tables.py --table 2` |
| **Table 3** (§6.2) | Domain-level results (excl. Phi-4) | same | 15,840 | `python3.12 analysis/reproduce_tables.py --table 3` |
| **Table 4** (§6.3) | Subtask-level results (excl. Phi-4) | same | 15,840 | `python3.12 analysis/reproduce_tables.py --table 4` |
| **Table 5** (§6.4) | Per-model results (all 9 models) | same | 17,820 | `python3.12 analysis/reproduce_tables.py --table 5` |
| **Table 6** (§6.5) | Cost and interaction overhead (excl. Phi-4) | same | 15,840 | `python3.12 analysis/reproduce_tables.py --table 6` |
| **Table 7** (§7.1) | Hermes: aggregate, domain, model | `results/hermes_agent/hermes_no_skill.jsonl` + `hermes_curated.jsonl` | 1,920 + 5,280 | `python3.12 analysis/reproduce_tables.py --table 7` |

## Paper Inline Claims → Verification

| Claim (section) | Value | How to verify |
|---|---|---|
| 2,603 task episodes (§1) | Count of unique task_ids across all episode JSONs | `python3.12 analysis/verify_claims.py --claim episode_count` |
| 17,820 total evaluations (§5) | Row count of `results_all.jsonl` | `python3.12 analysis/verify_claims.py --claim eval_count` |
| 96.4% valid JSON (§5) | `valid_json` field in results | `python3.12 analysis/verify_claims.py --claim validity` |
| 644 invalid_submission (§5) | `scoring_method == "invalid_submission"` | `python3.12 analysis/verify_claims.py --claim invalid_count` |
| 3,736 max_turns_exhausted (§5) | `error` field contains `max_turns` | `python3.12 analysis/verify_claims.py --claim max_turns` |
| Phi-4 mean 0.000 (§5) | Filter `model == "Phi-4"`, compute mean score | `python3.12 analysis/verify_claims.py --claim phi4` |
| Curated Δ +0.162 (§6.1) | Paired delta from Table 2 | `python3.12 analysis/verify_claims.py --claim curated_delta` |
| Self-gen Δ +0.005 (§6.1) | Paired delta from Table 2 | `python3.12 analysis/verify_claims.py --claim selfgen_delta` |
| Hermes overall Δ +0.325 (§7.1) | From Table 7 | `python3.12 analysis/verify_claims.py --claim hermes_delta` |
| 97.7% self-gen skill loading (§6.5) | Skill-load events in results | `python3.12 analysis/verify_claims.py --claim selfgen_load` |

Run `python3.12 analysis/verify_expected_results.py` to compare all Table 2-7 display values, inline claim values, and packaged coverage counts against the checked-in golden snapshot.

## Run Directory Provenance

The paper reports results from exactly two experiment configurations:

| Paper section | Experiment | Condition(s) | Models | Evaluations |
|---|---|---|---|---|
| §5–§6 (main results) | FinSkillBench Agent | no_skill, curated, self_generated | 9 (gpt-5.4, claude-sonnet-4.6, gpt-4.1, gemini-2.5-pro, grok-4, DeepSeek-V3.2, Phi-4, glm-5.1, gemini-3.1-flash-lite) | 17,820 |
| §7 (cross-harness, no-skill) | Hermes Agent | no_skill | 8 (gpt-5.4, gpt-4.1, claude-sonnet-4.6, gemini-2.5-pro, glm-5.1, grok-4.20, DeepSeek-V3.2, gemma-4-31b-it) | 1,920 |
| §7 (cross-harness, curated) | Hermes Agent | curated | 8 (same as above) | 5,280 |

**Note:** The Hermes experiment uses a different model set than the main experiment: it drops Phi-4 and gemini-3.1-flash-lite, adds gemma-4-31b-it, and uses grok-4.20 instead of grok-4.
