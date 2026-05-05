# Curated Skill Packages

These are the human-authored skill packages used in the **curated** experimental condition (paper §4.2).

## Structure

Each domain skill package contains:

```
<domain>/
├── SKILL.md          # Procedural document: workflow, data access, edge cases
├── references/       # Domain reference material (formulas, definitions)
└── scripts/          # Executable Python scripts (optimizers, parsers, etc.)
```

## Domains

| Directory | Domain | Paper section |
|---|---|---|
| `fundamental-analysis/` | Fundamental Analysis | §4.2 |
| `portfolio-construction/` | Portfolio Construction | §4.2 |
| `risk-management/` | Risk Management | §4.2 |

## Experimental Conditions

- **No-skill**: The skill-access interface is present but the skill corpus is empty.
- **Curated**: These packages are mounted. The agent discovers and reads SKILL.md, then executes scripts via `run_skill_script`.
- **Self-generated**: Curated material is withheld. The agent writes its own skill documents in a per-episode scratch space.

## How Skills Map to the Agent

The agent has three skill-related tools:
1. `load_skill` — Discovers and reads SKILL.md from the mounted skill directory
2. `load_references` — Reads reference documents from `references/`
3. `run_skill_script` — Executes scripts from `scripts/` with JSON I/O

In the curated condition, these tools resolve to the files in this directory. In the no-skill condition, they return "not found". In the self-generated condition, `save_skill` is additionally available for the agent to write its own SKILL.md.
