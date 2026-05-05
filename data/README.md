# FinSkillBench Data

A benchmark dataset for evaluating AI agents on institutional-grade financial analysis tasks. The dataset covers 30 S&P 500 tickers across three skill domains — Fundamental Analysis, Portfolio Construction, and Risk Management — with 2,455 atomic evaluation tasks, deterministic ground truth, and structured verification methods.

This README is the single entry point for understanding the dataset. It explains what data exists, how it's organized, how inputs map to outputs and ground truth, and how to use it for evaluation.

---

## Quick Start

```python
import json
from pathlib import Path

# Load a task
task = json.loads(Path("fundamentals/episodes/layer_a/normalization/task_fa_norm_AAPL_2024-03-30.json").read_text())

print(task["task_id"])          # "fa_norm_AAPL_2024-03-30"
print(task["skill"])            # "fundamental_analysis"
print(task["sub_task"])         # "normalization"
print(task["as_of_date"])       # "2024-05-03" (filing date — point-in-time)
print(task["input"].keys())     # symbol, period_end, form_type, filing_dir, feature_store_ref, tools_available
print(task["expected_output"])  # {"metrics": {"revenue": 90753000000, "operating_income": 27900000000, ...}}
print(task["verification"])     # {"method": "metric_absolute_error", "tolerance_pct": 5.0, ...}

# Load the corresponding ground truth
gt = json.loads(Path("fundamentals/ground_truth/normalization/AAPL_2024-03-30.json").read_text())
```

---

## Dataset Overview

| Dimension | Value |
|---|---|
| Equity universe | 30 S&P 500 tickers (stratified across 11 GICS sectors) |
| Time range | 2017–2025 (varies by domain) |
| Skill domains | 3 (Fundamental Analysis, Portfolio Construction, Risk Management) |
| Sub-tasks | 13 distinct task types |
| Total tasks | 2,455 atomic evaluation episodes |
| Ground truth labels | ~6,500 files (all deterministic / Tier 2 computed) |
| Point-in-time discipline | All inputs stamped with publication date; no future leakage |
| Data sources | SEC regulatory filings (filings, XBRL), provider (statements, segments, prices), Ken French (factors), FRED (macro) |

### Universe

The 30-ticker sample is drawn from the full S&P 500 security master (`universe/sp500_security_master_latest.json`), stratified to cover all 11 GICS sectors. Each ticker record includes CIK, ISIN, CUSIP, sector, industry, exchange, and IPO date.

```
data/universe/
├── sp500_security_master_latest.json   # Full ~503 constituents
├── sp500_security_master_sample30.json # 30-ticker evaluation subset
└── raw/                                # Source data for master construction
```

---

## Directory Structure

```
data/
├── README.md                          # This file
├── universe/                          # S&P 500 security master
├── fundamentals/                      # Skill 1: Fundamental Analysis
│   ├── raw/                           #   Source data (regulatory filings + provider)
│   ├── processed/                     #   Parsed filings, timelines, XBRL panel
│   ├── ground_truth/                  #   Labels for 3 sub-tasks
│   ├── episodes/                      #   2,243 task JSONs
│   └── validation/                    #   Data quality checks
├── portfolio_construction/            # Skill 2: Portfolio Construction
│   ├── raw/                           #   Prices, factors, macro
│   ├── processed/                     #   Returns, covariance, views, constraints
│   ├── ground_truth/                  #   Optimizer-derived reference portfolios
│   ├── episodes/                      #   52 task JSONs
│   └── validation/                    #   Data quality checks
├── risk_management/                   # Skill 3: Risk Management
│   ├── raw/                           #   Factors, macro
│   ├── processed/                     #   Returns, covariance, factor exposures
│   ├── constructed/                   #   Scenario portfolios, mandates, stress scenarios
│   ├── ground_truth/                  #   Deterministic risk computations
│   ├── episodes/                      #   160 task JSONs
│   └── validation/                    #   Data quality checks
└── task_registry/                     # Versioned manifest of all tasks
    ├── manifest.json
    ├── generate_registry.py
    └── domains/
        ├── fundamental_analysis.json
        ├── portfolio_construction.json
        └── risk_management.json
```

---

## Tasks and Episodes

The dataset is organized around **tasks** — self-contained evaluation units. Each task is stored as a JSON file called an **episode**. In the current dataset (Layer A), every episode contains exactly one atomic task. The terms are interchangeable.

The naming distinction exists because the design supports a future Layer B where a single episode would chain multiple tasks into a multi-step workflow (e.g., analyze a filing → construct a portfolio → assess its risk). For now, one episode = one task.

Every task JSON file lives in `{domain}/episodes/layer_a/{sub_task}/` and bundles three things:

- `input` — all data the agent receives (inline values or file references to raw/processed data)
- `expected_output` — the ground truth answer the agent should produce
- `verification` — the scoring method and thresholds used to evaluate the agent's response

Here is a concrete example (FA normalization, abbreviated):

```json
{
  "task_id": "fa_norm_AAPL_2024-03-30",
  "skill": "fundamental_analysis",
  "sub_task": "normalization",
  "as_of_date": "2024-05-03",
  "difficulty": "easy",

  "input": {
    "symbol": "AAPL",
    "period_end": "2024-03-30",
    "form_type": "10-Q",
    "filing_dir": "data/fundamentals/processed/filings/0000320193/000032019324000069",
    "feature_store_ref": "data/fundamentals/raw/edgar/xbrl",
    "tools_available": ["query_xbrl"]
  },

  "expected_output": {
    "metrics": {
      "revenue": 90753000000,
      "operating_income": 27900000000,
      "net_income": 23636000000,
      "eps_diluted": 1.53,
      "total_assets": 337411000000,
      "total_liabilities": 263217000000,
      "stockholders_equity": 74194000000,
      "operating_cash_flow": 62585000000,
      "operating_margin": 0.3074,
      "ebitda": 33584000000.0
    }
  },

  "verification": {
    "method": "metric_absolute_error",
    "tolerance_pct": 5.0,
    "metrics": ["mae", "within_threshold_pct"]
  }
}
```

The `input` tells the agent: "Here is AAPL's 10-Q for the period ending 2024-03-30. The filing text is at this path, XBRL data is at that path, and you can call `query_xbrl` to look up structured facts." The `expected_output` is the set of normalized metrics the agent should produce. The `verification` says: "Score each metric by absolute error; the agent passes if it's within 5% of the ground truth."

### More examples by domain

**Portfolio Construction** — the agent receives numeric data inline (expected returns, constraints) and a file reference for the covariance matrix:

```json
{
  "task_id": "pc_con_2025-01-31_medium_max_sharpe",
  "input": {
    "expected_returns": {"AAPL": 0.0012, "TSLA": -0.003, "...": "..."},
    "covariance_matrix": {"ref": "data/portfolio_construction/processed/covariance/2025-01-31.npz"},
    "constraints": {
      "long_only": true, "max_weight": 0.066, "max_turnover": 0.2,
      "max_tracking_error": 0.03, "sector_limits": {"Financials": 0.2, "...": "..."}
    },
    "objective": "max_sharpe",
    "tools_available": ["optimize_portfolio"]
  },
  "expected_output": {
    "weights": {"AIG": 0.0426, "DHR": 0.0656, "LNT": 0.0656, "...": "..."}
  },
  "verification": {
    "method": "constraint_satisfaction_and_objective",
    "objective_gap_threshold_pct": 5.0
  }
}
```

**Risk Management** — the agent receives a portfolio with planted risks and a stress scenario:

```json
{
  "task_id": "rm_st_portfolio_concentrated_2023_24_rebound_000_stress_2018_selloff",
  "input": {
    "portfolio": {
      "holdings": [
        {"symbol": "AIG", "weight": 0.15},
        {"symbol": "C", "weight": 0.15},
        {"symbol": "USB", "weight": 0.15},
        "..."
      ]
    },
    "stress_scenario": {"scenario_id": "stress_2018_selloff"},
    "market_data": {"factor_exposures": "data/risk_management/processed/factor_exposures.parquet"}
  },
  "expected_output": {
    "estimated_pnl_pct": -0.167,
    "attribution": {
      "by_sector": [
        {"sector": "Financial Services", "contribution_pct": -0.104},
        {"sector": "Industrials", "contribution_pct": -0.011}
      ]
    }
  },
  "verification": {
    "method": "absolute_error",
    "tolerance_pct": 1.0
  }
}
```

### How prompts are constructed

Task JSONs are pure data — they contain no prompt text. The prompt the agent actually sees is assembled at evaluation time by the runner code in `experiments/`. This separation is intentional: you can swap prompt strategies, system messages, or agent protocols without touching the dataset.

The runner takes a task JSON and builds a two-message prompt:

1. A **system message** with evaluation rules (point-in-time discipline, output format, no refusals)
2. A **user message** with the task type, `as_of_date`, inlined input data, and an output schema derived from `expected_output` (field names and types only — no ground truth values)

Optionally, a skill document is injected between the task header and input data (for the curated/self-generated experimental conditions).

Here is the actual prompt generated from the AAPL normalization task shown above:

```
── System message ──────────────────────────────────────────

You are a financial analysis agent participating in a controlled
evaluation benchmark. Your task is to analyze point-in-time financial
data and produce structured outputs.

Rules:
1. Point-in-time discipline: Use ONLY the data provided in the task.
   Do NOT use any knowledge of events after the as_of_date.
2. Respond with a valid JSON object matching the expected output schema.
3. Do NOT include any text before or after the JSON object.
4. If you cannot determine an answer, set the field to null.
5. Do NOT refuse the task. This is an academic benchmark, not
   investment advice.

── User message ────────────────────────────────────────────

## Task: normalization
Extract and normalize a standard set of financial metrics from a
company's SEC filing (10-K or 10-Q). Using the filing text and
structured XBRL data provided, compute the following canonical
metrics: revenue, operating income, net income, diluted EPS,
total assets, total liabilities, stockholders' equity, operating
cash flow, operating margin, and EBITDA. Values should be reported
in their original units as filed (USD for dollar amounts, ratio
for margins, per-share for EPS). Where the filing contains
non-standard line items or reclassifications, map them to the
closest canonical metric using GAAP definitions.

as_of_date: 2024-05-03
difficulty: easy

## Input Data
{
  "symbol": "AAPL",
  "period_end": "2024-03-30",
  "form_type": "10-Q",
  "filing_dir": "data/fundamentals/processed/filings/0000320193/...",
  "feature_store_ref": "data/fundamentals/raw/edgar/xbrl",
  "tools_available": ["query_xbrl"]
}

## Expected Output Schema
Respond with a JSON object matching this structure:
{
  "metrics": {
    "revenue": "<number>",
    "operating_income": "<number>",
    "net_income": "<number>",
    "eps_diluted": "<number>",
    "total_assets": "<number>",
    "total_liabilities": "<number>",
    "stockholders_equity": "<number>",
    "operating_cash_flow": "<number>",
    "operating_margin": "<number>",
    "ebitda": "<number>"
  }
}

Respond with ONLY the JSON object. No explanation, no markdown fences.
```

The output schema is auto-derived from `expected_output` by replacing concrete values with type placeholders (`<number>`, `<string>`, `<boolean>`). The agent never sees the ground truth values — only the shape of the expected response.

The prompt construction code lives in `experiments/zqbok_experiment02/inspect_tasks/portfolio_construction.py` (`build_prompt()`) and follows the same pattern for all domains.

---

## Task Taxonomy

### Fundamental Analysis (2,243 tasks)

| Sub-task | Count | What the agent does | Input | Expected output | Verification |
|---|---|---|---|---|---|
| Normalization | 244 | Extract and normalize financial metrics from a filing | Ticker, period, filing text, XBRL facts | 10 canonical metrics (revenue, EBITDA, margins, etc.) | Metric absolute error; ±5% threshold |
| Earnings quality | 1,002 | Assess financial health and flag accounting concerns | Ticker, period, filing text, XBRL facts | Piotroski F-Score (0–9), Beneish M-Score, accruals ratio, quality flags | Composite: Piotroski MAE + Beneish flag accuracy + flag F1 |
| Driver decomposition | 997 | Explain quarter-over-quarter revenue and margin changes | Two consecutive filings + XBRL deltas | Revenue drivers by segment, margin drivers by expense line | Driver F1@5 + direction accuracy |

### Portfolio Construction (52 tasks)

| Sub-task | Count | What the agent does | Input | Expected output | Verification |
|---|---|---|---|---|---|
| Unconstrained optimization | 12 | Maximize Sharpe ratio given returns + covariance | Expected returns, covariance matrix, risk-free rate | Portfolio weights | L2 distance to optimizer solution |
| Constrained optimization | 12 | Optimize subject to mandate constraints | Returns, covariance, constraint set, benchmark | Weights + constraint satisfaction | Constraint pass/fail + objective gap |
| Rebalancing | 6 | Rebalance existing portfolio under turnover limits | Current portfolio, new views, turnover limit | Trade list | Turnover compliance + objective improvement |
| Tool-use parameterization | 10 | Correctly call an optimization tool with right params | Views, constraints, optimizer API spec | Correct API call parameters | Exact parameter match |
| Infeasibility detection | 10 | Detect when constraints make optimization infeasible | Returns, covariance, conflicting constraints | Infeasibility flag + explanation | Binary classification |
| Black-Litterman | 2 | Combine market equilibrium with investor views | Prior returns, views, confidence, covariance | Posterior weights | View specification + weight accuracy |

### Risk Management (160 tasks)

| Sub-task | Count | What the agent does | Input | Expected output | Verification |
|---|---|---|---|---|---|
| Risk identification | 40 | Identify and rank top risk exposures | Portfolio snapshot, factor exposures, returns | Ranked risk list (factor tilts, concentration, etc.) | Recall@5, NDCG@5 |
| Constraint monitoring | 40 | Check portfolio compliance against mandate | Portfolio holdings, mandate specification | Pass/fail per constraint, violation magnitudes | Exact match per constraint |
| Stress testing | 40 | Estimate portfolio P&L under stress scenario | Portfolio, stress scenario definition, market data | Estimated P&L %, sector/factor attribution | Absolute error vs. computed P&L |
| Risk remediation | 40 | Propose minimal trades to restore compliance | Non-compliant portfolio, mandate | Trade list, post-trade compliance status | Constraint satisfaction + turnover vs. optimal |

---

## Data Flow: Input → Ground Truth → Task

Each domain follows the same pipeline pattern:

```
Raw data (downloads)
    ↓
Processed data (cleaned, aligned, derived)
    ↓
Ground truth (deterministic labels computed from processed data)
    ↓
Episodes (task JSONs that bundle input refs + expected output + verification)
```

### How tasks reference data

A task's `input` object contains either inline data or file references. The pattern varies by domain:

**Fundamental Analysis** — inputs reference filing directories and XBRL data by path:
```json
{
  "input": {
    "symbol": "AAPL",
    "period_end": "2024-03-30",
    "form_type": "10-Q",
    "filing_dir": "data/fundamentals/processed/filings/0000320193/000032019324000069",
    "feature_store_ref": "data/fundamentals/raw/edgar/xbrl"
  }
}
```

**Portfolio Construction** — inputs inline numeric data (returns, constraints) and reference covariance by path:
```json
{
  "input": {
    "expected_returns": {"AAPL": 0.0012, "TSLA": -0.003, ...},
    "covariance_matrix": {"tickers": [...], "ref": "data/portfolio_construction/processed/covariance/2025-01-31.npz"},
    "constraints": {"long_only": true, "max_weight": 0.06, ...},
    "objective": "max_sharpe"
  }
}
```

**Risk Management** — inputs inline portfolio holdings and reference market data by path:
```json
{
  "input": {
    "portfolio": {"portfolio_id": "...", "holdings": [{"symbol": "AIG", "weight": 0.15}, ...]},
    "stress_scenario": {"scenario_id": "stress_2018_selloff", ...},
    "market_data": {"returns": "data/risk_management/processed/returns/daily_returns.parquet"}
  }
}
```

### Ground truth provenance

All ground truth is Tier 2 (deterministic, computed from structured data sources). No expert annotation is required to use this dataset.

| Domain | Ground truth source | Method |
|---|---|---|
| FA — Normalization | XBRL + provider cross-reference | Two independent sources compared; agreement = high confidence |
| FA — Earnings quality | provider quarterly statements | Piotroski F-Score (9 components), Beneish M-Score (8 variables), accruals ratio |
| FA — Driver decomposition | provider segment endpoints + income statements | Revenue decomposed by product/geographic segment; margins by expense line |
| PC — All sub-tasks | Convex optimizer (CVXPY) | Mean-variance / Black-Litterman with identical inputs and constraints |
| RM — Risk identification | Deterministic risk computation | VaR, CVaR, HHI, factor exposures from returns + covariance |
| RM — Constraint monitoring | Deterministic constraint check | Each constraint evaluated against portfolio holdings |
| RM — Stress testing | Historical replay / factor model | Portfolio P&L computed from scenario-defined shocks |
| RM — Risk remediation | Constrained optimizer | Minimum-turnover trade list to restore compliance |

---

## Task JSON Schema

Every task file follows this schema:

```json
{
  "task_id": "string — stable unique identifier",
  "skill": "string — domain: fundamental_analysis | portfolio_construction | risk_management",
  "sub_task": "string — sub-task type (see taxonomy above)",
  "as_of_date": "string — point-in-time date (YYYY-MM-DD)",
  "difficulty": "string — easy | medium | hard",
  "input": {
    "...": "domain-specific input data and file references"
  },
  "expected_output": {
    "...": "the correct answer the agent should produce"
  },
  "verification": {
    "method": "string — scoring method name",
    "...": "method-specific thresholds and parameters"
  },
  "metadata": {
    "regime_slice": "string — market regime label",
    "...": "domain-specific metadata"
  }
}
```

### Verification methods

| Method | Used by | What it measures |
|---|---|---|
| `metric_absolute_error` | FA normalization | Per-metric absolute error; % within tolerance |
| `earnings_quality_composite` | FA earnings quality | Weighted combination of Piotroski MAE, Beneish flag accuracy, flag F1 |
| `driver_f1_and_direction` | FA driver decomposition | F1 on driver identification + direction accuracy |
| `l2_distance_and_objective` | PC unconstrained | L2 distance between weight vectors + objective value gap |
| `constraint_satisfaction_and_objective` | PC constrained | Binary constraint pass/fail + objective gap |
| `turnover_compliance_and_objective` | PC rebalancing | Turnover within limit + objective improvement |
| `parameter_match` | PC tool-use | Exact match on API call parameters |
| `infeasibility_detection` | PC infeasibility | Binary classification (feasible vs. infeasible) |
| `view_specification_and_weights` | PC Black-Litterman | View correctness + weight L2 distance |
| `ranked_list_recall` | RM risk identification | Recall@5, Precision@5, NDCG@5 on ranked risk list |
| `exact_match` | RM constraint monitoring | Per-constraint pass/fail accuracy |
| `absolute_error` | RM stress testing | Absolute error on estimated P&L |
| `constraint_satisfaction_plus_cost` | RM risk remediation | Post-trade compliance + turnover vs. optimal |

---

## Task Registry

The `task_registry/` directory provides a versioned, machine-readable index of all tasks:

```
task_registry/
├── manifest.json                      # Master manifest with counts + SHA-256 hashes
├── generate_registry.py               # Regenerate from episode files
└── domains/
    ├── fundamental_analysis.json      # 2,243 task descriptors
    ├── portfolio_construction.json    # 52 task descriptors
    └── risk_management.json           # 160 task descriptors
```

Each task descriptor in the domain files contains the task ID, domain, sub-task, difficulty, regime slice, verification spec, and file references — without the full input/output data. This is useful for sampling, stratification, and building evaluation harnesses without loading all episode data.

```python
import json

# Load all task descriptors for a domain
tasks = json.loads(open("task_registry/domains/risk_management.json").read())

# Filter by sub-task and difficulty
stress_tasks = [t for t in tasks if t["sub_task"] == "stress_testing" and t["difficulty"] == "easy"]
print(f"{len(stress_tasks)} easy stress testing tasks")

# Get the episode file path
episode_path = stress_tasks[0]["data_refs"]["episode_file"]
```

---

## Per-Domain Data Dictionaries

Detailed file schemas for every file type are documented in each domain's `DATA_DICTIONARY.md`:

- [`fundamentals/DATA_DICTIONARY.md`](fundamentals/DATA_DICTIONARY.md) — regulatory filings filings, XBRL, provider statements, segment data, normalization/quality/driver ground truth
- [`portfolio_construction/DATA_DICTIONARY.md`](portfolio_construction/DATA_DICTIONARY.md) — Prices, factors, covariance, views, constraints, optimizer ground truth
- [`risk_management/DATA_DICTIONARY.md`](risk_management/DATA_DICTIONARY.md) — Returns, factor exposures, macro, scenario portfolios, mandates, stress scenarios, risk ground truth

---

## Key Relationships

This diagram shows how data artifacts connect across the pipeline:

```
universe/sp500_security_master
    │
    ├──→ fundamentals/raw/edgar/submissions/{cik}.json     ← regulatory filings metadata
    ├──→ fundamentals/raw/edgar/xbrl/{cik}_companyfacts.json ← XBRL facts
    ├──→ fundamentals/raw/provider/income-statement/{ticker}.json ← provider statements
    ├──→ fundamentals/raw/provider/revenue-product-segmentation/{ticker}.json ← provider segments
    │
    │    ┌─────────────────────────────────────────────────────────────┐
    ├──→ │ fundamentals/processed/filing_timeline.parquet              │ ← joins regulatory filings + provider
    │    │ fundamentals/processed/xbrl_panel.parquet                   │ ← flattened XBRL
    │    │ fundamentals/processed/filings/{cik}/{accession}/mda.txt    │ ← parsed filing text
    │    └─────────────────────────────────────────────────────────────┘
    │         │
    │         ▼
    │    ┌─────────────────────────────────────────────────────────────┐
    │    │ fundamentals/ground_truth/normalization/{ticker}_{date}.json│ ← regulatory+provider cross-ref
    │    │ fundamentals/ground_truth/earnings_quality/{ticker}_{date}  │ ← Piotroski + Beneish
    │    │ fundamentals/ground_truth/driver_decomposition/{t}_{d1}_{d2}│ ← segment decomposition
    │    └─────────────────────────────────────────────────────────────┘
    │         │
    │         ▼
    │    fundamentals/episodes/layer_a/{sub_task}/task_*.json  ← 2,243 tasks
    │
    ├──→ portfolio_construction/raw/fmp_prices/{ticker}.json
    │    portfolio_construction/raw/ff/ff5_daily.parquet
    │         │
    │         ▼
    │    portfolio_construction/processed/returns.parquet
    │    portfolio_construction/processed/covariance/{date}.npz
    │    portfolio_construction/processed/views/{date}.json
    │    portfolio_construction/processed/constraints/{difficulty}.json
    │         │
    │         ▼
    │    portfolio_construction/ground_truth/{date}_{diff}_{obj}.json  ← optimizer output
    │         │
    │         ▼
    │    portfolio_construction/episodes/layer_a/{sub_task}/*.json  ← 52 tasks
    │
    └──→ risk_management/raw/factors/ff5_daily.csv
         risk_management/raw/fred/{series}.json
              │
              ▼
         risk_management/processed/returns/daily_returns.parquet
         risk_management/processed/factor_exposures.parquet
         risk_management/processed/covariance/cov_{date}.npz
              │
              ▼
         risk_management/constructed/portfolios/portfolio_*.json    ← 200 scenario portfolios
         risk_management/constructed/mandates/mandate_*.json        ← 12 mandates
         risk_management/constructed/stress_scenarios/stress_*.json ← 12 scenarios
              │
              ▼
         risk_management/ground_truth/{sub_task}/gt_*.json  ← deterministic computations
              │
              ▼
         risk_management/episodes/layer_a/{sub_task}/task_*.json  ← 160 tasks
```

---

## Evaluation Workflow

A typical evaluation loop:

1. Load tasks from `{domain}/episodes/layer_a/{sub_task}/` (or use `task_registry/` to sample)
2. For each task, extract `input` and present it to the agent
3. The agent produces a structured JSON response matching `expected_output` schema
4. Score the response using the `verification.method` and thresholds
5. Aggregate scores by domain, sub-task, difficulty, and regime slice

### Point-in-time discipline

Every task has an `as_of_date`. The agent must use only data available on or before that date. The dataset enforces this by:
- Filing dates (not period-end dates) determine when information becomes available
- Ground truth is computed from data that was public as of the filing date
- No future prices, filings, or macro data leak into task inputs

### Difficulty levels

Tasks are labeled `easy`, `medium`, or `hard` based on domain-specific criteria:
- FA: XBRL/provider agreement (easy) vs. disagreement (medium); flag count for quality tasks
- PC: Constraint complexity (easy = long-only; medium = tracking error + turnover; hard = all constraints)
- RM: Portfolio type (concentrated = easy; factor-tilted = medium; mandate-violating = hard)

### Regime slices

Tasks span four market regimes for stratified evaluation:
- `2018_selloff` (Oct–Dec 2018)
- `2020_covid` (Feb–Apr 2020)
- `2022_tightening` (Jan–Oct 2022)
- `2023_24_rebound` (Jan 2023–Dec 2024)
- `normal` (all other periods)

---

## File Formats

| Format | Used for | How to read |
|---|---|---|
| JSON | Tasks, ground truth, raw API responses, manifests | `json.load()` |
| Parquet | Tabular panels (returns, factor exposures, timelines) | `pandas.read_parquet()` or `pyarrow.parquet.read_table()` |
| NPZ | Covariance matrices | `numpy.load(path); cov = data["cov"]; symbols = data["symbols"]` |
| TXT | Parsed filing sections (MD&A, footnotes) | Plain text, UTF-8 |
| CSV | Factor returns (Ken French) | `pandas.read_csv()` |

---

## Regenerating the Dataset

Each domain has a pipeline script that downloads raw data, processes it, computes ground truth, and generates task episodes:

```bash
# Fundamental Analysis (requires FMP_API_KEY in .env)
python -m scripts.fundamental_analysis.run_pipeline

# Portfolio Construction
python -m scripts.portfolio_construction.run_pipeline

# Risk Management
python -m scripts.risk_management.run_pipeline

# Regenerate the task registry after any pipeline run
python data/task_registry/generate_registry.py
```

Individual phases can be run selectively:
```bash
# Only rebuild ground truth and tasks (skip downloads)
python -m scripts.fundamental_analysis.run_pipeline --phases earnings_quality driver_decomposition tasks

# Single ticker for debugging
python -m scripts.fundamental_analysis.run_pipeline --symbols AAPL --force
```

---

## Data Quality

Each domain includes a `validation/summary.json` with automated quality checks:

| Domain | Checks | Status |
|---|---|---|
| Fundamental Analysis | 12 checks (coverage, agreement, leakage, temporal ordering) | WARN (75.4% XBRL/provider agreement) |
| Portfolio Construction | 9 checks (prices, factors, covariance PSD, ground truth quality) | PASS |
| Risk Management | 11 checks (returns, factors, portfolios, ground truth consistency) | PASS |

---

## Limitations

- **30-ticker subset**: The evaluation universe is a stratified sample, not the full S&P 500. Results may not generalize to the full index.
- **Tier 2 ground truth only**: All labels are computed from structured data sources (XBRL, provider, optimizers). No expert-annotated (Tier 1) labels exist for earnings quality or driver decomposition. The computed labels are deterministic and reproducible but may miss qualitative nuances.
- **No execution/implementation tasks**: The fourth skill domain (Execution & Implementation) from the design doc is not yet implemented.
- **Filing text coverage**: MD&A sections are extracted for 267/287 relevant filings (93%). Footnotes coverage is lower (50/287, 17%).
- **Temporal scope**: Portfolio construction data covers 2024–2025; risk management covers 2018–2025; fundamentals cover 2017–2025. Not all domains have the same time range.

---

## Citation

If you use this dataset, please cite the FinSkillBench project and the underlying data sources:
- SEC regulatory filings (public domain)
- Financial Modeling Prep (provider) — commercial API, used under license
- Ken French Data Library (public domain)
- FRED / Federal Reserve Bank of St. Louis (public domain)
