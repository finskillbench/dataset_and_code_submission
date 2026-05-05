---
name: fundamental-analysis
description: >
  Extract, normalize, and analyze financial metrics from XBRL data.
  Covers normalization (metric extraction), earnings quality (Piotroski/Beneish),
  and driver decomposition (revenue/margin change attribution).
---

# Fundamental Analysis Skill

## Tools Available

You have access to these tools:

- **`query_xbrl(ticker, period, metrics)`** — Query the XBRL financial data panel.
  Returns: revenue, operating_income, net_income, eps_diluted, total_assets,
  total_liabilities, stockholders_equity, operating_cash_flow,
  depreciation_amortization, shares_outstanding, eps_basic.
  - Call with just `ticker` to see all available periods.
  - Call with `ticker` + `period` to get all metrics for that period.
  - Call with `ticker` + `period` + `metrics` (list) to filter specific metrics.

- **`get_task_data(field)`** — Retrieve task input fields (symbol, period_end, etc.)

- **`run_skill_script(script_path, input_data)`** — Run curated scripts for earnings quality, decomposition, normalization, and optional XBRL querying.

## Scripts (use via `run_skill_script`)

- **`scripts/earnings_quality.py`** — Computes Piotroski F-Score, Beneish M-Score,
  accruals ratio, income quality ratio. Input: `{"current": {...}, "prior": {...}}`
  where each contains financial metrics from `query_xbrl`.

- **`scripts/driver_decomposition.py`** — Computes revenue delta, margin drivers.
  Input: `{"current": {...}, "prior": {...}, "segments": {...}}`.
  Segments are optional; without them, only aggregate and margin analysis is produced.

- **`scripts/normalize_metrics.py`** — Applies GAAP normalization adjustments.
  Input: `{"metrics": {...}, "adjustments": [...]}`.

## Procedures by Subtask

### A. Normalization

1. Call `query_xbrl(ticker=SYMBOL, period=PERIOD_END)` to get all metrics.
2. Map the returned fields to the expected output:
   - `revenue`, `operating_income`, `net_income`, `eps_diluted`
   - `total_assets`, `total_liabilities`, `stockholders_equity`
   - `operating_cash_flow`
   - `operating_margin` = operating_income / revenue
   - `ebitda` = operating_income + depreciation_amortization
3. Submit the `{"metrics": {...}}` JSON directly.

**Key**: `query_xbrl` returns values in USD (unscaled). No unit conversion needed.
EBITDA and operating_margin must be computed — they are not returned directly.

### B. Earnings Quality

1. Call `query_xbrl(ticker=SYMBOL)` to list available periods.
2. Call `query_xbrl(ticker=SYMBOL, period=CURRENT_PERIOD)` for current data.
3. Find the prior comparable period (same quarter previous year, or prior quarter).
   Call `query_xbrl(ticker=SYMBOL, period=PRIOR_PERIOD)` for prior data.
4. Either:
   - Call `run_skill_script("scripts/earnings_quality.py", {"current": {...}, "prior": {...}})`, OR
   - Compute manually using the formulas in `references/earnings_quality_formulas.md`

**Piotroski F-Score** (9 binary components):
- Profitability: roa_positive, cfo_positive, delta_roa_positive, accruals_negative
- Leverage: delta_leverage_negative, delta_current_ratio_positive, no_dilution
- Efficiency: delta_gross_margin_positive, delta_asset_turnover_positive

**Beneish M-Score**: M > -1.78 → flag as manipulation risk.

**Accruals ratio**: (net_income - operating_cash_flow) / total_assets

**Income quality ratio**: operating_cash_flow / net_income

If prior period data is unavailable, set delta components to 0 and compute
only single-period metrics (ROA, CFO, accruals, income quality).

### C. Driver Decomposition

1. Call `query_xbrl(ticker=SYMBOL, period=CURRENT_PERIOD)` for current data.
2. Call `query_xbrl(ticker=SYMBOL, period=PRIOR_PERIOD)` for prior data.
3. Compute `revenue_delta = current_revenue - prior_revenue`.
4. For `top_revenue_drivers`: The XBRL panel has aggregate data only.
   If segment data is not available from the tool, use domain knowledge about
   the company's business segments to attribute the revenue change.
   For example, for AAPL: iPhone, Mac, iPad, Services, Wearables (product);
   Americas, Europe, Greater China, Japan, Rest of Asia Pacific (geographic).
5. For `margin_drivers`: Compute operating_margin, net_margin for both periods.
   Report the delta in percentage points.
6. Optionally use `run_skill_script("scripts/driver_decomposition.py", {...})`.

## Common Mistakes

1. **Not computing EBITDA**: It's not in the XBRL panel directly. Calculate as
   `operating_income + depreciation_amortization`.
2. **Not computing operating_margin**: Calculate as `operating_income / revenue`.
3. **Wrong period for prior data**: For Piotroski, use same quarter previous year
   (e.g., 2024-03-30 → 2023-03-30). If not available, try adjacent quarters.
4. **Passing wrong parameter names to query_xbrl**: Use `ticker` (not `symbol`),
   `period` (not `period_end` or `end_date`).
5. **Forgetting to check available periods**: Call `query_xbrl(ticker=X)` first
   to see what periods exist before querying a specific one.
