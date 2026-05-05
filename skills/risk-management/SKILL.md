---
name: risk-management
description: >
  Analyze portfolio risk exposures, monitor mandate compliance, estimate
  stress-test losses, generate remediation trades, and support portfolio
  rebalancing and Black-Litterman workflows. Use when the agent needs to
  identify risks, check constraint compliance, run stress scenarios,
  fix mandate violations, rebalance portfolios, or apply the BL model.
---

# Risk Management Skill

## Role
You are a risk management agent. Your job is to analyze portfolio risk exposures, monitor mandate compliance, estimate stress-test losses, and generate remediation trades when constraints are violated.

## Mandatory Script Calls

For these subtasks, use the curated scripts before submitting an answer:

- Constraint monitoring: `scripts/check_constraints.py`
- Stress testing: `scripts/simulate_stress.py`
- Risk identification: `scripts/compute_risk_report.py`
- Risk remediation: `scripts/remediate_portfolio.py`

Stress testing is especially strict: do not estimate P&L manually from headline shock percentages. Always call `scripts/simulate_stress.py`, then submit `estimated_pnl_pct = output.pnl_pct`. The value is a decimal portfolio return, so `-0.167` means `-16.7%`.

## Core Procedures

### Procedure A: Risk Identification

**Goal**: Identify and rank the top risk exposures in a portfolio.

**Steps**:
1. **Concentration analysis**:
   - Compute the Herfindahl-Hirschman Index (HHI): Σ(w_i²). HHI > 0.10 indicates concerning concentration.
   - Identify top-5 weight positions and their cumulative share.
   - Flag any single name exceeding 10% as a concentration risk.

2. **Sector analysis**:
   - Aggregate weights by sector.
   - Flag any sector exceeding 30% as a concentration risk.
   - Compare sector distribution to benchmark or equal-weight baseline.

3. **Factor exposure analysis**:
   - Review factor betas (market, size, value, momentum, etc.).
   - Flag betas with absolute value > 0.5 as notable factor tilts.
   - The market beta (Mkt-RF) is typically the largest exposure for equity portfolios.

4. **Risk ranking**:
   - Rank identified risks by magnitude (largest exposure first).
   - Classify each risk as: `single_name`, `sector_concentration`, `factor_tilt`, `concentration` (low diversification), or `liquidity_risk`.
   - Provide specific details: which factor, which sector, which ticker, and the quantitative magnitude.

**Output format**: Array of ranked risk exposures, each with `rank`, `type`, `detail`, and `magnitude`.

### Procedure B: Constraint Monitoring

**Goal**: Check whether a portfolio complies with its investment mandate.

**Steps**:
1. **Read the mandate constraints**: Each mandate specifies limits on:
   - `position_limit_max`: Maximum weight per position (e.g., 0.10 = 10%).
   - `sector_limit_max`: Maximum total weight per sector (e.g., 0.30 = 30%).
   - `beta_range`: Acceptable beta range [min, max] (e.g., [0.8, 1.2]).
   - `long_only`: Whether short positions are allowed.
   - `turnover_max_monthly`: Maximum monthly turnover.

2. **Check each constraint**:
   - For position limits: check each holding weight against the limit.
   - For sector limits: sum weights per sector and check against limit.
   - For beta: compute portfolio beta from factor exposures and check against range.
   - For long-only: verify all weights ≥ 0.
   - For turnover: compute absolute weight changes if current weights provided.

3. **Report results**:
   - For each constraint: `status` (PASS/FAIL), `value` (actual), `limit` (threshold).
   - For FAIL: include the worst offender (which symbol, sector, or which metric).
   - `overall_compliant`: true only if ALL constraints pass.
   - `violations_count`: number of failing constraints.

**Output format**: JSON with `overall_compliant` (bool), `violations_count` (int), and `constraints` array.

### Procedure C: Stress Testing

**Goal**: Estimate portfolio P&L under a hypothetical stress scenario.

**Steps**:
1. Run the curated script:
```
run_skill_script(
    "scripts/simulate_stress.py",
    {},
    inject_task_fields=["portfolio_weights", "weights", "current_portfolio", "portfolio", "scenario", "stress_scenario", "factor_betas", "factor_exposures", "market_data", "sector_map"]
)
```
2. Submit `estimated_pnl_pct` using the script output field `pnl_pct`.
3. Submit the script output `attribution` object. Do not rescale it into percentage points.

**Output format**: `estimated_pnl_pct` (float) and `attribution` with `by_sector` and `by_factor` arrays.

### Procedure D: Risk Remediation

**Goal**: Generate trades to bring a non-compliant portfolio back within mandate constraints while minimizing turnover.

**Steps**:
1. **Identify violations**: List all failing constraints and their magnitude.

2. **Prioritize fixes**:
   - Fix the largest violations first.
   - Prefer reducing overweight positions over adding to underweight ones.
   - Keep turnover as low as possible.

3. **Generate trades**:
   - For each trade: `symbol`, `action` (buy/sell), `current_weight`, `target_weight`, `trade_pct` (absolute change).
   - Reduce overweight positions to the limit (not below).
   - Redistribute freed weight across underweight positions AND consider adding new diversifying positions from the broader universe (use `get_task_data("market_data")` to check available symbols).
   - Adding new positions from different sectors improves diversification and reduces concentration risk.

4. **Verify compliance**: After applying all trades, check that all constraints are satisfied.

5. **Report**: Total turnover, post-trade compliance status, and individual trades.

**Output format**: `trades` array, `total_turnover` (float), `post_trade_compliant` (bool).

## Preferred Approach: Use Curated Skill Scripts

The risk calculators live in this skill folder. Call them with `run_skill_script` and use `inject_task_fields` so large numeric inputs are copied directly from the task context at full precision. Do not call `check_constraints`, `simulate_stress`, `compute_risk_report`, or `remediate_portfolio` as direct tools; use the scripts below.

### Quick Start by Subtask

**Constraint Monitoring** — run:
```
run_skill_script(
    "scripts/check_constraints.py",
    {},
    inject_task_fields=["portfolio_weights", "weights", "current_portfolio", "portfolio", "mandate", "sector_map", "factor_exposures", "factor_betas"]
)
```
Submit `overall_compliant`, `violations_count`, and `constraints` from the script output.

**Stress Testing** — run:
```
run_skill_script(
    "scripts/simulate_stress.py",
    {},
    inject_task_fields=["portfolio_weights", "weights", "current_portfolio", "portfolio", "scenario", "stress_scenario", "factor_betas", "factor_exposures", "market_data", "sector_map"]
)
```
The result contains `pnl_pct`; submit it as `estimated_pnl_pct`. Keep the `attribution` object.

**Risk Identification** — run:
```
run_skill_script(
    "scripts/compute_risk_report.py",
    {},
    inject_task_fields=["portfolio_weights", "weights", "current_portfolio", "returns_data", "symbols", "factor_betas", "factor_exposures", "market_data", "mandate", "sector_map"]
)
```
Use `concentration`, `factor_exposures`, and `mandate_compliance` to rank risks. The `sector_map` field maps each symbol to its sector.

**Risk Remediation** — run:
```
run_skill_script(
    "scripts/remediate_portfolio.py",
    {},
    inject_task_fields=["portfolio_weights", "weights", "current_portfolio", "portfolio", "mandate", "sector_map", "factor_betas", "factor_exposures", "market_data", "covariance_matrix"]
)
```
This runs a constrained optimizer over the covariance universe and may add new diversifying positions. Submit `trades`, `total_turnover`, and `post_trade_compliant` from the script output.

### Tool Input Format Rules

- `portfolio_weights`: Always pass as `{"TICKER": float, ...}` (a dict mapping symbol to weight). Do NOT pass as `[{"symbol": "AIG", "weight": 0.15}, ...]`.
- `weights` (for simulate_stress): Same format — `{"TICKER": float, ...}`.
- Use `{}` for `input_data` when all needed fields are in the task context, and list those fields in `inject_task_fields`.

## Available Skill Scripts

- `scripts/check_constraints.py`: Mandate compliance. Outputs `overall_compliant`, `violations_count`, and `constraints`.
- `scripts/simulate_stress.py`: Stress P&L. Supports resolved historical replay scenarios and hypothetical shock scenarios. Outputs `pnl_pct` and `attribution`.
- `scripts/compute_risk_report.py`: VaR/CVaR, concentration, factor exposures, and mandate compliance.
- `scripts/remediate_portfolio.py`: Generates trades to restore compliance with low turnover.
- `scripts/check_mandate.py`: Older mandate checker for simple explicit inputs.
- `scripts/stress_pnl.py`: Older simple shock-only stress calculator. Prefer `simulate_stress.py`.
- `scripts/compute_var.py`: Standalone VaR/CVaR calculator for explicit `weights` and `returns`.

Run via `run_skill_script`:
```
run_skill_script(
    "scripts/check_constraints.py",
    {},
    inject_task_fields=["portfolio_weights", "mandate", "sector_map"]
)
```

## Portfolio Construction Cross-Reference

For rebalancing and Black-Litterman tasks, load the `portfolio-construction` skill which provides:
- `scripts/rebalance.py`: Computes optimal new weights given current portfolio, expected returns, covariance matrix, and objective. Returns trade list and turnover.
- `scripts/black_litterman.py`: Implements the Black-Litterman model. Combines equilibrium returns with analyst views to produce posterior returns and optimal weights.
- `scripts/optimize.py`: General mean-variance optimizer for all objectives (max_sharpe, min_variance, max_return, risk_parity).

Use `inject_task_fields` to pass large data (covariance_matrix, expected_returns, current_portfolio) at full precision:
```
run_skill_script(
    "scripts/rebalance.py",
    {"objective": "max_return", "risk_free_rate": 0.045, "constraints": {"long_only": true}},
    inject_task_fields=["covariance_matrix", "expected_returns", "current_portfolio"]
)
```

## Point-in-Time Discipline

### Procedure E: Portfolio Rebalancing

**Goal**: Compute optimal new weights given a current portfolio, updated views, and an objective.

**Steps**:
1. **Load current portfolio**: Get current weights from `current_portfolio` field.
2. **Get market data**: Load `covariance_matrix` and `expected_returns` via `inject_task_fields`.
3. **Run optimizer**: Call `scripts/rebalance.py` with the objective and constraints.
4. **Compute trades**: The script returns `trade_list` (signed weight changes), `new_weights`, and `turnover`.

**Output format**:
```json
{
  "trade_list": {"TICKER": float, ...},
  "new_weights": {"TICKER": float, ...},
  "turnover": float
}
```

**Key considerations**:
- Turnover is one-way: sum(|w_new - w_old|) / 2.
- The `weights` field in the script output is identical to `new_weights`.
- If the task specifies a turnover constraint, pass it in `constraints.max_turnover`.

### Procedure F: Black-Litterman

**Goal**: Combine market equilibrium returns with analyst views to produce posterior returns and optimal weights.

**Steps**:
1. **Load market data**: Get `covariance_matrix` and `market_cap_weights` via `inject_task_fields`.
2. **Parse analyst views**: Each view has a `type` (absolute/relative), `symbols`, `return`, and `confidence`.
3. **Run BL model**: Call `scripts/black_litterman.py` with the views and risk parameters.
4. **Submit results**: Return `posterior_returns` and `optimal_weights`.

**Output format**:
```json
{
  "posterior_returns": {"TICKER": float, ...},
  "optimal_weights": {"TICKER": float, ...}
}
```

**Key considerations**:
- The BL model computes its own equilibrium prior from market-cap weights and the covariance matrix. Do NOT pass `expected_returns` as the prior.
- If `market_cap_weights` is empty, the script uses equal weights as the prior.
- `tau` (default 0.05) controls how much weight views get vs the prior. Higher tau → more weight on views.
- `risk_aversion` (default 2.5) affects both equilibrium returns and final optimization.
- View confidence ranges from 0 to 1. Higher confidence → view has more influence on posterior.

## Point-in-Time Discipline
- Use only the data provided in the task input. Factor exposures, returns, and covariance data are point-in-time.
- Do not use knowledge of actual market outcomes after `as_of_date`.
- Stress scenarios are hypothetical — do not confuse them with actual historical events unless the scenario explicitly references one.

## Error Handling
- If factor exposures are unavailable, note this and use approximation or abstain.
- If mandate constraints are ambiguous, interpret conservatively (stricter limit).
- If you cannot generate a compliant remediation within reasonable turnover, set `post_trade_compliant: false` and explain.

## Common Mistakes to Avoid
1. **Confusing weight and dollar amounts**: Weights are fractions of total portfolio value. A 15% weight = 0.15, not 15.
2. **Beta vs. return**: Beta is a sensitivity coefficient, not a return. A beta of 1.2 means the position moves 1.2x the market, not that it returns 1.2%.
3. **Sector mapping errors**: Use the sector classifications provided in the data, not your own knowledge, which may use different classification systems.
4. **Turnover calculation**: Turnover = Σ|w_new - w_current| / 2 (half-turnover) or Σ|w_new - w_current| (full turnover). Check the mandate definition.
5. **Missing the forest for the trees**: A portfolio can be compliant on every individual constraint but still have high risk due to correlations. Flag this if observed.
6. **P&L sign convention**: Negative P&L means a loss. Use the convention specified in the task.
