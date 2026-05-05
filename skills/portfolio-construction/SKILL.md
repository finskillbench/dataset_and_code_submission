---
name: portfolio-construction
description: >
  Construct optimal portfolio allocations from expected returns, covariance
  matrices, and constraint sets. Includes optimizer scripts for mean-variance,
  risk parity, rebalancing, Black-Litterman, and constraint checking.
---

# Portfolio Construction Skill

## Quick Start

To solve a portfolio optimization task, call `run_skill_script` with `inject_task_fields` to pass the large data at full precision:

```
run_skill_script(
    "scripts/optimize.py",
    {
        "objective": "risk_parity",
        "constraints": {"long_only": true},
        "risk_free_rate": 0.045
    },
    inject_task_fields=["covariance_matrix", "expected_returns"]
)
```

This is the recommended approach. The `inject_task_fields` parameter tells the tool to pull `covariance_matrix` and `expected_returns` directly from the task data at full numeric precision. It automatically extracts the `matrix` (2D array) and `symbols` (ticker list) from the covariance data. You only need to specify the small fields (`objective`, `constraints`, `risk_free_rate`) in `input_data`.

Then submit the `weights` dict from the script output as your answer.

## Why inject_task_fields?

The covariance matrix is a 30×30 array (900 floating-point values). If you retrieve it via `get_task_data` and pass it through `input_data`, the values lose precision during the function call round-trip. This causes the optimizer to produce slightly wrong weights. Using `inject_task_fields` bypasses this by injecting the data directly from the task context at full precision.

---

## Scripts

### optimize.py — General Mean-Variance Optimizer

Solves mean-variance optimization for all four objectives.

**Call with inject_task_fields (recommended):**
```
run_skill_script(
    "scripts/optimize.py",
    {"objective": "max_sharpe", "constraints": {"long_only": true}, "risk_free_rate": 0.045},
    inject_task_fields=["covariance_matrix", "expected_returns"]
)
```

**Output:**
```json
{
  "status": "optimal",
  "weights": {"AIG": 0.044, "ARE": 0.027, ...},
  "expected_return": 0.000475,
  "expected_risk": 0.115827,
  "sharpe_ratio": -0.3844,
  "constraint_satisfaction": {"long_only": true, "max_weight": true}
}
```

### rebalance.py — Portfolio Rebalancing

Given a current portfolio, computes optimal new weights, trade list, and turnover. Use for rebalancing tasks.

**Call with inject_task_fields (recommended):**
```
run_skill_script(
    "scripts/rebalance.py",
    {
        "objective": "max_return",
        "risk_free_rate": 0.045,
        "constraints": {"long_only": true}
    },
    inject_task_fields=["covariance_matrix", "expected_returns", "current_portfolio"]
)
```

The script auto-reads `current_portfolio` from the task data via `inject_task_fields`. It computes the optimal new weights, then derives the trade list (signed weight changes) and one-way turnover.

**Output:**
```json
{
  "status": "optimal",
  "new_weights": {"AIG": 0.044, ...},
  "trade_list": {"AIG": -0.02, "ZBRA": 0.05, ...},
  "turnover": 0.15,
  "weights": {"AIG": 0.044, ...},
  "expected_return": 0.000475,
  "expected_risk": 0.115827
}
```

The `weights` and `new_weights` fields are identical (both contain the post-rebalance weights). `trade_list` contains only tickers with non-zero trades. `turnover` is one-way (sum of absolute changes / 2).

### black_litterman.py — Black-Litterman Model

Combines market-implied equilibrium returns with analyst views to produce posterior expected returns, then optimizes to get optimal weights. Use for Black-Litterman tasks.

**Call with inject_task_fields (recommended):**
```
run_skill_script(
    "scripts/black_litterman.py",
    {
        "risk_free_rate": 0.045,
        "analyst_views": [
            {
                "view": {
                    "type": "relative",
                    "symbols": ["AAPL", "MSFT"],
                    "return": 0.05,
                    "confidence": 0.7
                }
            }
        ]
    },
    inject_task_fields=["covariance_matrix", "market_cap_weights"]
)
```

Pass `analyst_views` in `input_data` (they're small). Use `inject_task_fields` for the covariance matrix and market-cap weights.

**View types:**
- `absolute`: Single symbol, expected return is the view return.
- `relative`: Two symbols, first outperforms second by the view return amount.

**Parameters (optional in input_data):**
- `tau`: Scalar (default 0.05). Controls how much weight the views get vs the prior. Higher tau → more weight on views.
- `risk_aversion`: Scalar (default 2.5). Used for equilibrium return computation and final optimization.

**Output:**
```json
{
  "status": "optimal",
  "posterior_returns": {"AIG": 0.026, "ARE": 0.056, ...},
  "optimal_weights": {"AIG": 0.024, "ARE": 0.024, ...}
}
```

**Important:** The script computes equilibrium returns from the covariance matrix and market-cap weights (or equal-weight if market-cap weights are empty). It does NOT use `expected_returns` as the prior — the BL model derives its own prior from market equilibrium.

### check_constraints.py — Constraint Checker

Checks whether weights satisfy constraints. Use after optimization for constrained tasks.

```
run_skill_script("scripts/check_constraints.py", {
    "weights": {"AIG": 0.044, ...},
    "constraints": {"long_only": true, "max_weight": 0.06, "min_names": 20}
})
```

Returns `{"satisfied": bool, "violations": [...], "constraint_satisfaction": {"max_weight": true, ...}}`.

### compute_weights.py — Weight Normalizer

Normalizes weights to sum to 1.0.

```
run_skill_script("scripts/compute_weights.py", {"weights": {"AIG": 0.3, "ZBRA": 0.3}})
```

---

## Objectives

- `max_sharpe`: Maximize risk-adjusted return. Concentrated toward high-return, low-correlation assets.
- `min_variance`: Minimize portfolio volatility. Diversified, favors low-vol assets.
- `max_return`: Maximize expected return. Without constraints, 100% in highest-return asset.
- `risk_parity`: Equalize risk contributions. Well-diversified, 25-35 positions, weights 0.01-0.07.

## Output Schema

For unconstrained tasks:
```json
{"weights": {"TICKER": float, ...}}
```

For constrained tasks:
```json
{"weights": {"TICKER": float, ...}, "constraint_satisfaction": {"max_weight": bool, "long_only": bool, ...}}
```

For rebalancing tasks:
```json
{"trade_list": {"TICKER": float, ...}, "new_weights": {"TICKER": float, ...}, "turnover": float}
```

For Black-Litterman tasks:
```json
{"posterior_returns": {"TICKER": float, ...}, "optimal_weights": {"TICKER": float, ...}}
```

## Common Mistakes

1. Not using `inject_task_fields` — passing the covariance matrix through `input_data` causes precision loss.
2. Passing `covariance_matrix` as `{"symbols": [...], "matrix": [...]}` instead of just the 2D array (only matters if not using `inject_task_fields`).
3. Forgetting the `symbols` field when assembling manually.
4. Wrong objective string — must be exactly: `max_sharpe`, `min_variance`, `max_return`, `risk_parity`.
5. For Black-Litterman: passing `expected_returns` as the prior. The BL model computes its own equilibrium prior from market-cap weights and the covariance matrix.
6. For rebalancing: forgetting to include `current_portfolio` in `inject_task_fields`.
