# Earnings Quality Formulas

## Piotroski F-Score (9 components, each 0 or 1)

### Profitability (4 signals)
1. **roa_positive**: `net_income / avg_total_assets > 0`
2. **cfo_positive**: `operating_cash_flow > 0`
3. **delta_roa_positive**: `ROA_current > ROA_prior` (improving return on assets)
4. **accruals_negative**: `operating_cash_flow > net_income` (cash earnings exceed accrual earnings)

### Leverage / Liquidity (3 signals)
5. **delta_leverage_negative**: `(TL/TA)_current < (TL/TA)_prior` (decreasing leverage)
6. **delta_current_ratio_positive**: `CR_current > CR_prior` (improving liquidity)
7. **no_dilution**: `shares_outstanding_current <= shares_outstanding_prior`

### Efficiency (2 signals)
8. **delta_gross_margin_positive**: `GM%_current > GM%_prior` (use operating_margin as proxy if COGS unavailable)
9. **delta_asset_turnover_positive**: `(revenue/avg_assets)_current > (revenue/assets)_prior`

**Score interpretation**: 0–3 = weak, 4–6 = moderate, 7–9 = strong

## Beneish M-Score (8-variable model)

```
M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI
    + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI
```

| Variable | Formula | Meaning |
|---|---|---|
| DSRI | (Receivables/Revenue)_t / (Receivables/Revenue)_t-1 | Days Sales in Receivables Index |
| GMI | GrossMargin_t-1 / GrossMargin_t | Gross Margin Index (>1 = deteriorating) |
| AQI | (1 - PPE - CA)/TA_t / (1 - PPE - CA)/TA_t-1 | Asset Quality Index |
| SGI | Revenue_t / Revenue_t-1 | Sales Growth Index |
| DEPI | DepRate_t-1 / DepRate_t | Depreciation Index |
| SGAI | (SGA/Revenue)_t / (SGA/Revenue)_t-1 | SGA Expense Index |
| TATA | (NI - CFO) / TA | Total Accruals to Total Assets |
| LVGI | (TL/TA)_t / (TL/TA)_t-1 | Leverage Index |

**Flag threshold**: M-Score > -1.78 suggests earnings manipulation risk.

## Accruals Ratio

```
accruals_ratio = (net_income - operating_cash_flow) / total_assets
```

- Positive = earnings driven by accruals (lower quality)
- Negative = earnings backed by cash (higher quality)
- Flag if |accruals_ratio| > 0.10

## Income Quality Ratio

```
income_quality_ratio = operating_cash_flow / net_income
```

- > 1.0 = high quality (cash exceeds reported earnings)
- < 1.0 = lower quality (accruals inflate earnings)
- < 0 = flag (cash flow negative while income positive, or vice versa)

## Data Requirements

For Piotroski delta components and Beneish ratios, you need **two periods** of data:
- Current period: the period being evaluated
- Prior period: the immediately preceding comparable period (same quarter YoY, or prior quarter)

Use `query_xbrl(ticker, period)` for each period. If prior period data is unavailable,
the delta components default to 0 (neutral) and Beneish ratios default to 1.0.
