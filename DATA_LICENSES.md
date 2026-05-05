# Data Licenses and Provenance

| Source | License | Redistribution | Notes |
|---|---|---|---|
| SEC regulatory filings (XBRL) | Public domain | Yes | US government works |
| Licensed financial data provider | Commercial | Derived features only | Raw API responses not shipped |
| Fama-French factors | Academic use | Yes | Ken French Data Library |
| FRED macroeconomic series | Public domain | Yes | Federal Reserve Economic Data |
| S&P 500 constituent list | Derived | Ticker/date pairs only | No proprietary index weights |
| Benchmark code | Apache 2.0 | Yes | This repository |
| Curated skill documents | Apache 2.0 | Yes | Human-authored for this benchmark |

## Notes

- Raw data from the licensed financial data provider is **not** included in this submission. Only derived benchmark tasks (ground-truth computations, task episodes) are shipped.
- All ground-truth values are computed outputs (optimizer solutions, accounting derivations, risk metrics), not raw provider data.
- The `data/universe/` security master contains publicly available ticker symbols, exchange listings, and CIK identifiers.
