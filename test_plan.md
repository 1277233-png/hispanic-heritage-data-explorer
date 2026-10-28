# Test plan

Record expected behavior, actual behavior, and fixes before submission.

| Test | Expected behavior | Actual result / notes |
|---|---|---|
| Verify two source values | Values and units match official World Bank series | |
| Edit then reset data | Fit changes after edit; reset restores source | |
| Duplicate years | Validation rejects duplicates | |
| Too few distinct years | Fit is blocked with clear message | |
| Equation and metrics | Equation/R²/MAE appear; independent check recorded | |
| Single/batch agreement | Same year gives same model estimate | |
| Invalid custom years | Invalid entries are reported and ignored | |
| Missing/API failure | Clear error or empty-series message; no fabricated data | |
| Comparison, marker interval, horizon, printer mode | Each control visibly changes output | |
| Exports | CSV, XLSX, PDF open and reflect current selection | |
| Deployment | Fresh viewer visit and core workflows work | |
