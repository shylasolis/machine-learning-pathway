# Assignment Rubric: Two-Feature Linear Regression Extension

Total: 100 points

| Criterion | Points | Details |
|---|---|---|
| Data loading | 10 | Correctly loads `bmi`, `age`, `expenses` from `insurance.csv` |
| Normal Equation implementation | 20 | Correctly generalizes to 2 features (not hardcoded 2x2 shortcut); solves for w0, w1, w2 |
| Gradient Descent implementation | 20 | Standardizes both features; converges; converts weights back to original units |
| Metrics computed correctly | 15 | MSE, RMSE, MAE, R^2 computed for both methods and match expected magnitude |
| Baseline comparison | 10 | Compares against `bmi`-only baseline (R^2 ~ 0.039) |
| Comparison table export | 10 | Produces a readable printed table and a CSV export |
| Written memo | 15 | Answers all four reflection questions with correct, plain-language reasoning |
