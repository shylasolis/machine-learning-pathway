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

## Reference Answer Key Values (from `assignment_two_feature_model.py`)

| Model | w0 | w1 (bmi) | w2 (age) | R^2 |
|---|---|---|---|---|
| Baseline (bmi only) | 1178.18 | 394.33 | — | 0.0394 |
| Normal Equation (bmi + age) | -6437.35 | 333.39 | 241.90 | 0.1173 |
| Gradient Descent (bmi + age) | -6437.35 | 333.39 | 241.90 | 0.1173 |

Accept student values within a small tolerance (+/- 1-2% on weights and R^2)
due to floating-point and convergence differences.

## Common Deductions

- (-10) Normal Equation still hardcoded for exactly one feature.
- (-10) Gradient Descent weights not converted back from standardized units.
- (-5) No comparison against the single-feature baseline.
- (-5) Memo restates numbers without interpreting them for a non-technical
  audience.
