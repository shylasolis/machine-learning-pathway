# Assignment: Two-Feature Linear Regression Extension

**Based on:** `01_simple_linear.py` -> `04_compare_models_visual.py` (Module 2, `linear-regression/`)
**Format:** Take-home coding assignment
**Estimated effort:** 2-3 hours

## Context

In class, scripts 01-04 build a single-feature model:

```
expenses = w0 + w1 * bmi
```

You will extend this to a **two-feature model** that also uses `age`:

```
expenses = w0 + w1 * bmi + w2 * age
```

## Your Task

Using `src/assignment_two_feature_model.py` will be your
reference target (do not just copy it — build your own version first, then
compare), write a script that:

1. Loads `bmi`, `age`, and `expenses` from `insurance.csv`.
2. Fits the two-feature model with the **Normal Equation** (general matrix
   solve, not the 2x2 shortcut used in `02_ols_normal_equation.py`).
3. Fits the same model with **Gradient Descent**, standardizing both features
   first (like `03_gradient_descent.py` does for one feature).
4. Computes MSE, RMSE, MAE, and R^2 for both trained models.
5. Compares both two-feature models against the single-feature `bmi`-only
   baseline from `02_ols_normal_equation.py`.
6. Prints a clear comparison table and writes it to
   `reports/assignment_results.csv`.

## Deliverables

1. Your script (`.py` file).
2. The exported `assignment_results.csv`.
3. A short written memo (10-15 sentences) answering:
   - Did adding `age` improve R^2 over the `bmi`-only baseline? By how much?
   - Do the Normal Equation and Gradient Descent weights agree with each
     other? Why should they, in theory?
   - In plain language, what does the sign and size of `w2` (age) tell a
     non-technical stakeholder?
   - Is this two-feature model good enough to deploy? Why or why not?

## Grading Criteria

See `assignment_rubric.md` for the full point breakdown.

## Hints

- Reuse the standardize/mse/r2/mae helpers pattern from the Module 2 scripts
  instead of re-deriving them from scratch.
- The Normal Equation for multiple features needs a general linear solver
  (e.g., Gauss-Jordan elimination), not just the 2x2 formula from script 2.
- Standardize gradient descent features the same way script 3 standardizes
  `bmi`, then convert weights back to original units at the end.
