from __future__ import annotations

"""
Student-friendly comparison script:
- Fits the same linear model with two optimization paths
  1) Analytical OLS (Normal Equation)
  2) Iterative OLS (Gradient Descent)
- Evaluates both on train/test data
- Prints side-by-side metrics and sample predictions
- Optionally saves a plot when matplotlib is available
"""

import csv
import math
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "insurance-premium-prediction" / "insurance.csv"
PLOT_PATH = PROJECT_ROOT / "data" / "model_comparison_bmi_expenses.png"
FEATURE = "bmi"
TARGET = "expenses"

SEED = 7
TRAIN_RATIO = 0.8
LEARNING_RATE = 0.05
EPOCHS = 6000


def load_xy(csv_path: Path, feature: str, target: str) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row[feature]))
            y.append(float(row[target]))
    return x, y


def train_test_split(x: list[float], y: list[float], train_ratio: float, seed: int) -> tuple[list[float], list[float], list[float], list[float]]:
    indices = list(range(len(x)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int(len(indices) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]

    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return x_train, y_train, x_test, y_test


def mse(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    return 1.0 - (ss_res / ss_tot if ss_tot else 0.0)


def mae(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n


def fit_normal_equation_single_feature(x: list[float], y: list[float]) -> tuple[float, float]:
    # Closed-form OLS solution for one feature.
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(v * v for v in x)
    sum_xy = sum(vx * vy for vx, vy in zip(x, y))

    a = float(n)
    b = sum_x
    c = sum_x
    d = sum_x2
    det = a * d - b * c
    if det == 0:
        raise ValueError("Singular matrix in normal equation; cannot invert X^T X.")

    inv_xtx = [[d / det, -b / det], [-c / det, a / det]]
    xty = [sum_y, sum_xy]

    w0 = inv_xtx[0][0] * xty[0] + inv_xtx[0][1] * xty[1]
    w1 = inv_xtx[1][0] * xty[0] + inv_xtx[1][1] * xty[1]
    return w0, w1


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0 for _ in values], mean, 1.0
    return [(v - mean) / std for v in values], mean, std


def fit_gradient_descent_single_feature(x: list[float], y: list[float], learning_rate: float, epochs: int) -> tuple[float, float, list[tuple[int, float]]]:
    # We standardize x for stable and faster gradient descent updates.
    x_std, mean_x, std_x = standardize(x)

    w0 = 0.0
    w1 = 0.0
    n = len(x_std)
    trace: list[tuple[int, float]] = []

    for epoch in range(1, epochs + 1):
        preds = [w0 + w1 * xi for xi in x_std]
        errors = [p - yi for p, yi in zip(preds, y)]

        grad_w0 = (2.0 / n) * sum(errors)
        grad_w1 = (2.0 / n) * sum(err * xi for err, xi in zip(errors, x_std))

        w0 -= learning_rate * grad_w0
        w1 -= learning_rate * grad_w1

        if epoch == 1 or epoch % 1000 == 0 or epoch == epochs:
            loss = mse(y, [w0 + w1 * xi for xi in x_std])
            trace.append((epoch, loss))

    # Map weights back to original feature units.
    slope = w1 / std_x
    intercept = w0 - (w1 * mean_x / std_x)
    return intercept, slope, trace


def predict(x: list[float], w0: float, w1: float) -> list[float]:
    return [w0 + w1 * xi for xi in x]


def print_pipeline_outline() -> None:
    """Print an ordered lesson outline for script 4."""
    print("Pipeline outline (script 4):")
    print("  1) Read data: load bmi and expenses.")
    print("  2) Shuffle rows: randomize order so train/test split is fair and not ordered-biased.")
    print("  3) Split data: create train (fit) and test (unseen evaluation).")
    print("  4) Train method A: Normal Equation learns w0/w1 in one direct solve.")
    print("  5) Train method B: Gradient Descent standardizes feature, then learns w0/w1 iteratively.")
    print("  6) Evaluate both: compare MSE, RMSE, MAE, R^2, and train-test gap.")
    print("  7) Interpret for business: convert metrics to plain-language impact.")
    print("  8) If results do not fit needs: tune hyperparameters, add stronger features, then retrain and re-evaluate.")
    print()


def print_split_explainer(train_size: int, test_size: int) -> None:
    """Explain train/test and how it relates to train/val/test workflows."""
    total = train_size + test_size
    train_pct = (train_size / total) * 100 if total else 0.0
    test_pct = (test_size / total) * 100 if total else 0.0

    print("What 'train/test split' means in this script:")
    print(f"  train set ({train_size} rows, {train_pct:.1f}%): used to fit model parameters")
    print(f"  test set  ({test_size} rows, {test_pct:.1f}%): used only for final evaluation")
    print()
    print("How this compares to train/val/test in real workflows:")
    print("  train/val/test is common when tuning many hyperparameters")
    print("  validation set helps pick settings (like learning rate) without touching test")
    print("  test set is kept as the final unbiased score")
    print("  here we use train/test only to keep the lesson focused and simple")
    print()


def print_normal_equation_explainer() -> None:
    """Clarify what the normal equation is mathematically."""
    print("Is the Normal Equation matrix multiplication?")
    print("  yes, mostly. It combines matrix multiplication and a matrix inverse:")
    print("    theta = (X^T X)^(-1) X^T y")
    print("  in this script, we compute the same result with scalar sums for 1 feature")
    print("  that is equivalent to the matrix formula for this simple case")
    print()


def print_loss_checkpoint_explainer(loss_trace: list[tuple[int, float]]) -> None:
    """Explain how to interpret loss checkpoints and why practitioners track them."""
    mostly_converged = False
    if len(loss_trace) >= 2:
        first_loss = loss_trace[0][1]
        last_loss = loss_trace[-1][1]
        rel_change = abs(last_loss - first_loss) / (abs(first_loss) if first_loss else 1.0)
        mostly_converged = rel_change < 0.01

    print("How to interpret Gradient Descent loss checkpoints:")
    print("  each checkpoint shows MSE at a certain epoch")
    print("  if MSE drops quickly, learning is working")
    print("  if MSE plateaus, training has mostly converged")
    print("  'mostly converged' means later epochs barely change loss and weights")
    print("  if MSE rises/oscillates, learning rate may be too high or setup needs tuning")
    if mostly_converged:
        print("  in this run, checkpoints suggest we are mostly converged")
    else:
        print("  in this run, checkpoints suggest there may still be room to improve")
    print()
    print("If setup needs tuning, check these first:")
    print("  1) learning rate (too high = unstable, too low = slow)")
    print("  2) feature scaling (especially important for gradient descent)")
    print("  3) number of epochs (stop too early vs train long enough)")
    print("  4) feature quality (single-feature model may underfit)")
    print("  5) data issues (outliers, leakage, wrong units, missing values)")
    print()
    print("Do people do this in practice?")
    print("  yes. Monitoring training curves is standard for iterative optimizers")
    print("  it helps detect convergence, instability, and wasted extra epochs")
    print()


def print_mse_scale_explainer(y_train: list[float], train_mse: float, test_mse: float, model_name: str) -> None:
    """Explain why MSE has no fixed max and how to judge it for this dataset."""
    mean_target = sum(y_train) / len(y_train)
    baseline_train_pred = [mean_target for _ in y_train]
    baseline_train_mse = mse(y_train, baseline_train_pred)

    rmse_train = math.sqrt(train_mse)
    rmse_test = math.sqrt(test_mse)
    baseline_rmse = math.sqrt(baseline_train_mse)

    print(f"MSE interpretation for {model_name}:")
    print("  MSE has no universal maximum; scale depends on target units and spread")
    print("  evaluate it relative to a baseline (predicting mean expenses)")
    print(f"  baseline train MSE (predict mean): {baseline_train_mse:.2f}")
    print(f"  model train MSE:                 {train_mse:.2f}")
    print(f"  model test  MSE:                 {test_mse:.2f}")
    print(f"  model train RMSE:                {rmse_train:.2f}")
    print(f"  model test  RMSE:                {rmse_test:.2f}")
    print(f"  baseline RMSE:                   {baseline_rmse:.2f}")
    print("  lower than baseline means the model learned signal beyond a naive guess")
    print()


def print_real_world_evaluation_guide() -> None:
    """Summarize how regression is usually evaluated in production workflows."""
    print("Real-world regression evaluation guide:")
    print("  1) Split strategy: use train/validation/test or cross-validation")
    print("  2) Tune hyperparameters on validation only (never tune on test)")
    print("  3) Report multiple metrics (RMSE, MAE, R^2) with confidence intervals when possible")
    print("  4) Check residual behavior: bias, outliers, heteroscedasticity")
    print("  5) Check subgroup performance for fairness and business risk")
    print("  6) Compare against simple baselines and current business process")
    print()


def print_stakeholder_summary_from_results(test_r2: float, test_rmse: float, target_mean: float) -> None:
    """Provide a plain-language summary suitable for meetings."""
    rel_error = (test_rmse / target_mean) * 100 if target_mean else 0.0
    print("Stakeholder-ready summary for this current model:")
    print(f"  this model captures only about {test_r2 * 100:.2f}% of the pattern in costs")
    print("  in plain language: most cost differences between people are still not explained by this model")
    print(f"  typical prediction error is about ${test_rmse:.2f} RMSE (~{rel_error:.1f}% of mean expense)")
    print("  takeaway: useful as a teaching baseline, not strong enough for decision-grade forecasting")
    print("  recommended next step: add key features (smoker, age, region, interactions) and re-evaluate")
    print()


def print_stakeholder_talking_points(test_r2: float, test_rmse: float, target_mean: float) -> None:
    """Print three ready-to-read bullets for business meetings."""
    rel_error = (test_rmse / target_mean) * 100 if target_mean else 0.0
    pattern_pct = test_r2 * 100
    print("Three meeting bullets:")
    print(f"  1) Current performance: this model captures about {pattern_pct:.2f}% of the cost pattern.")
    print(f"  2) Practical impact: typical miss is about ${test_rmse:.2f} (~{rel_error:.1f}% of average cost).")
    print("  3) Action plan: treat this as a baseline and add stronger features before business use.")
    print()


def print_generalization_note(train_mse: float, test_mse: float) -> None:
    """Provide a simple interpretation of train-vs-test behavior."""
    gap = test_mse - train_mse
    print("Train vs test interpretation:")
    print(f"  train MSE = {train_mse:.2f}")
    print(f"  test  MSE = {test_mse:.2f}")
    print(f"  gap       = {gap:.2f} (test - train)")
    if gap > 0:
        print("  test error is higher, which is normal; the key is whether the gap is acceptably small")
    elif gap < 0:
        print("  test error is lower here due to sample variation; this can happen on random splits")
    else:
        print("  train and test errors are equal on this split")
    print()


def print_sample_predictions(x: list[float], y_true: list[float], y_pred_a: list[float], y_pred_b: list[float], rows: int = 8) -> None:
    print("\nSampled data points from test set [text-based table preview]:")
    print("  bmi     actual      normal_eq     grad_desc")
    points = sorted(zip(x, y_true, y_pred_a, y_pred_b), key=lambda p: p[0])
    step = max(1, len(points) // rows)
    for xi, yi, pa, pb in points[::step][:rows]:
        print(f"  {xi:5.1f}   {yi:9.2f}   {pa:11.2f}   {pb:11.2f}")


def maybe_save_plot(x_train: list[float], y_train: list[float], x_test: list[float], y_test: list[float], w_ne: tuple[float, float], w_gd: tuple[float, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nPlot skipped: matplotlib is not installed in this environment.")
        print("Install with: pip install matplotlib")
        return

    x_line = [min(x_train + x_test), max(x_train + x_test)]
    y_line_ne = predict(x_line, w_ne[0], w_ne[1])
    y_line_gd = predict(x_line, w_gd[0], w_gd[1])

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.35, label="train points")
    plt.scatter(x_test, y_test, alpha=0.45, label="test points")
    plt.plot(x_line, y_line_ne, linewidth=2.5, label="Normal Equation line")
    plt.plot(x_line, y_line_gd, linewidth=2.5, linestyle="--", label="Gradient Descent line")
    plt.title("Insurance Data: BMI vs Expenses")
    plt.xlabel("BMI")
    plt.ylabel("Expenses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    print(f"\nSaved comparison plot to: {PLOT_PATH}")


def main() -> None:
    print_pipeline_outline()

    x, y = load_xy(CSV_PATH, FEATURE, TARGET)
    x_train, y_train, x_test, y_test = train_test_split(x, y, TRAIN_RATIO, SEED)

    w0_ne, w1_ne = fit_normal_equation_single_feature(x_train, y_train)
    w0_gd, w1_gd, gd_trace = fit_gradient_descent_single_feature(
        x_train,
        y_train,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )

    pred_train_ne = predict(x_train, w0_ne, w1_ne)
    pred_test_ne = predict(x_test, w0_ne, w1_ne)
    pred_train_gd = predict(x_train, w0_gd, w1_gd)
    pred_test_gd = predict(x_test, w0_gd, w1_gd)

    train_mse_ne = mse(y_train, pred_train_ne)
    test_mse_ne = mse(y_test, pred_test_ne)
    train_r2_ne = r2_score(y_train, pred_train_ne)
    test_r2_ne = r2_score(y_test, pred_test_ne)

    train_mse_gd = mse(y_train, pred_train_gd)
    test_mse_gd = mse(y_test, pred_test_gd)
    train_r2_gd = r2_score(y_train, pred_train_gd)
    test_r2_gd = r2_score(y_test, pred_test_gd)

    print("Model comparison on train/test split")
    print(f"Dataset: {CSV_PATH}")
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
    print()

    print_split_explainer(len(x_train), len(x_test))
    print_normal_equation_explainer()

    print("Normal Equation (Analytical)")
    print(f"  Equation: expenses = {w0_ne:.4f} + {w1_ne:.4f} * bmi")
    print(f"  Train MSE: {train_mse_ne:.2f}")
    print(f"  Test  MSE: {test_mse_ne:.2f}")
    print(f"  Train MAE: {mae(y_train, pred_train_ne):.2f}")
    print(f"  Test  MAE: {mae(y_test, pred_test_ne):.2f}")
    print(f"  Train R^2: {train_r2_ne:.4f}")
    print(f"  Test  R^2: {test_r2_ne:.4f}")
    print()
    print_generalization_note(train_mse_ne, test_mse_ne)
    print_mse_scale_explainer(y_train, train_mse_ne, test_mse_ne, "Normal Equation")

    print("Gradient Descent (Iterative)")
    print(f"  Equation: expenses = {w0_gd:.4f} + {w1_gd:.4f} * bmi")
    print(f"  Train MSE: {train_mse_gd:.2f}")
    print(f"  Test  MSE: {test_mse_gd:.2f}")
    print(f"  Train MAE: {mae(y_train, pred_train_gd):.2f}")
    print(f"  Test  MAE: {mae(y_test, pred_test_gd):.2f}")
    print(f"  Train R^2: {train_r2_gd:.4f}")
    print(f"  Test  R^2: {test_r2_gd:.4f}")
    print()
    print_generalization_note(train_mse_gd, test_mse_gd)
    print_mse_scale_explainer(y_train, train_mse_gd, test_mse_gd, "Gradient Descent")

    print("Should test error be lower than train error?")
    print("  usually no. Train error is often lower because the model fit that data directly.")
    print("  if test is slightly lower, that can happen by random split variation.")
    print("  worry only when train is much better than test (overfitting warning).")
    print()

    print("\nGradient Descent loss checkpoints:")
    for epoch, loss in gd_trace:
        print(f"  epoch={epoch:5d} -> mse={loss:.2f}")
    print()
    print_loss_checkpoint_explainer(gd_trace)

    print_real_world_evaluation_guide()
    test_rmse = math.sqrt(test_mse_gd)
    target_mean = sum(y_test) / len(y_test)
    print_stakeholder_summary_from_results(test_r2_gd, test_rmse, target_mean)
    print_stakeholder_talking_points(test_r2_gd, test_rmse, target_mean)

    print("\nTerminology note for students:")
    print("  ASCII = American Standard Code for Information Interchange.")
    print("  In this project, ASCII visuals are text-only charts made from keyboard characters.")
    print("  Sampled data points are selected rows shown as a compact table, not a plotted image.")

    print_sample_predictions(x_test, y_test, pred_test_ne, pred_test_gd)
    maybe_save_plot(x_train, y_train, x_test, y_test, (w0_ne, w1_ne), (w0_gd, w1_gd))


if __name__ == "__main__":
    main()
