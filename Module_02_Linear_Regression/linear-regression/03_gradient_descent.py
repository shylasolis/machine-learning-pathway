from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "insurance-premium-prediction" / "insurance.csv"
FEATURE = "bmi"
TARGET = "expenses"

LEARNING_RATE = 0.05
EPOCHS = 10000
LOG_EVERY = 1000


def load_xy(csv_path: Path, feature: str, target: str) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row[feature]))
            y.append(float(row[target]))
    return x, y


def standardize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0 for _ in values], mean, 1.0
    return [(v - mean) / std for v in values], mean, std


def mse(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    return 1.0 - (ss_res / ss_tot if ss_tot else 0.0)


def print_loss_bars(loss_points: list[tuple[int, float]]) -> None:
    """Render a tiny bar chart of losses in plain text."""
    if not loss_points:
        return

    losses = [loss for _, loss in loss_points]
    top = max(losses)
    bottom = min(losses)
    spread = max(top - bottom, 1e-9)

    print("\nLoss trend (smaller is better):")
    for epoch, loss in loss_points:
        # Shorter bars mean lower loss, so students see descent visually.
        normalized = (loss - bottom) / spread
        bar_len = int((1.0 - normalized) * 28)
        bar = "#" * max(1, bar_len)
        print(f"  epoch={epoch:5d} | {bar:<28} | mse={loss:.2f}")


def print_transition_explainer() -> None:
    """Explain why students move from script 2 to script 3."""
    print("How script 3 differs from script 2:")
    print("  script 2 (Normal Equation): finds weights in one closed-form calculation")
    print("  script 3 (Gradient Descent): finds weights step-by-step by reducing error")
    print("  reason to learn script 3: it scales better when datasets/features become very large")
    print()
    print("What counts as 'very large' (rule of thumb):")
    print("  smaller problems: use Normal Equation when feature count is low to moderate")
    print("    for class intuition: roughly up to a few hundred features is usually comfortable")
    print("  larger problems: prefer Gradient Descent when feature count grows into the high hundreds/thousands")
    print("    and/or when row count is very large and repeated matrix inversion becomes expensive")
    print("  this happens because Normal Equation needs an inverse of (X^T X), which gets costly as")
    print("  feature count grows")
    print()
    print("Quick classroom decision guide:")
    print("  1) one or few features (like this BMI example) -> Normal Equation is great")
    print("  2) many engineered/encoded features -> Gradient Descent is usually safer")
    print("  3) if inversion is slow or memory-heavy -> switch to Gradient Descent")
    print()


def print_epoch_learning_rate_explainer() -> None:
    """Explain epoch and learning rate in plain language for students."""
    print("Key training terms:")
    print("  epoch: one full pass through all training rows")
    print("  in this script: we use full-batch updates, so weights update once per epoch")
    print("  learning_rate: step size for each weight update")
    print()
    print("How they affect the model:")
    print("  very small learning_rate -> stable but slow learning")
    print("  very large learning_rate -> can overshoot and fail to converge")
    print("  more epochs -> more opportunities to improve until loss plateaus")
    print()
    print("Rule of thumb:")
    print("  start with rates like 0.1, 0.05, 0.01, 0.001 and monitor loss")
    print("  if loss oscillates/diverges, lower learning_rate")
    print("  if loss decreases too slowly, try a slightly larger learning_rate")
    print("  stop when loss has clearly flattened")
    print()
    print("Classroom note: interactive tuning experiments are saved for the Jupyter notebook phase.")
    print()


def print_interpretability_summary(slope: float, intercept: float, model_mse: float, model_r2: float) -> None:
    """Print a student-friendly interpretation of model outputs."""
    direction = "increase" if slope >= 0 else "decrease"
    print("Interpretability guide:")
    print("  intercept (w0): baseline prediction when bmi=0")
    print("  slope (w1): expected change in predicted expenses for each +1 BMI")
    print("  MSE: average squared prediction error (lower is better)")
    print("  R^2: percent of variation explained by this model (closer to 1 is better)")
    print()
    print("Student interpretation summary:")
    print(f"  Because w1 is {slope:.2f}, predicted expenses {direction} by about ${abs(slope):.2f} per +1 BMI.")
    print(f"  R^2={model_r2:.4f} means this single-feature model explains about {model_r2 * 100:.2f}% of expense variation.")
    print(f"  MSE={model_mse:.2f} is in squared-dollar units, so use it mainly to compare models.")


def print_pipeline_outline() -> None:
    """Print an ordered lesson outline for script 3."""
    print("Pipeline outline (script 3):")
    print("  1) Read data: load bmi (feature) and expenses (target).")
    print("  2) Standardize feature: center/scale bmi so gradient descent is stable and faster.")
    print("  3) Train iteratively: run epochs, update w0/w1 to reduce loss.")
    print("  4) Learn parameters: convert trained weights back to original bmi units.")
    print("  5) Evaluate fit: compute MSE and R^2 and inspect loss checkpoints.")
    print("  6) If fit is not good enough: tune learning rate/epochs, improve features, and compare on train/test.")
    print()


def main() -> None:
    print_pipeline_outline()

    # ----------------------------------------------------------------------
    # Gradient Descent intuition:
    # 1) start with random/zero weights
    # 2) compute prediction errors
    # 3) move weights opposite the gradient
    # 4) repeat until loss stabilizes
    # ----------------------------------------------------------------------
    x_raw, y = load_xy(CSV_PATH, FEATURE, TARGET)
    x, x_mean, x_std = standardize(x_raw)

    w0 = 0.0
    w1 = 0.0
    n = len(x)

    print("Method: Iterative OLS via Gradient Descent")
    print(f"Dataset: {CSV_PATH}")
    print(f"Model: {TARGET} = w0 + w1*{FEATURE}")
    print(f"Hyperparameters: learning_rate={LEARNING_RATE}, epochs={EPOCHS}")
    print()
    print_transition_explainer()
    print_epoch_learning_rate_explainer()
    print("What you are looking at during training:")
    print("  Each epoch updates w0 and w1 to reduce MSE.")
    print("  If epoch MSE stops changing much, the model has likely converged.")

    loss_points: list[tuple[int, float]] = []

    for epoch in range(1, EPOCHS + 1):
        preds_std = [w0 + w1 * xi for xi in x]
        errors = [p - yi for p, yi in zip(preds_std, y)]

        grad_w0 = (2.0 / n) * sum(errors)
        grad_w1 = (2.0 / n) * sum(err * xi for err, xi in zip(errors, x))

        w0 -= LEARNING_RATE * grad_w0
        w1 -= LEARNING_RATE * grad_w1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            current_loss = mse(y, [w0 + w1 * xi for xi in x])
            print(f"epoch={epoch:5d} mse={current_loss:.2f}")
            loss_points.append((epoch, current_loss))

    # Convert coefficients back from standardized-feature space to original x space.
    slope_original = w1 / x_std
    intercept_original = w0 - (w1 * x_mean / x_std)

    preds = [intercept_original + slope_original * xi for xi in x_raw]
    model_mse = mse(y, preds)
    model_r2 = r2_score(y, preds)

    print()
    print("Final coefficients in original feature units:")
    print(f"w0 (intercept): {intercept_original:.6f}")
    print(f"w1 (slope):     {slope_original:.6f}")
    print(f"MSE:            {model_mse:.2f}")
    print(f"R^2:            {model_r2:.4f}")
    print()
    print_interpretability_summary(slope_original, intercept_original, model_mse, model_r2)
    print_loss_bars(loss_points)


if __name__ == "__main__":
    main()
