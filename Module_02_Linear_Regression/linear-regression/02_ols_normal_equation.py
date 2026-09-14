from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "insurance-premium-prediction" / "insurance.csv"
FEATURE = "bmi"
TARGET = "expenses"


def load_xy(csv_path: Path, feature: str, target: str) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row[feature]))
            y.append(float(row[target]))
    return x, y


def fit_normal_equation_single_feature(x: list[float], y: list[float]) -> tuple[float, float]:
    """Fit y = w0 + w1*x using theta=(X^T X)^-1 X^T y for one feature."""
    # ----------------------------------------------------------------------
    # We avoid external libraries so students can map each scalar below to
    # the matrix expression for OLS:
    #   theta = (X^T X)^(-1) X^T y
    # where theta = [w0, w1].
    # ----------------------------------------------------------------------
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(v * v for v in x)
    sum_xy = sum(vx * vy for vx, vy in zip(x, y))

    # X^T X = [[n, sum_x], [sum_x, sum_x2]]
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


def mse(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    return 1.0 - (ss_res / ss_tot if ss_tot else 0.0)


def print_prediction_table(x: list[float], y: list[float], y_pred: list[float], rows: int = 10) -> None:
    """Print a compact table to visualize prediction quality."""
    print("\nSample predictions (sorted by feature):")
    print("  bmi     actual_expense   predicted_expense   residual")

    triples = sorted(zip(x, y, y_pred), key=lambda t: t[0])
    step = max(1, len(triples) // rows)
    for xi, yi, pi in triples[::step][:rows]:
        residual = yi - pi
        print(f"  {xi:5.1f}   {yi:13.2f}   {pi:17.2f}   {residual:9.2f}")


def print_pipeline_outline() -> None:
    """Print an ordered lesson outline for script 2."""
    print("Pipeline outline (script 2):")
    print("  1) Read data: load feature (bmi) and target (expenses).")
    print("  2) Train model: solve OLS with Normal Equation in one closed-form step.")
    print("  3) Learn parameters: get w0 (intercept) and w1 (slope).")
    print("  4) Evaluate fit: compute MSE and R^2 on the dataset.")
    print("  5) Inspect predictions: compare actual vs predicted and residuals.")
    print("  6) If fit is not good enough: move to script 3/4, add features, and test split-based evaluation.")
    print()


def main() -> None:
    print_pipeline_outline()

    # Single-feature regression is great for teaching because the line equation
    # is easy to interpret: predicted_expenses = w0 + w1 * bmi.
    x, y = load_xy(CSV_PATH, FEATURE, TARGET)
    w0, w1 = fit_normal_equation_single_feature(x, y)
    preds = [w0 + w1 * xi for xi in x]
    model_mse = mse(y, preds)
    model_r2 = r2_score(y, preds)

    # Slope interpretation: change in predicted expenses for +1 unit of BMI.
    direction = "increase" if w1 >= 0 else "decrease"

    print("Method: Analytical OLS via Normal Equation")
    print(f"Dataset: {CSV_PATH}")
    print(f"Model: {TARGET} = w0 + w1*{FEATURE}")
    print()

    print("How script 2 differs from script 1:")
    print("  script 1: explores/describes data (shape, summaries, correlations, text visuals)")
    print("  script 2: trains a linear model and finds best-fit coefficients with one closed-form step")
    print("  script 2 output is about prediction quality, not just description")
    print()

    print("What you are looking at:")
    print("  w0 (intercept): predicted expenses when bmi=0 (baseline in this line equation)")
    print("  w1 (slope): expected change in predicted expenses for each +1 BMI")
    print("  MSE: average squared prediction error (lower is better)")
    print("  R^2: fraction of variance explained by the model (closer to 1 is better)")
    print()

    print(f"w0 (intercept): {w0:.6f}")
    print(f"w1 (slope):     {w1:.6f}")
    print(f"MSE:            {model_mse:.2f}")
    print(f"R^2:            {model_r2:.4f}")
    print()

    print("Student interpretation summary:")
    print(f"  Because w1 is {w1:.2f}, predicted expenses {direction} by about ${abs(w1):.2f} per +1 BMI.")
    print(f"  R^2={model_r2:.4f} means this single-feature model explains about {model_r2 * 100:.2f}% of expense variation.")
    print("  This is expected to be limited because real medical costs depend on more than BMI alone.")
    print_prediction_table(x, y, preds)


if __name__ == "__main__":
    main()
