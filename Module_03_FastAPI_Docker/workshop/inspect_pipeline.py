"""Print the supplied, trusted teaching pipeline and its learned values."""

from pathlib import Path

import joblib


def main() -> None:
    path = Path(__file__).resolve().parent / "artifacts" / "teaching_pipeline.joblib"
    pipeline = joblib.load(path)

    print("FILE:", path)
    print("\nPIPELINE")
    print(pipeline)

    print("\nSTEPS")
    for name, step in pipeline.named_steps.items():
        print(f"{name}: {type(step).__name__}")

    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]

    print("\nSCALER - learned from the training data")
    print("Feature order: mean radius, mean texture")
    print("Training means:", scaler.mean_)
    print("Training scales:", scaler.scale_)

    print("\nLOGISTIC REGRESSION - learned model values")
    print("Classes:", model.classes_, "(0 = benign, 1 = malignant)")
    print("Coefficients (for standardized features):", model.coef_)
    print("Intercept:", model.intercept_)

    print("\nMODEL SETTINGS")
    print(model.get_params())


if __name__ == "__main__":
    main()
