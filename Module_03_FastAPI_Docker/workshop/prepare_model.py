"""Rebuild the supplied teaching artifact; not part of API startup."""

import json
from pathlib import Path
import platform
from zipfile import ZipFile

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent


def main() -> None:
    data = load_breast_cancer()
    features = ["mean radius", "mean texture"]
    x = data.data[:, [list(data.feature_names).index(name) for name in features]]
    y = (data.target == 0).astype(int)
    train, _ = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42,
    )
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000),
    )
    pipeline.fit(x[train], y[train])
    example = np.array([[14.0, 20.0]])
    probability = float(pipeline.predict_proba(example)[0, 1])
    metadata = {
        "model_id": "logistic-teaching-v1",
        "features": features,
        "positive_class": "malignant",
        "negative_class": "benign",
        "threshold": 0.5,
        "purpose": "Educational inference demonstration; not for clinical use.",
        "dataset": "scikit-learn Wisconsin Diagnostic Breast Cancer",
        "training": "80% stratified split, random_state=42; C=1.0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "example": {
            "request": {"mean_radius": 14.0, "mean_texture": 20.0},
            "probability_class_1": probability,
            "predicted_class": int(probability >= 0.5),
        },
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(pipeline, artifacts / "teaching_pipeline.joblib")
    (artifacts / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    model = pipeline.named_steps["logisticregression"]
    scaler = pipeline.named_steps["standardscaler"]
    weights = {
        "feature_order": features,
        "coefficient_units": "standardized features",
        "intercept": float(model.intercept_[0]),
        "coefficients": model.coef_[0].tolist(),
        "classes": model.classes_.tolist(),
        "threshold": 0.5,
    }
    preprocessing = {
        "feature_order": features,
        "transformation": "(raw_value - training_mean) / training_scale",
        "training_mean": scaler.mean_.tolist(),
        "training_scale": scaler.scale_.tolist(),
    }
    for filename, content in [
        ("logistic_weights.json", weights),
        ("preprocessing.json", preprocessing),
        ("example_request.json", metadata["example"]["request"]),
    ]:
        (artifacts / filename).write_text(
            json.dumps(content, indent=2) + "\n", encoding="utf-8",
        )
    with ZipFile(ROOT / "model_handoff.zip", "w") as archive:
        for filename in [
            "teaching_pipeline.joblib", "model_metadata.json",
            "logistic_weights.json", "preprocessing.json", "example_request.json",
        ]:
            archive.write(artifacts / filename, arcname=f"artifacts/{filename}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
