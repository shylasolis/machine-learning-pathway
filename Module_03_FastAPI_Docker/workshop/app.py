"""Local-only teaching API: load once, validate inputs, reuse fitted preprocessing."""

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import sklearn
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
FEATURES = ["mean radius", "mean texture"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mean_radius: float = Field(strict=True, gt=0, le=100)
    mean_texture: float = Field(strict=True, gt=0, le=100)


class PredictionResponse(BaseModel):
    model_id: str
    probability_class_1: float = Field(ge=0, le=1)
    predicted_class: Literal[0, 1]
    label: Literal["benign", "malignant"]
    threshold: float
    positive_class: Literal["malignant"]
    educational_only: Literal[True] = True


def load_pipeline() -> tuple[Pipeline, dict]:
    metadata_path = ROOT / "artifacts" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["features"] != FEATURES or metadata["threshold"] != 0.5:
        raise RuntimeError("Artifact schema or threshold does not match this API.")
    if metadata["scikit_learn"] != sklearn.__version__:
        raise RuntimeError("scikit-learn version mismatch; install requirements.txt.")
    # Only load the supplied trusted artifact: joblib/pickle can execute code.
    pipeline = joblib.load(ROOT / "artifacts" / "teaching_pipeline.joblib")
    if not isinstance(pipeline, Pipeline) or pipeline.n_features_in_ != 2:
        raise RuntimeError("Expected a fitted two-feature scikit-learn pipeline.")
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError("Expected benign=0 and malignant=1 model classes.")
    return pipeline, metadata


@asynccontextmanager
async def lifespan(application: FastAPI):
    pipeline, metadata = load_pipeline()
    application.state.pipeline = pipeline
    application.state.metadata = metadata
    yield


app = FastAPI(
    title="Logistic Regression Inference Workshop",
    description=(
        "Local classroom demonstration only. Not a diagnostic service. "
        "Inputs are cell-nucleus measurements, not patient demographics."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready", "model_id": app.state.metadata["model_id"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    features = np.array([[request.mean_radius, request.mean_texture]])
    probability = float(app.state.pipeline.predict_proba(features)[0, 1])
    threshold = app.state.metadata["threshold"]
    predicted_class = int(probability >= threshold)
    return PredictionResponse(
        model_id=app.state.metadata["model_id"],
        probability_class_1=probability,
        predicted_class=predicted_class,
        label="malignant" if predicted_class else "benign",
        threshold=threshold,
        positive_class="malignant",
    )
