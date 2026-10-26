import unittest
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from app import ROOT, app, load_pipeline


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_health(self) -> None:
        result = self.client.get("/health")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "ready")

    def test_probability_matches_saved_pipeline_and_reference(self) -> None:
        pipeline, metadata = load_pipeline()
        result = self.client.post("/predict", json=metadata["example"]["request"])
        self.assertEqual(result.status_code, 200)
        expected = float(pipeline.predict_proba(np.array([[14.0, 20.0]]))[0, 1])
        self.assertAlmostEqual(result.json()["probability_class_1"], expected)
        self.assertAlmostEqual(expected, metadata["example"]["probability_class_1"])
        self.assertTrue(result.json()["educational_only"])

    def test_invalid_inputs_return_422(self) -> None:
        for payload in [
            {}, {"mean_radius": 14},
            {"mean_radius": -1, "mean_texture": 20},
            {"mean_radius": 0, "mean_texture": 20},
            {"mean_radius": 101, "mean_texture": 20},
            {"mean_radius": "14", "mean_texture": 20},
            {"mean_radius": True, "mean_texture": 20},
            {"mean_radius": 14, "mean_texture": None},
            {"mean_radius": 14, "mean_texture": 20, "unknown": 1},
        ]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_probability_equal_to_threshold_is_class_one(self) -> None:
        with patch.object(
            app.state.pipeline, "predict_proba", return_value=np.array([[0.5, 0.5]]),
        ):
            result = self.client.post("/predict", json={"mean_radius": 14, "mean_texture": 20})
        self.assertEqual(result.json()["predicted_class"], 1)

    def test_requests_do_not_fit_or_change_scaler(self) -> None:
        scaler = app.state.pipeline.named_steps["standardscaler"]
        original = scaler.mean_.copy()
        with patch.object(app.state.pipeline, "fit", side_effect=AssertionError("No fitting at inference")):
            for _ in range(2):
                result = self.client.post("/predict", json={"mean_radius": 14, "mean_texture": 20})
                self.assertEqual(result.status_code, 200)
        np.testing.assert_array_equal(scaler.mean_, original)

    def test_readable_weights_and_scaler_reproduce_pipeline(self) -> None:
        weights = json.loads(
            (ROOT / "artifacts" / "logistic_weights.json").read_text(encoding="utf-8"),
        )
        scaler = json.loads(
            (ROOT / "artifacts" / "preprocessing.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(weights["feature_order"], scaler["feature_order"])
        examples = np.array([[14.0, 20.0], [10.0, 12.0], [25.0, 30.0]])
        scaled = (examples - scaler["training_mean"]) / scaler["training_scale"]
        score = scaled @ np.array(weights["coefficients"]) + weights["intercept"]
        actual = np.exp(-np.logaddexp(0, -score))
        np.testing.assert_allclose(
            actual, app.state.pipeline.predict_proba(examples)[:, 1], atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
