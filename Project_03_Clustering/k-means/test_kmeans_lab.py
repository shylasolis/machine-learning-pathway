import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from kmeans_lab import (
    INITIAL_CENTERS, TOY, lloyd_trace, make_customers, partition_indices,
    run_experiment, squared_distances, stream_minibatches, write_outputs,
)


class LloydTests(unittest.TestCase):
    def test_assignment_update_and_objective(self):
        trace = lloyd_trace(TOY, INITIAL_CENTERS)
        self.assertTrue(trace[-1]["stopped"])
        np.testing.assert_array_equal(trace[0]["labels"], [0, 1, 0, 1, 1, 1])
        selected = trace[0]["distances"][np.arange(len(TOY)), trace[0]["labels"]]
        np.testing.assert_array_equal(selected, [0, 0, 5, 61, 80, 100])
        self.assertEqual(selected.sum(), 246)
        self.assertEqual(trace[0]["inertia_before"], 246)
        np.testing.assert_allclose(
            ((TOY - trace[0]["centers_after"][trace[0]["labels"]]) ** 2).sum(axis=1),
            [1.25, 44.3125, 1.25, 1.8125, 6.3125, 11.3125],
        )
        self.assertAlmostEqual(trace[0]["inertia_after"], 66.25)
        self.assertAlmostEqual(trace[1]["inertia_before"], 25.1875)
        self.assertAlmostEqual(trace[0]["shift"], 6.75)
        self.assertAlmostEqual(trace[1]["shift"], np.sqrt(1/9 + 1/4 + 121/36 + 25/16))
        self.assertEqual(trace[-1]["shift"], 0)
        np.testing.assert_array_equal(trace[-1]["labels"], [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(trace[-1]["centers_after"], [[5/3, 2], [25/3, 8]])
        self.assertAlmostEqual(trace[-1]["inertia_after"], 28/3)
        for step in trace:
            self.assertLessEqual(step["inertia_after"], step["inertia_before"] + 1e-9)
            for j in range(2):
                np.testing.assert_allclose(
                    step["centers_after"][j], TOY[step["labels"] == j].mean(axis=0)
                )
        values = [t["inertia_after"] for t in trace]
        self.assertTrue(np.all(np.diff(values) <= 1e-9))

    def test_squared_distance_and_tie_rule(self):
        np.testing.assert_array_equal(squared_distances(TOY[:1], INITIAL_CENTERS), [[0, 4]])
        trace = lloyd_trace(np.array([[0.], [1.], [2.]]), np.array([[0.], [2.]]))
        self.assertEqual(trace[0]["labels"][1], 0)

    def test_cap_not_convergence(self):
        trace = lloyd_trace(TOY, INITIAL_CENTERS, max_iter=1)
        self.assertEqual(len(trace), 1)
        self.assertFalse(trace[-1]["stopped"])

    def test_worked_customer_silhouette(self):
        labels = lloyd_trace(TOY, INITIAL_CENTERS)[-1]["labels"]
        distances = np.sqrt(((TOY - TOY[0]) ** 2).sum(axis=1))
        np.testing.assert_allclose(distances[1:], [2, np.sqrt(5), np.sqrt(85), 10, np.sqrt(128)])
        a = distances[1:3].mean()
        b = distances[3:].mean()
        scores = silhouette_samples(TOY, labels)
        self.assertAlmostEqual(scores[0], (b - a) / max(a, b))
        self.assertAlmostEqual(scores.mean(), silhouette_score(TOY, labels))

    def test_worked_scaling_example(self):
        raw = np.array([[2, 20], [2, 40], [6, 20], [6, 40]], dtype=float)
        scaler = StandardScaler().fit(raw)
        np.testing.assert_array_equal(scaler.mean_, [4, 30])
        np.testing.assert_array_equal(scaler.var_, [4, 100])
        np.testing.assert_array_equal(scaler.scale_, [2, 10])
        scaled = scaler.transform(raw)
        np.testing.assert_array_equal(scaled, [[-1, -1], [-1, 1], [1, -1], [1, 1]])
        self.assertEqual(((raw[3] - raw[0]) ** 2).sum(), 416)
        self.assertEqual(((scaled[3] - scaled[0]) ** 2).sum(), 8)

    def test_invalid_or_empty_input_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Empty cluster"):
            lloyd_trace(TOY, np.array([[1., 1.], [1., 1.]]))
        with self.assertRaises(ValueError):
            lloyd_trace(TOY, INITIAL_CENTERS, max_iter=0)
        with self.assertRaises(ValueError):
            stream_minibatches(TOY, k=2, batch_size=1)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_experiment()

    def test_reproducibility_and_disjoint_splits(self):
        x = make_customers()
        np.testing.assert_array_equal(x, make_customers())
        self.assertEqual(x.shape, (600, 2))
        self.assertTrue((x > 0).all())
        train, validation, test, development = partition_indices(len(x))
        self.assertEqual([len(train), len(validation), len(test), len(development)],
                         [360, 120, 120, 480])
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(development) & set(test))
        self.assertEqual(set(train) | set(validation), set(development))
        self.assertEqual(set(development) | set(test), set(range(600)))

    def test_scaling_selection_and_no_leakage(self):
        r = self.result
        np.testing.assert_allclose(r["scaler"].mean_, r["x"][r["train"]].mean(axis=0))
        np.testing.assert_allclose(
            r["final_scaler"].mean_, r["x"][r["development"]].mean(axis=0)
        )
        self.assertEqual(len(r["selection"]), 5)
        expected = min(r["selection"], key=lambda row: (-row["validation_silhouette"], row["k"]))
        self.assertEqual(r["k"], expected["k"])
        for row in r["selection"]:
            self.assertTrue(-1 <= row["validation_silhouette"] <= 1)

    def test_inertia_residuals_and_streaming(self):
        r = self.result
        labels = r["final"].predict(r["x_development"])
        residuals = r["x_development"] - r["final"].cluster_centers_[labels]
        self.assertAlmostEqual((residuals ** 2).sum(), r["final"].inertia_)
        self.assertEqual(r["mini"].cluster_centers_.shape, (r["k"], 2))
        self.assertEqual(len(r["stability"]), 10)
        self.assertTrue(np.isfinite(r["stability"]).all())
        self.assertFalse(hasattr(r["final"], "partial_fit"))
        raw = r["final_scaler"].inverse_transform(r["final"].cluster_centers_)
        np.testing.assert_allclose(
            r["final_scaler"].transform(raw), r["final"].cluster_centers_
        )

    def test_outputs(self):
        with tempfile.TemporaryDirectory() as destination:
            write_outputs(self.result, Path(destination))
            names = {p.name for p in Path(destination).iterdir()}
            self.assertEqual(names, {
                "k_selection.csv", "test_metrics.csv", "cluster_profiles.csv",
                "toy_iterations.csv", "toy_iterations.png", "k_selection.png",
                "test_clusters.png", "scaling_comparison.png",
            })
            self.assertTrue(all(p.stat().st_size > 50 for p in Path(destination).iterdir()))
            specifications = {
                "k_selection.csv": (5, [
                    "k", "train_mean_squared_distance", "validation_mean_squared_distance",
                    "validation_silhouette", "n_iter",
                ]),
                "test_metrics.csv": (2, ["model", "mean_squared_distance", "silhouette"]),
                "cluster_profiles.csv": (self.result["k"], [
                    "cluster", "development_count", "orders_per_month", "average_basket_dollars",
                ]),
            }
            for name, (count, columns) in specifications.items():
                with (Path(destination) / name).open(newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    self.assertEqual(reader.fieldnames, columns)
                    rows = list(reader)
                self.assertEqual(len(rows), count)
                if name == "cluster_profiles.csv":
                    self.assertEqual(sum(int(row["development_count"]) for row in rows), 480)


class TeachingLayoutTests(unittest.TestCase):
    def test_tall_equations_leave_space_before_next_paragraph(self):
        from build_kmeans_materials import BLACK, ORANGE, Document

        self.assertEqual(BLACK, "#231f20")
        self.assertEqual(ORANGE, "#e65300")
        with tempfile.TemporaryDirectory() as destination:
            doc = Document(Path(destination) / "equation.pdf", "Equation layout")
            doc.page("Readable mathematics")
            doc.math(r"J=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2")
            equation = doc.ax.texts[-1]
            doc.text("This explanation must start below the entire equation.")
            paragraph = doc.ax.texts[-1]
            doc.fig.canvas.draw()
            renderer = doc.fig.canvas.get_renderer()
            self.assertGreater(
                equation.get_window_extent(renderer).y0,
                paragraph.get_window_extent(renderer).y1,
            )
            doc.close()


if __name__ == "__main__":
    unittest.main()
