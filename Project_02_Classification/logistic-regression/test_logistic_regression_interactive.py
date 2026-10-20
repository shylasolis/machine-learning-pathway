"""Run with: python -m unittest discover -s . -p "test_*.py"."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from logistic_regression_interactive import LogisticRegressionExplorer, sigmoid


class SigmoidTests(unittest.TestCase):
    def test_known_probabilities_and_extreme_scores(self) -> None:
        with np.errstate(over="raise", invalid="raise"):
            actual = sigmoid([-1000, -np.log(3), 0, np.log(3), 1000])
        np.testing.assert_allclose(actual, [0, 0.25, 0.5, 0.75, 1])


class ExplorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explorer = LogisticRegressionExplorer()
        # Draw explicitly in the render test; avoid rendering for every slider event.
        self.explorer.figure.canvas.draw_idle = lambda: None

    def tearDown(self) -> None:
        plt.close(self.explorer.figure)

    def test_initial_prediction_includes_threshold_equality(self) -> None:
        np.testing.assert_allclose(
            self.explorer.curve.get_ydata(), sigmoid(self.explorer.x_values),
        )
        np.testing.assert_allclose(self.explorer.selected_point.get_ydata(), [0.5])
        self.assertIn("Predicted class: 1", self.explorer.readout.get_text())

    def test_coefficient_intercept_and_input_update_artists(self) -> None:
        self.explorer.intercept_slider.set_val(2.0)
        self.explorer.coefficient_slider.set_val(-0.5)
        self.explorer.input_slider.set_val(3.0)
        np.testing.assert_allclose(
            self.explorer.curve.get_ydata(),
            sigmoid(2.0 - 0.5 * self.explorer.x_values),
        )
        np.testing.assert_allclose(self.explorer.selected_point.get_xdata(), [3])
        np.testing.assert_allclose(
            self.explorer.selected_point.get_ydata(), [float(sigmoid(0.5))],
        )
        np.testing.assert_allclose(self.explorer.input_guide.get_xdata(), [3, 3])
        np.testing.assert_allclose(
            self.explorer.input_guide.get_ydata(), [0, float(sigmoid(0.5))],
        )

    def test_threshold_changes_class_but_not_probability(self) -> None:
        original_curve = self.explorer.curve.get_ydata().copy()
        self.explorer.threshold_slider.set_val(0.51)
        self.assertIn("Predicted class: 0", self.explorer.readout.get_text())
        self.assertEqual(self.explorer.selected_point.get_color(), "tab:red")
        np.testing.assert_allclose(
            self.explorer.threshold_line.get_ydata(), [0.51, 0.51],
        )
        np.testing.assert_allclose(self.explorer.curve.get_ydata(), original_curve)
        np.testing.assert_allclose(self.explorer.selected_point.get_ydata(), [0.5])
        self.explorer.threshold_slider.set_val(0.49)
        self.assertIn("Predicted class: 1", self.explorer.readout.get_text())
        self.assertEqual(self.explorer.selected_point.get_color(), "tab:green")

    def test_zero_coefficient_produces_constant_probability(self) -> None:
        self.explorer.coefficient_slider.set_val(0)
        self.explorer.intercept_slider.set_val(-2)
        self.explorer.input_slider.set_val(10)
        np.testing.assert_allclose(
            self.explorer.curve.get_ydata(), float(sigmoid(-2)),
        )
        np.testing.assert_allclose(
            self.explorer.selected_point.get_ydata(), [float(sigmoid(-2))],
        )

    def test_slider_extremes_stay_finite(self) -> None:
        for intercept, coefficient, x, threshold in (
            (-10, 5, -10, 0.01),
            (10, -5, -10, 0.99),
            (10, 5, 10, 0.99),
            (-10, -5, 10, 0.01),
        ):
            for slider, value in zip(
                self.explorer.sliders, (intercept, coefficient, x, threshold),
            ):
                slider.set_val(value)
            probabilities = self.explorer.curve.get_ydata()
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_reset_button_restores_all_defaults(self) -> None:
        for slider, value in zip(self.explorer.sliders, (2, -1, 4, 0.8)):
            slider.set_val(value)
        self.explorer.reset_button._observers.process("clicked", None)
        self.assertEqual(
            [float(slider.val) for slider in self.explorer.sliders],
            [0.0, 1.0, 0.0, 0.5],
        )
        self.assertIn("Predicted class: 1", self.explorer.readout.get_text())
        np.testing.assert_allclose(
            self.explorer.curve.get_ydata(), sigmoid(self.explorer.x_values),
        )

    def test_canvas_renders(self) -> None:
        self.explorer.figure.canvas.draw()
        self.assertGreater(len(self.explorer.figure.canvas.buffer_rgba()), 0)


if __name__ == "__main__":
    unittest.main()
