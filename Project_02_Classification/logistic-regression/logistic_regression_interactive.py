"""Explore a one-feature logistic regression model with live sliders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from numpy.typing import ArrayLike, NDArray


def sigmoid(logits: ArrayLike) -> NDArray[np.float64]:
    """Compute probabilities without overflowing for large positive/negative scores."""
    scores = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -scores))


class LogisticRegressionExplorer:
    """Keep the plot and widgets alive for the lifetime of the window."""

    def __init__(self) -> None:
        self.x_values = np.linspace(-10.0, 10.0, 801)
        self.figure = plt.figure(figsize=(12, 8))
        self.figure.canvas.manager.set_window_title("Logistic Regression Explorer")
        self.figure.suptitle("Logistic Regression Explorer", fontsize=20, y=0.97)
        self.figure.text(
            0.09, 0.92,
            r"Explore how $p(y=1 \mid x) = 1 / (1 + e^{-(\beta_0 + \beta_1 x)})$ changes.",
            fontsize=12,
        )
        self.axes = self.figure.add_axes((0.09, 0.44, 0.61, 0.45))
        self.axes.set(
            xlim=(-10, 10),
            ylim=(-0.08, 1.08),
            xlabel="Feature value x",
            ylabel="Predicted probability of class 1",
        )
        self.axes.grid(alpha=0.2)

        self.curve, = self.axes.plot(
            self.x_values, sigmoid(self.x_values),
            color="tab:blue", linewidth=2.5, label="Model probability",
        )
        self.threshold_line = self.axes.axhline(
            0.5, color="black", linestyle="--", linewidth=1.5,
            label="Classification threshold",
        )
        self.input_guide, = self.axes.plot(
            [0, 0], [0, 0.5], color="tab:orange", linestyle=":",
        )
        self.selected_point, = self.axes.plot(
            [0], [0.5], marker="o", markersize=10, color="tab:orange",
            linestyle="none", zorder=5,
        )

        # Fixed, illustrative observations; moving sliders never fits these points.
        self.axes.scatter(
            [-8, -6, -4, -2, 1], [0] * 5, marker="x", s=55,
            color="tab:red", label="Observed class 0", zorder=4,
        )
        self.axes.scatter(
            [-1, 2, 4, 6, 8], [1] * 5, marker="^", s=55,
            color="tab:green", label="Observed class 1", zorder=4,
        )
        self.axes.legend(loc="upper left", fontsize=8)
        self.readout = self.figure.text(
            0.75, 0.86, "", va="top", fontsize=12, linespacing=1.7,
        )
        self.figure.text(
            0.75, 0.49,
            "Decision rule:\nclass 1 if p >= threshold;\notherwise class 0.",
            fontsize=10, linespacing=1.5,
        )

        self.intercept_slider = Slider(
            self.figure.add_axes((0.23, 0.30, 0.46, 0.025)),
            r"Intercept $\beta_0$", -10.0, 10.0, valinit=0.0, valstep=0.1,
        )
        self.coefficient_slider = Slider(
            self.figure.add_axes((0.23, 0.245, 0.46, 0.025)),
            r"Coefficient $\beta_1$", -5.0, 5.0, valinit=1.0, valstep=0.1,
        )
        self.input_slider = Slider(
            self.figure.add_axes((0.23, 0.19, 0.46, 0.025)),
            "Input x", -10.0, 10.0, valinit=0.0, valstep=0.1,
        )
        self.threshold_slider = Slider(
            self.figure.add_axes((0.23, 0.135, 0.46, 0.025)),
            "Threshold", 0.01, 0.99, valinit=0.5, valstep=0.01,
            valfmt="%1.2f",
        )
        self.sliders = (
            self.intercept_slider,
            self.coefficient_slider,
            self.input_slider,
            self.threshold_slider,
        )
        for slider in self.sliders:
            slider.on_changed(self.update)

        self.reset_button = Button(
            self.figure.add_axes((0.79, 0.125, 0.12, 0.055)), "Reset",
        )
        self.reset_button.on_clicked(self.reset)
        self.figure.text(
            0.09, 0.055,
            "Teaching visualization: coefficients are set manually, not learned from the observations.",
            fontsize=10,
        )
        self.update()

    def update(self, _value: float | None = None) -> None:
        intercept = float(self.intercept_slider.val)
        coefficient = float(self.coefficient_slider.val)
        x = float(self.input_slider.val)
        threshold = float(self.threshold_slider.val)
        score = intercept + coefficient * x
        probability = float(sigmoid(score))
        predicted_class = int(probability >= threshold)

        self.curve.set_ydata(sigmoid(intercept + coefficient * self.x_values))
        self.threshold_line.set_ydata([threshold, threshold])
        self.input_guide.set_data([x, x], [0, probability])
        self.selected_point.set_data([x], [probability])
        self.selected_point.set_color(
            "tab:green" if predicted_class == 1 else "tab:red",
        )
        self.readout.set_text(
            "Current prediction\n"
            f"Intercept: {intercept:.1f}\n"
            f"Coefficient: {coefficient:.1f}\n"
            f"Input x: {x:.1f}\n"
            f"Score z: {score:.2f}\n"
            f"Probability: {probability:.4f}\n"
            f"Threshold: {threshold:.2f}\n"
            f"Predicted class: {predicted_class}"
        )
        self.figure.canvas.draw_idle()

    def reset(self, _event: object = None) -> None:
        for slider in self.sliders:
            slider.reset()


def main() -> None:
    explorer = LogisticRegressionExplorer()
    plt.show()
    # Retain widget callbacks until the blocking GUI event loop has finished.
    plt.close(explorer.figure)


if __name__ == "__main__":
    main()
