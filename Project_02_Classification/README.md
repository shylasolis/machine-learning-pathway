# Project 2: Classification

This project introduces classification concepts and model-selection
considerations.

## Material

- [Classification foundations lecture](Classification_Foundations_Lecture.pptx)

The lecture covers classification targets and use cases; choosing algorithms
for tabular, text, and image data; model-specific preprocessing; image and mask
preparation; data leakage; class imbalance; evaluation metrics and thresholds;
calibration; and error analysis.

## Project Assignment 2: Classification

After the logistic regression lesson, complete
[Logistic Regression from Scores to Decisions](assignment_instructions.md).
You will implement NumPy gradient descent on two features from scikit-learn's
bundled breast cancer dataset, compare it with a library model and baseline,
and evaluate a validation-selected classification threshold.
Continue Project 1's professional portfolio expectations: publish in your own
well-organized Git repository with a descriptive project name and beginner
Windows/macOS README instructions. A peer must reproduce the final version
without your help. Submit the Git link and completed validation PDF on Canvas;
do not commit the completed form to Git. See the assignment for the full process.

See the [student rubric](assignment_rubric.md) for the expectations used to
review your work. The [assignment dependencies](assignment_requirements.txt)
include scikit-learn in addition to the GUI's NumPy and Matplotlib requirements.
This is a classroom modeling exercise, not a medical diagnostic tool.

For viewing or printing, use the Pacific orange-and-black
[assignment PDF](assignment_instructions.pdf) and
[student rubric PDF](assignment_rubric.pdf). The student rubric lists
expectations without point weights.

## Interactive Logistic Regression Lab

The [Logistic Regression Explorer](logistic-regression/logistic_regression_interactive.py)
is a standalone desktop GUI built with NumPy and Matplotlib. It uses illustrative
observations and requires no dataset downloads or web browser.

### Run in VS Code or Visual Studio

Use Python 3.10 or newer. From the repository root, run these commands in a
PowerShell terminal with your Python environment activated:

```powershell
cd .\Project_02_Classification\logistic-regression
python -m pip install -r requirements.txt
python logistic_regression_interactive.py
```

In VS Code, select the same environment with **Python: Select Interpreter**.
Run the script in the terminal (not an inline notebook or Plot Viewer) so that
Matplotlib opens a separate interactive window. In Visual Studio, use the same
environment and run the script as your startup file.

### Explore the model

- **Intercept (beta_0), -10 to 10:** shifts the sigmoid horizontally when the
  coefficient is nonzero. When the coefficient is zero, it changes the constant
  probability instead.
- **Coefficient (beta_1), -5 to 5:** changes steepness and direction. Negative
  values give a decreasing curve; zero gives a flat curve.
- **Input x, -10 to 10:** moves the highlighted point along the curve and updates
  its score, probability, and predicted class.
- **Threshold, 0.01 to 0.99:** moves the dashed decision threshold. A probability
  greater than or equal to the threshold predicts class 1; otherwise class 0.
  This changes the classification, not the probability curve.
- **Reset:** restores all four sliders to their starting values.

The observed class-0 and class-1 markers stay fixed as reference examples.
**Moving sliders manually changes the model; it does not train on those
observations.** There is no Train Model or gradient-descent feature in this lab.

Try reversing the coefficient, setting it to zero, and changing the threshold
while keeping the coefficients and input fixed.

### Troubleshooting

- Missing NumPy or Matplotlib: rerun the dependency installation with the same
  interpreter you use to launch the script.
- No interactive window: use a local desktop Python session with a GUI backend,
  such as TkAgg or QtAgg, rather than the noninteractive Agg backend. An explicit
  `MPLBACKEND=Agg` setting or a headless/remote session cannot open a desktop GUI.
  If Python reports that Tk is missing, install/enable Tcl/Tk through your Python
  installation, or use an available Qt backend.

### Check the lab

From the lab folder, run the headless tests (no desktop window required):

```powershell
python -m unittest discover -s . -p "test_*.py"
```
