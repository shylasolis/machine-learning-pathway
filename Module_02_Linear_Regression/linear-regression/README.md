# Linear Regression Learning Lab

A beginner-friendly project for learning linear regression from first principles using real insurance data.

Students will learn two ways to train the same type of model:
- Normal Equation (analytical solution)
- Gradient Descent (iterative optimization)

The project includes Python scripts and a Jupyter notebook so you can learn either in code-first or notebook-first style.

## What You Are Downloading

This repository contains:
- Educational Python scripts that build from data exploration to model comparison
- A Jupyter notebook version of the lesson with visual outputs and interpretations
- A local copy of the insurance dataset used in all examples
- A saved model-comparison chart

## Learning Objectives

By the end of this lab, students should be able to:
1. Load and inspect a real dataset.
2. Explain the difference between features and target.
3. Train a single-feature linear regression model with the Normal Equation.
4. Train the same model with Gradient Descent.
5. Understand why feature standardization helps Gradient Descent.
6. Evaluate model quality using MSE, RMSE, MAE, and R2.
7. Compare train vs test performance and discuss generalization.
8. Translate technical results into plain-language stakeholder takeaways.

## Repository Structure

- 01_simple_linear.py
- 02_ols_normal_equation.py
- 03_gradient_descent.py
- 04_compare_models_visual.py
- linear_regression_lab.ipynb
- requirements.txt
- data/insurance-premium-prediction/insurance.csv
- data/model_comparison_bmi_expenses.png

## Prerequisites

- Python 3.10 or newer recommended
- Git
- Internet access (only needed for initial clone and optional package installs)

## 1) Clone the Project

Run in a terminal:

```bash
git clone https://github.com/shylasolis/linear-regression-with-standardization.git
cd linear-regression-with-standardization
```

## 2) Create and Activate a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS (Terminal or zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install Dependencies

### Windows

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 4) Run the Scripts in Order

### Windows

```powershell
python 01_simple_linear.py
python 02_ols_normal_equation.py
python 03_gradient_descent.py
python 04_compare_models_visual.py
```

### macOS

```bash
python3 01_simple_linear.py
python3 02_ols_normal_equation.py
python3 03_gradient_descent.py
python3 04_compare_models_visual.py
```

Why this order:
1. Explore and understand the data
2. Learn analytical OLS
3. Learn iterative OLS with Gradient Descent
4. Compare both methods and review business interpretation

## 5) Run the Jupyter Notebook

Start Jupyter:

### Windows

```powershell
python -m notebook
```

### macOS

```bash
python3 -m notebook
```

Then open:
- linear_regression_lab.ipynb

Run cells from top to bottom.

## Key Concepts Students Should Notice

- The Normal Equation and Gradient Descent should converge to very similar model parameters for this problem.
- Standardizing BMI improves optimization stability for Gradient Descent.
- A single-feature model has limited predictive power, which is expected and educational.
- Metrics are only useful when interpreted in context of business goals and baseline performance.

## Troubleshooting

- ModuleNotFoundError:
  - Make sure your virtual environment is activated.
  - Re-run package installation from requirements.txt.
- Wrong Python interpreter in VS Code:
  - Use Command Palette -> Python: Select Interpreter -> choose .venv.
- Notebook import issues:
  - Ensure the notebook kernel is set to the same .venv environment.

## Suggested Class Activity

1. Try several learning rates and compare convergence curves.
2. Add more features and discuss changes in R2 and RMSE.
3. Create a validation split and tune hyperparameters.
4. Present results to a non-technical audience in plain language.

## License and Data

- Educational use repository.
- Dataset source: Kaggle insurance premium prediction dataset.
