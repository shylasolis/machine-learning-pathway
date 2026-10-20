# Student Rubric: Project Assignment 2 - Classification

**Portfolio project:** Logistic Regression from Scores to Decisions

This rubric describes what will be checked. Use it with the
[assignment instructions](assignment_instructions.md) before submitting.
Evaluation considers both the correctness of your work and the evidence that
you understand the model; submitting an output file alone is not sufficient.

## Review criteria

| Area | Expectations for complete work | Evidence reviewed |
|---|---|---|
| Score, sigmoid, and threshold | All four exploration cases have correct scores, probabilities, and labels; case A includes a worked calculation. Explanations distinguish coefficient effects, a zero coefficient, threshold changes, and manual adjustment from training. | Calculation table and report; GUI or equivalent Python calculations |
| Dataset and positive class | Uses the bundled dataset and only `mean radius` and `mean texture`, in order. Correctly recodes malignant as 1, acknowledges the source, and reports missingness and class counts. | Loading code, console output, and report |
| Experimental design and preprocessing | Uses the specified reproducible, stratified training/validation/test split; verifies no row overlap. Fits the scaler on training rows only and reuses it. Does not use test results for development or refit after threshold selection. | Split/scaling code, partition counts, and leakage explanation |
| NumPy learning implementation | Implements its own intercept, stable sigmoid, mean binary cross-entropy, and full-batch gradient updates without a fitting library. Uses the specified initialization and settings or justifies changes. Shows finite, converged loss and labeled standardized-unit coefficients. | Training code, initial/final loss, and loss plot |
| Reference and baseline comparison | Fits the unregularized scikit-learn reference and prior baseline on identical training rows. Uses the correct positive-probability column. Compares coefficients and validation probabilities and investigates substantial disagreement. | Reference/baseline code, coefficient table, and numerical comparison |
| Metrics and threshold selection | Correctly computes labeled confusion counts, accuracy, precision, recall, F1, and probability-based log loss. Evaluates all seven validation thresholds, applies the stated F1/tie rule, and freezes the selected threshold before testing. Evaluates all required configurations, including the baseline. | Evaluation code, printed table, and both CSVs |
| Professional portfolio and reproducibility | Uses a descriptive professional repository name, logical organization, and meaningful commits. The root README explains the question, data, methods, findings, and complete beginner-friendly setup for Windows and macOS, including software installation, cloning, environments, dependencies, commands, outputs, versions, and troubleshooting. The script and all required outputs are present and reproducible. | Git repository, README, dependency file, script, report, CSVs, plot, and commit history |
| Independent validation and Canvas submission | A peer reproduces the final version on their own computer using only the README without author assistance. Problems are fixed and independently retested; the tested OS and final commit are recorded. Canvas receives the accessible Git link and signed validation PDF. The completed form and private reviewer details are not committed to Git. | Final repository/version link and peer-validation PDF submitted privately on Canvas |
| Interpretation and limitations | Explains learning, coefficient units, baseline limitations, a confusion matrix, and observed threshold trade-offs using actual results. Distinguishes probability from label, association from causation, and classroom performance from clinical readiness. Documents genuine challenges, investigation, resolutions or design checks, and sources. | Written report with the required calculations, tables, and research-process section |

## How work will be evaluated

- **Complete and correct:** requirements are implemented correctly, evidence is
  reproducible, and interpretations match the submitted results.
- **Partially complete:** the central approach is sound, but requirements,
  calculations, outputs, or explanations are missing or contain identifiable
  errors. Credit reflects the correct work demonstrated.
- **Needs substantial revision:** major logic or evaluation errors undermine
  the results, such as reversed class meanings, test leakage, an incorrect
  gradient, or threshold selection on the test set.
- **Missing or unverifiable:** a component is absent or there is insufficient
  evidence to establish that it was completed correctly.

An honest explanation of a limitation is better than an unsupported claim.
You are not graded on achieving the highest accuracy, matching every reference
decimal, or making the selected threshold outperform 0.50 on the test set.
However, explanations do not replace a correct, working implementation.

These professional and independent-replication expectations continue the
practice established in Project 1: Linear Regression. The course replication
policy on the validation form also applies. Optional extensions should be
identified in the Canvas note and kept separate from the required experiment;
trying every algorithm is not required.
