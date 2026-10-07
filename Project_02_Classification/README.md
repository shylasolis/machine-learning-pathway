# Project 2: Classification

**Status: foundation lecture and logistic-regression masterclass ready.** The dataset, guided lab, and final
assignment have not been selected or implemented yet.

This project follows [Project 1: Linear Regression](../Project_01_Linear_Regression/)
and applies [Module 2 foundations](../Module_02_Preprocessing_Models_Diagnostics/).

## Start with the lecture

- [Classification foundations PowerPoint](Classification_Foundations_Lecture.pptx):
  38 slides with speaker notes, class questions, expected answers, cautions,
  and official documentation sources.
- [Classification foundations slide PDF](Classification_Foundations_Lecture.pdf):
  a static version for reading and sharing.

The lecture covers when to use classification; binary, multiclass, and multilabel
targets; algorithm selection for tables, text, and images; model-specific
preprocessing; built-in capabilities and their limits; image/mask preparation;
leakage, imbalance, metrics, thresholds, calibration, and error analysis.

Suggested delivery is two sessions: slides 1-19 for foundations and algorithm
selection, then slides 20-36 for preparation and diagnostics. Slides 37-38
contain clickable references. Practical model families were checked against
official documentation on October 4, 2026; they are not a universal ranking.

The deck contains original editable diagrams and clearly labeled simulated
inspection pictures. These are teaching illustrations, not real dataset
examples, measured results, or the proposed assignment data.

An instructor preparation PDF is stored separately in the locally available
`Project_02_Classification_Instructor` directory. That directory remains
excluded from Git under the existing instructor-material policy.

## Continue with logistic regression

- [Logistic regression masterclass PowerPoint](Logistic_Regression_Masterclass.pptx):
  58 slides with detailed speaker notes, class prompts, worked calculations,
  colorful diagrams, mathematical plots, and clickable sources.
- [Logistic regression masterclass slide PDF](Logistic_Regression_Masterclass.pdf):
  a static version for reading and sharing.

The progression covers intuition and vocabulary; linear scores, sigmoid,
odds and log-odds; likelihood, loss, gradients, convexity, and regularization;
preprocessing, solvers, interpretation, thresholds, and calibration; then an
executed UCI Bank Marketing example and production workflows, monitoring,
image embeddings, and safe release.

Suggested delivery is three 60-75 minute sessions: slides 1-21, 22-38, and
39-56. Slides 57-58 contain references. The separate instructor guide provides
all slide talking points, derivations, common misconceptions, measured results,
and reproduction code.

The bank example uses the original 45,211-row `bank-full.csv` with a chronological
holdout. Results are measured for this lecture, not a published benchmark or
a production-ready recommendation. Inspection pictures remain explicitly
simulated. This lecture example does **not** select or change the image-project
dataset; the guided lab and assignment remain undecided.

## Next: guided lab and assignment

The direction under consideration is a real-world image classification problem
with meaningful image preprocessing. Pixel-level segmentation may be included
as an extension if the selected dataset has suitable annotations and the
workload fits student hardware.

Before implementation, confirm:

- Dataset access, licensing, raw-data suitability, and annotation quality.
- Business question, classification labels, and optional segmentation targets.
- Preprocessing and leakage-safe train/validation/test strategy.
- Model scope, compute requirements, and appropriate evaluation metrics.

This is an applied project, not "Module 3"; Module 3 remains FastAPI and Docker
containerization in the syllabus.
