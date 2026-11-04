# Project 2: Classification

Classification predicts a category from input measurements. Logistic regression
is one approach, not the whole subject. Start with the shared foundations,
then open the folder for the algorithm you are studying.

## Shared foundations

- [Classification foundations lecture](Classification_Foundations_Lecture.pptx)
- [Lecture follow-along, Part 1](Classification-ppt-follow-read-along-Part-1.docx)
- [Lecture follow-along, Part 2](Classification-ppt-follow-read-along-Part-2%20.docx)

The foundations cover targets, algorithm selection, preprocessing, data leakage,
class imbalance, metrics, thresholds, calibration, and error analysis.

## Algorithm lessons and assignments

| Topic | Materials |
|---|---|
| [Logistic regression](logistic-regression/README.md) | Slides, likelihood worksheet, interactive explorer, [assignment](logistic-regression/assignment_instructions.md), and [student rubric](logistic-regression/assignment_rubric.md) |
| [K-nearest neighbors (KNN)](k-nearest-neighbors/README.md) | Visual guide, comparison notebook, [homework](k-nearest-neighbors/homework_instructions.md), and [student rubric](k-nearest-neighbors/homework_rubric.md) |

Additional algorithm lessons will be linked when they are ready for students.
The KNN notebook compares KNN, logistic regression, a decision tree, and a
baseline; it does not require a separate decision-tree lesson folder.

## Dependencies and reproduction

The [shared assignment dependencies](assignment_requirements.txt) serve both
the logistic regression assignment and the KNN lab/homework, so they remain
at this level. Each algorithm's README gives its setup and run instructions.
The logistic regression GUI also has a smaller local dependency file.

Assignments continue the course's professional portfolio expectations:
use a descriptive Git repository, self-contained beginner Windows/macOS
instructions, and independent peer reproduction without author assistance.
Submit the repository/version link and signed validation PDF privately on
Canvas; never commit the completed form or private reviewer information.
Consult each assignment for its exact experimental design and deliverables.
