# K-Nearest Neighbors (KNN) Lab

Start with the [visual beginner guide](KNN_Beginner_Guide.pdf), then work through
the [comparison notebook](KNN_Compared_with_Logistic_Regression_and_Decision_Trees.ipynb).
The guide begins with the **Iris flower dataset** bundled in scikit-learn.
Inputs are **petal length and petal width in centimeters**; the target is the
**species name**. A measurement sketch explains exactly what the numbers mean.
Five real reference flowers (zero-based dataset rows 54, 106, 50, 119, 133)
illustrate versicolor versus virginica; row 51 is excluded from the references
and used as the query. These deliberately selected rows teach the arithmetic,
not overall Iris classifier performance. The full dataset has three species,
including setosa, which is not included in this small example.
It includes rendered mathematical notation, two fully worked distance
calculations, a worked neighbor vote and scaling example, then a student
exercise, model-building explanation, and interpretation practice. The student
PDF retains the worked teaching examples but omits the exercise answer key.
Before programming, the guide explicitly connects distance sorting to selecting
the first k neighbors and counting their labels. It explains the optional vote
percentage, includes fit/predict pseudocode, and distinguishes KNN's noniterative
fitting from validation-based hyperparameter selection and final testing.
Pages 12-21 add a clearly **hypothetical 85-million-row flower case study**:
five fake records illustrate a full 51-million training / 17-million validation /
17-million test split, followed by refitting on 68 million references. Invented
candidate scores show exactly how k is selected; a confusion matrix explains
accuracy, species-specific precision, recall, and F1 in words as well as numbers.
The case distinguishes fitting from tuning, diagnoses overfitting, and covers
batch processing, exact versus approximate neighbor search, and practical
memory/latency limits. Batches do not restrict eligible neighbors or update
weights. No 85-million-row dataset is supplied or executed by this builder or
the notebook. KNN cannot discover an unknown species count; clustering plus
expert review is suggested for that different task.
Rebuilding also creates a local `KNN_Beginner_Guide_Instructor_Copy.pdf` with the
key, visibly labeled **Instructor Copy** on every page. The repository's
`*Instructor*` ignore rule excludes that PDF from Git.

The notebook remains a guided, self-checking lab with worked solutions and saved
outputs; it is not an answer-free assessment.

## Homework

After the lesson, complete [From Measurements to Neighbor Votes](homework_instructions.md)
and review the [student rubric](homework_rubric.md). Printable copies are available
as the [homework PDF](homework_instructions.pdf) and [rubric PDF](homework_rubric.pdf).
The homework includes an end-to-end hand calculation of training means,
population standard deviations, scaling, distances, and votes using six
invented flowers. A separate real-Iris coding experiment checks leakage-free
k selection, final refitting, baseline comparison, and multiclass metrics.
Large-data questions cover search costs and batching without requiring a huge
dataset. Follow the established course portfolio, Windows/macOS documentation,
independent peer-validation, and private Canvas submission requirements.
These student materials contain no answer key.

The notebook compares a majority baseline, logistic regression, a shallow
decision tree, and validation-selected KNN on the same rows. It uses Project 2's
two breast-cancer features and label convention: **0 = benign, 1 = malignant**.
This is a classroom example, not a medical diagnostic tool.

## Open and run

In VS Code, open the notebook and select your course Python environment in the
kernel picker. If prompted, enable notebook support and install the kernel
package in that environment. The notebook includes saved outputs and six plots
so you can review it before running code. Run the cells from top to bottom. The notebook
includes instructions, predictions to make before running code, interpretation
sentence starters, and extension exercises. No dataset download is needed.

Install the existing course dependencies if needed. From this folder:

```powershell
python -m pip install -r ..\assignment_requirements.txt
```

On macOS, use `python3` and `../assignment_requirements.txt`.
Allow approximately 60-90 minutes, including discussion.

## Lesson sequence

1. Distinguish petal measurements (inputs) from the known flower species (target).
2. Follow worked distance and voting examples before calculating distances and
   comparing 1, 3, and 5 neighbors yourself.
3. Understand training-only scaling and what KNN retains.
4. Select k using validation rows; keep test rows untouched.
5. Inspect neighbors, decision boundaries, and train/validation curves.
6. Refit using training plus validation rows and evaluate once on the test set.
7. Explain false positives, false negatives, and model limitations.

KNN tuning uses validation accuracy in this demonstration, not a clinical
objective. Logistic regression and tree settings are fixed comparison models,
not equally tuned competitors. Results do not establish a universally best
algorithm.

## Rebuild the PDF

Run the [builder](build_knn_guide.py) from this folder:

```powershell
python build_knn_guide.py
```

It writes both 24-page editions beside itself and checks page/card text bounds.
Equations use Matplotlib's built-in Mathtext rendering for square roots,
superscripts, and stacked fractions; no external LaTeX installation is needed.
