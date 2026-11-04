# Student Rubric: KNN Homework

**Portfolio investigation:** From Measurements to Neighbor Votes

Use this rubric with the [homework instructions](homework_instructions.md).
Evidence must demonstrate both a working implementation and your understanding.
This student rubric lists expectations without point weights or an answer key.

## Review criteria

| Area | Expectations for complete work | Evidence reviewed |
|---|---|---|
| KNN concepts and limitations | Distinguishes features, labels, queries, k, scaler fitting, and KNN storage. Explains distance selection, votes, noniterative fitting, tuning, and prediction stopping. Recognizes that known-class classification does not discover new species. | Part 1 explanations and algorithm discussion |
| Training statistics and scaling | Computes both feature means and population standard deviations from the six references only, with deviations and squared deviations. Applies the same unrounded statistics to every reference and Q. Distinguishes scaling from classification. | Part 2 arithmetic and seven-row scaled table |
| Distances and votes | Shows D's complete scaled-distance calculation, all six sorted distances, voters and labels for k = 1 and 3, and the winning fraction. Computes raw distances and compares neighbor order. Explains probability limitations, unknown-label accuracy, and multiclass ties. | Hand calculations, sorted tables, vote counts, and optional arithmetic-only Python check |
| Dataset and leakage prevention | Uses real Iris with only petal length/width and original species coding. Documents source, units, missingness, class counts, specified stratified split, index disjointness/coverage, and partition roles. Keeps fictional records separate. | Data-loading code, assertions, printed partition counts, and report |
| Validation-based k selection | Fits training-only scaler/KNN pipelines for all seven prescribed k values. Uses fixed Euclidean/uniform settings, records both accuracies, plots them, and selects highest validation accuracy with smaller-k score tie-break before testing. Explains observed gaps and library tie behavior. | Pipeline code, seven-row CSV, selection plot, chosen k, and interpretation |
| Final refit and baseline | Freezes k, fits a new scaler/KNN pipeline on development rows in the prescribed order, and fits the most-frequent baseline on the same rows. Evaluates both once on held-out test rows without retuning. Distinguishes old validation scores from the refitted model. | Final-fit/evaluation code and report |
| Measured metrics and outputs | Supplies exactly the required CSV row shapes and columns, metric units/precision, integer support, and labeled plots/matrices. Correctly reports accuracy, macro-F1, and per-species metrics with an explained undefined-metric convention. Interprets actual errors and baseline comparisons. | Three CSVs, two PNGs, console tables, and report |
| Hypothetical metrics by hand | Correctly derives total/correct/error counts, accuracy, virginica one-versus-rest confusion counts, precision, recall, and F1. Uses correct denominators and explains practical error meanings and distinction from macro-F1. | Part 6A substitutions and interpretation |
| Large-data reasoning | Identifies eligible references before/after refit, leakage risk, exhaustive comparison and batch counts, reusable neighbor prefixes, global top-k merging, and batching without weight updates. Distinguishes global chunked scaling, repeated fit replacement, exact indexes, approximate search, and practical limits. | Part 6B reasoning and calculations |
| Professional portfolio and reproducibility | Provides a descriptive repository, readable runnable script, dependencies, complete report/calculations, five generated outputs, and meaningful commits. README is self-contained for beginners on Windows/macOS, including installation, clone/setup/run commands, versions, outputs, and troubleshooting. | Repository, README, files, and commit history |
| Independent validation and private submission | A classmate reproduces the final version on their own computer using only the README without author assistance. Problems are documented, fixed, and independently retested. Canvas receives the accessible repository/version link and signed form; private form/reviewer details stay out of Git. | Final validated commit and privately submitted validation PDF |
| Interpretation and research process | Uses results rather than unsupported claims; separates hypothetical and measured evidence, explains limitations and refit changes, documents genuine investigation/checks, and acknowledges sources and assistance. | Report and research-process section |

## How work will be evaluated

- **Complete and correct:** calculations, code, artifacts, and explanations meet
  the instructions and can be independently reproduced.
- **Partially complete:** the main approach is sound, but evidence is missing
  or contains identifiable mistakes. Credit reflects correct work demonstrated.
- **Needs substantial revision:** errors undermine the conclusions, such as
  fitting a scaler on test data, selecting k using test accuracy, using labels
  as features, or confusing means with votes.
- **Missing or unverifiable:** a component is absent or there is insufficient
  evidence that it works or that you understand it.

No particular test accuracy or selected k is required. Honest limitations are
better than inflated claims, but explanations cannot replace correct work.
Do not submit instructor answers or the guided notebook's saved output as
evidence of your own experiment. The replication form's course policy applies.
