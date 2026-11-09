# K-means: From Customers to Centroids

Our first unsupervised-learning lesson asks: **Can shopping measurements suggest
useful groups of customers when we do not already have segment labels?**
Unlike KNN classification, K-means does not ask known neighbors to vote.
It learns k representative centers by repeatedly assigning and averaging.
Each feature is an input column. With two input columns and k=2, the model
learns two center points, each described by two numbers (one per column).
That is two coordinate pairs, not four centers.

The fitting objective is **Within-Cluster Sum of Squares (WCSS)**: total squared
distance to assigned centers. **J** is its mathematical symbol; scikit-learn
calls the fitted total **inertia** (`inertia_`). The guide connects WCSS to
regression's MSE and logistic regression's BCE, while distinguishing ordinary
KNN's stored-example fitting and validation-based tuning from loss optimization.

## Materials

| Item | Purpose |
|---|---|
| [Beginner guide](KMeans_Beginner_Guide.pdf) | 38 visual pages: MSE/BCE/WCSS and KNN comparison, worked arithmetic, error-total and center-movement walkthroughs, practice, scaling, elbow and silhouette explanations, stability, usefulness, stopping/validation, and large-data reasoning |
| [Guided notebook](KMeans_Customer_Segmentation.ipynb) | Predict-before-running prompts, complete arithmetic, visualizations, and self-checking lab outputs; not an answer-free assessment |
| [Runnable lab](kmeans_lab.py) | Reproduces a six-customer Lloyd trace and a 600-record synthetic retail experiment |
| [Homework](homework_instructions.md) / [printable PDF](homework_instructions.pdf) | Independent hand work, coding, interpretations, and scale reasoning |
| [Student rubric](homework_rubric.md) / [printable PDF](homework_rubric.pdf) | Expectations without weights or answers |
| [Tests](test_kmeans_lab.py) | Numerical, split, preprocessing, objective, and output checks |
| [Material builder](build_kmeans_materials.py) | Rebuilds PDFs; optionally regenerates the notebook source |

Instructor editions, teaching notes, solutions, and a weighted grading guide
are saved separately in the repository's existing Git-ignored instructor
area. Student PDFs omit the independent practice solutions. Worked teaching
examples remain in the student guide intentionally.

## Data dictionary and boundaries

All records are **synthetic**, generated locally. There are no real customer
records, external datasets, accounts, API keys, or downloads.

- Hand example: six invented customers, monthly order count and average items
  per order. A value of 3 means three items per order, not a rating.
  Raw counts keep the first distance calculations visible; this does not
  imply that equal units or ranges guarantee appropriate feature weights.
- Coding case: 600 synthetic records, `orders_per_month` and
  `average_basket_dollars`. A value of 75 means an average $75 basket.
  No target column is returned or used. The simulation has deliberately clear
  structure; its easy scores are not promises about actual customers.
  Fractional order frequencies represent rates over a longer observation window.

The lab uses a reproducible 360/120/120 training/validation/test split. Training
fits the scaler and centers; validation selects k from 2-6 by the predeclared
highest-silhouette/smaller-k tie rule. The final model refits both preprocessing
and centers on 480 development rows, then evaluates 120 test rows once.
This models assigning **future similar customers**. Purely descriptive
clustering can instead fit all records, but then has no independent held-out
estimate. Real repeated customers require customer-group splits; future
deployment may require chronological holdouts.

Stability compares five independently seeded single fits on the same validation
rows using pairwise adjusted Rand index. It does not use customer ground truth.
The final mini-batch comparison uses the same fixed development scaler and k.
Five caller-controlled `partial_fit` passes are a budget, not proof of convergence.

## Setup and run

Use Python 3.10+ and a terminal at the **repository root**. Existing course
environments with the required packages can be reused. Otherwise:

### Windows PowerShell

```powershell
python -m venv .venv-kmeans
.\.venv-kmeans\Scripts\python.exe -m pip install -r .\Project_03_Clustering\k-means\requirements.txt
.\.venv-kmeans\Scripts\python.exe .\Project_03_Clustering\k-means\kmeans_lab.py
```

Calling the environment's interpreter directly avoids activation-policy changes.
Use `py -3` instead of `python` to create the environment if needed.

### macOS Terminal

```bash
python3 -m venv .venv-kmeans
.venv-kmeans/bin/python -m pip install -r Project_03_Clustering/k-means/requirements.txt
.venv-kmeans/bin/python Project_03_Clustering/k-means/kmeans_lab.py
```

Python is available from <https://www.python.org/downloads/> and Git from
<https://git-scm.com/downloads>. Verify with `python --version` (`python3` on
macOS) and `git --version`. To obtain the course repository, follow its GitHub
Code menu to copy the clone URL, run `git clone URL` with that actual URL,
then navigate into the downloaded repository. No VS Code installation is required
for the script.

### Notebook

Open the notebook in VS Code/Jupyter and select the same Python environment.
If notebook support is missing, install `ipykernel`, `nbformat`, and `nbclient`
in that environment. Start the notebook kernel with this folder as its working
directory, or run from the repository root; the first cell handles both.
Run all cells top to bottom. Saved outputs allow review before execution.
The notebook deliberately displays solutions after the practice prompt.

## Outputs and checks

The script writes beside itself, regardless of the terminal's directory:

- `reports/toy_iterations.csv` and `reports/toy_iterations.png`.
- `reports/k_selection.csv` and `reports/k_selection.png`.
- `reports/test_metrics.csv` and `reports/test_clusters.png`.
- `reports/cluster_profiles.csv` and `reports/scaling_comparison.png`.

CSVs open in a spreadsheet; PNGs open in an image viewer. Existing generated
files are overwritten by the next run. `--output PATH` selects another output
directory. Small-data runs should take seconds to a minute, depending on hardware.
No 85-million-row dataset is generated: that section is a cost illustration.

From this folder with the environment's Python:

```powershell
python -m unittest discover -s . -p "test_*.py"
python build_kmeans_materials.py
```

The PDFs use UOP orange (#E65300), black (#231F20), white, and neutral grays.
The PDF builder uses Matplotlib/Mathtext for readable superscripts, fractions,
and square roots rather than inline programming notation or external LaTeX. To explicitly
regenerate the notebook source, append `--notebook`; this replaces its saved
execution outputs, so only do so when intentionally rebuilding the notebook.
Keep instructor outputs local; the `*Instructor*` ignore rule excludes them.
The optional private teaching/grading PDF is rebuilt only when its local
instructor Markdown source is available; a student clone does not need it.

Tested runtime and measured results are recorded in the notebook and generated
reports. Seeds and explicit `n_init` improve reproducibility, but cluster IDs
are arbitrary and numerical/library changes can alter a near-tied solution.
Do not compare ID numbers across independent fits as if they were stable names.

If imports fail, install requirements with the interpreter actually running the
script/kernel. If a command cannot find a file, check its documented working
directory. If a notebook imports a different module, restart its kernel and
rerun from the top. Do not mix scaled coordinates with raw-dollar centroids.

## Teaching sequence

Allow two 75-90 minute meetings plus homework:

1. What changes from supervised to unsupervised learning; KNN versus K-means.
2. Read the six rows; assign, update, reassign, and calculate squared error.
3. Try a new hand problem before opening solutions.
4. Compute scaler statistics, apply fixed scaling, and choose k using evidence.
5. Interpret silhouettes, inertia, stability, cluster sizes, and original-unit
   profiles; learn why none is classification accuracy.
6. Trace stopping within a run versus restart selection versus choosing k.
7. Process large data, distinguish full-data blocks from mini-batch updates,
   and discuss drift, fairness, outliers, and method limitations.

## Sources

- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [MiniBatchKMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [Adjusted Rand index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

These are geometric segments, not causal findings, protected-group identities,
creditworthiness, or proof of customer needs. Review privacy, representativeness,
and the consequences of any proposed business action before use.
