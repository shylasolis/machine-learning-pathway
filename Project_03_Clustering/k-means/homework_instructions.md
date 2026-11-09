# K-means Homework: Can We Justify These Groups?

**Format:** Individual portfolio investigation.

**Estimated effort:** 4-6 hours plus independent peer validation.

## Purpose and scope

Demonstrate the full path from measurements to centroids and from cluster
scores to a defensible interpretation. Use the [guide](KMeans_Beginner_Guide.pdf),
[lab](kmeans_lab.py), and [notebook](KMeans_Customer_Segmentation.ipynb) as cited
references, not as a completed submission. No customer data collection or
85-million-row execution is required.

## Part 1: Hand calculations before code

These are four invented customers. Inputs are monthly orders and average
items per order, not known customer segment labels.

| Customer | Monthly orders | Average items per order |
|---|---|---|
| H | 1 | 1 |
| I | 3 | 2 |
| J | 8 | 6 |
| L | 10 | 8 |

Use k = 2, initial center 0 = H (1, 1), and center 1 = J (8, 6).
Use raw features for this hand exercise only. An equal-distance tie goes to
the smaller center index. Do not round until reporting four decimal places.

1. Explain why this is unsupervised and how k differs from KNN's k.
2. Calculate each customer's squared Euclidean distance to both initial
   centers. Show all substitutions for I. Assign all four customers.
3. Calculate each updated centroid by averaging each assigned feature.
   Repeat assignment/update until centers do not change. Show a table for
   every iteration, not only the final answer.
4. Calculate total squared error before and after the first update, then at
   stopping. Explain why averaging improves the objective for fixed assignments.
5. Assign new customer T = (3, 1) using the final centers, without refitting.
   Does assigning T update a center? Is the distance a membership probability?
6. Can reaching max_iter alone prove convergence? Explain why different
   initializations may produce different results and why cluster 0 is not a
   known semantic class.

Calculate a separate scaling example from training rows (2,10), (2,50),
(8,10), (8,50), where feature 1 is monthly orders and feature 2 is dollars.
Compute both training means and population standard deviations by hand
(divide squared deviations by 4, not 3), then scale query (8,50).
Explain why these statistics prepare the geometry but do not replace
assigning records to centroids. Do not mix these rows with H,I,J,L.

## Part 2: Reproducible coding investigation

Use the supplied deterministic `make_customers(seed=42)` generator:
600 synthetic records, two named features, no target. You may reuse/cite the
generator and lab helpers, but write your own documented end-to-end script
and explain your decisions.

1. Report features/units, row count, missingness, and descriptive ranges.
   Explain that the simulation is invented and deliberately easy.
2. Use the lab's exact two-stage seed-42 index split: 20% test, then 25% of
   development for validation, preserving returned order. Verify disjointness
   and coverage. State why labels are not available to stratify.
3. Fit StandardScaler on training only. Fit full KMeans for k in [2,3,4,5,6],
   using init="k-means++", n_init=10, max_iter=300, tol=1e-4,
   random_state=42, algorithm="lloyd".
4. For each k, report training mean squared distance, validation mean squared
   distance, validation silhouette, and n_iter_. Plot compactness and silhouette
   against k. Select highest validation silhouette; exact score ties choose
   smaller k. Do not select by smallest training inertia.
5. For the selected k, fit five training-only, single-initialization models
   with seeds 0-4 and n_init=1. Compare their predictions on the same validation
   rows using pairwise adjusted Rand index. Explain what this stability check
   does and does not establish; it has no true customer labels.
6. Freeze k and preprocessing choices. Refit a new scaler and full KMeans on
   the original 480 development indices, then score the 120 test records once.
7. Fit MiniBatchKMeans using the same fixed development scaler and k, with
   batch_size=64, n_init=1, random_state=42, and five caller-controlled shuffled
   passes of partial_fit, as in the lab. Compare test mean squared distance and
   silhouette without using test results to retune anything.
8. Explain why five passes are a computational budget, not proof of
   convergence, and why partial_fit does not enforce a global max_iter loop.
   Full KMeans n_init means full restarts; MiniBatchKMeans initialization
   selection is not the same as ten complete mini-batch training runs.
9. Inverse-transform full-model centroids into orders/month and dollars,
   report development cluster sizes, and propose cautious descriptive names.
   Cluster IDs are arbitrary: do not compare the two models by raw ID equality.

### Required files

Your script must regenerate these outputs relative to its own location.
Use decimal metrics with at least six places and integer counts/settings.

| File | Required content |
|---|---|
| reports/k_selection.csv | Five rows: k,train_mean_squared_distance,validation_mean_squared_distance,validation_silhouette,n_iter |
| reports/test_metrics.csv | Two rows: model,mean_squared_distance,silhouette; model values full_kmeans and streamed_minibatch |
| reports/cluster_profiles.csv | One row per final full-model center: cluster,development_count,orders_per_month,average_basket_dollars |
| reports/k_selection.png | Both compactness and silhouette curves, labeled |
| reports/test_clusters.png | Both test assignments plotted in original units with development centers |

Also print the chosen k, five-seed pairwise ARI summary, and full-model n_iter_.
No particular k or score earns credit by itself. Explain results you actually
observe, including equal scores if they occur.

## Part 3: Interpretation and scale

1. Interpret your test mean squared distance: what is being averaged and in
   what coordinate system? Why is it not accuracy? Why does total inertia
   depend on dataset size and typically decrease with more centers?
2. For one hypothetical record with a = 0.4 and b = 1.0, calculate its
   silhouette (b-a)/max(a,b). What do a and b represent? Interpret scores near
   1, near 0, and below 0. Are these ground-truth correctness probabilities?
3. Explain why stable compact groups could still be useless or harmful, and
   why outliers, unequal densities, curved groups, and categorical codes can
   mislead K-means. Suggest a different method for one failure case.
4. Hypothetically, 85 million rows, two float64 features, k=3, 20 full passes:
   calculate raw-feature memory and point-to-center comparisons per pass
   and across 20 passes. Then account for ten complete KMeans restarts.
   Distinguish comparisons from individual coordinate arithmetic operations.
5. With 100,000 rows per chunk, how many chunks make one 85-million-row pass?
   Contrast full Lloyd accumulation (fixed centers during a pass; global sums
   and counts; update after the pass) with mini-batch updates after each chunk.
6. Describe two-pass global scaler fitting/transformation, representative
   initialization, a bounded streaming budget plus a development-only stopping
   check, and how to sample silhouette instead of constructing a huge pairwise
   distance matrix. K-means compares to k centers, not all other records.
7. Is a train/validation/test split compulsory for purely descriptive
   clustering? Explain why this lab uses one for future assignment assessment.
   If customers repeat or deployment is in the future, what must change?
8. Explain how to assign a new customer without retraining and how to monitor
   drift, changing cluster sizes, distances, and downstream usefulness.

## Report and professional portfolio

Submit a report of approximately 700-1,000 words excluding hand calculations,
tables, and figures. Address every interpretation question; include Part 1's
complete hand work and show how your chosen k follows the declared rule.
Include a short research-process section outside the word count: real
challenges/design decisions, investigation, corrections, evidence, and sources
(including course code and any assistance). Do not invent problems.

Use your own professional Git repository with a descriptive title such as
"Retail Patterns: An Unsupervised Customer Segmentation Study". Include
your script, dependency file, report, required outputs, meaningful commits,
and an appropriate .gitignore. Do not commit environments, credentials,
private customer records, or signed validation forms.

Your root README must be self-contained for a beginner on both Windows and
macOS: Python/Git installation links and verification; opening a terminal;
actual clone URL and directory commands; separate OS-appropriate environment,
dependency, and run commands; a direct interpreter option if PowerShell
activation is blocked; tested versions; expected outputs, approximate runtime,
and how to open them; and troubleshooting. Do not assume VS Code or depend
on the author's absolute paths or private help.

## Independent peer validation and Canvas submission

Follow the existing [course replication form](../../assignment-validation.pdf)
and its policy. A classmate must obtain a fresh copy of the final repository,
follow only your README on their own computer without your help, and verify
that all outputs regenerate. Record the actual OS, Python version, branch,
and final successful commit. Record/fix failures and repeat independent
validation before both parties sign. Do not claim untested platforms.

Submit an accessible Git URL with branch and validated commit, the signed
verification PDF privately on Canvas, and a note identifying optional
extensions or required-scope-only completion. Keep the form and private reviewer
details out of Git. A ZIP or screenshot is not a substitute. Extra tuning
must remain separate and must not use the test results.

Check the [student rubric](homework_rubric.md) before submitting.
