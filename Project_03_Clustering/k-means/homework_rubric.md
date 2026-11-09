# Student Rubric: K-means

Use with the [homework instructions](homework_instructions.md).
This rubric has no point weights or reference answers.

| Area | Complete work demonstrates |
|---|---|
| Unsupervised reasoning | Features versus unavailable targets, K-means versus KNN, arbitrary cluster IDs, learned centers versus chosen settings |
| Hand assignment and update | All squared distances, correct assignments, coordinate means, every iteration, objective before/after, stopping, and new-point assignment without an update |
| Scaling | Both population means/standard deviations calculated from training rows, query transformation, and distinction between preprocessing and clustering |
| Experimental design | Reproducible synthetic data, source/units, exact split, disjointness/coverage, train-only preprocessing, frozen choices, fresh development refit, untouched test until final evaluation |
| Choosing k and stability | All five candidates, correct metrics, declared silhouette/tie rule, five-seed label-invariant ARI comparison, interpretation without ground-truth claims |
| Full versus mini-batch | Correct fixed scaler, streaming updates/budget, fair data/setting comparison, distinction between partial_fit and fit stopping, and initialization semantics |
| Profiles and output artifacts | Required CSV columns/rows/precision, counts, labeled plots, inverse-scaled centers and cautious profiles, runnable script with robust paths |
| Metrics and limitations | Inertia/mean distance/silhouette interpreted with correct denominators, limitations of internal scores, shapes/outliers/feature coding, sensible alternatives |
| Scale reasoning | Verified memory/comparison/chunk counts, all-center search versus all-pairs, full-pass versus mini-batch updates, global scaler, sampled evaluation and drift |
| Portfolio and independent replication | Professional Git repository, complete beginner Windows/macOS README, dependencies/report/output files, peer-only reproduction, final commit evidence, private Canvas form |
| Research process | Honest claims, genuine design/debug checks, sources and assistance acknowledged, no fabricated problems |

## Evaluation

Complete and correct work is reproducible and explains the evidence.
Partially complete work earns credit for correct reasoning and implementation
demonstrated. Major errors such as test-driven k selection undermine conclusions;
an output table alone does not demonstrate a correct process. Missing or
unverifiable components cannot establish completion.

No particular accuracy, winning k, silhouette threshold, or mini-batch advantage
is required. A negative result with sound reasoning is preferable to a false
success claim. The existing replication policy applies.
