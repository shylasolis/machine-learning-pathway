"""Reproducible, label-free retail clustering and a small Lloyd walkthrough."""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TOY = np.array([[1., 1.], [1., 3.], [3., 2.], [7., 8.], [9., 7.], [9., 9.]])
TOY_IDS = list("ABCDEF")
INITIAL_CENTERS = TOY[:2].copy()
FEATURES = ["orders_per_month", "average_basket_dollars"]
CANDIDATES = [2, 3, 4, 5, 6]
COLORS = ["#187a78", "#c65324", "#7954a1", "#2465aa", "#ad3866", "#617330"]


def squared_distances(x, centers):
    """Only for the small classroom arrays, not a huge N by k allocation."""
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def lloyd_trace(x, initial_centers, max_iter=20, tol=1e-10):
    """Return full assignment/update steps; reject empty clusters explicitly."""
    x = np.asarray(x, dtype=float)
    centers = np.asarray(initial_centers, dtype=float).copy()
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("Data and centers must be 2-D with matching feature counts.")
    if len(x) < len(centers) or len(centers) < 1:
        raise ValueError("Use at least one center and at least as many rows as centers.")
    if not np.isfinite(x).all() or not np.isfinite(centers).all():
        raise ValueError("Data and centers must contain finite values.")
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter must be positive and tol nonnegative.")
    trace = []
    for iteration in range(1, max_iter + 1):
        distances = squared_distances(x, centers)
        labels = distances.argmin(axis=1)
        if len(np.unique(labels)) != len(centers):
            raise ValueError("Empty cluster in teaching implementation; choose different seeds.")
        updated = np.array([x[labels == j].mean(axis=0) for j in range(len(centers))])
        before = distances[np.arange(len(x)), labels].sum()
        after = ((x - updated[labels]) ** 2).sum()
        shift = float(np.linalg.norm(updated - centers))
        trace.append(dict(
            iteration=iteration, centers_before=centers.copy(), distances=distances,
            labels=labels.copy(), centers_after=updated.copy(),
            inertia_before=float(before), inertia_after=float(after), shift=shift,
            stopped=shift <= tol,
        ))
        centers = updated
        if shift <= tol:
            break
    return trace


def make_customers(seed=42):
    """Invent 600 records; component membership is not an observed target."""
    rng = np.random.default_rng(seed)
    blocks = [
        rng.normal([2., 25.], [.7, 5.], size=(200, 2)),
        rng.normal([8., 75.], [1.4, 8.], size=(200, 2)),
        rng.normal([15., 35.], [1.8, 6.], size=(200, 2)),
    ]
    x = np.maximum(np.vstack(blocks), [.2, 1.])
    return x[rng.permutation(len(x))]


def partition_indices(n):
    indices = np.arange(n)
    development, test = train_test_split(indices, test_size=.20, random_state=42)
    train, validation = train_test_split(development, test_size=.25, random_state=42)
    return train, validation, test, development


def new_model(k, seed=42, n_init=10):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                  max_iter=300, tol=1e-4, random_state=seed, algorithm="lloyd")


def select_k(x_train, x_validation):
    records = []
    for k in CANDIDATES:
        model = new_model(k).fit(x_train)
        labels = model.predict(x_validation)
        silhouette = silhouette_score(x_validation, labels)
        records.append(dict(
            k=k, train_mean_squared_distance=model.inertia_ / len(x_train),
            validation_mean_squared_distance=-model.score(x_validation) / len(x_validation),
            validation_silhouette=float(silhouette), n_iter=int(model.n_iter_),
        ))
    selected = min(records, key=lambda row: (-row["validation_silhouette"], row["k"]))
    return records, selected["k"]


def stream_minibatches(x, k, batch_size=64, epochs=5):
    """Caller owns passes and stopping for partial_fit, unlike estimator.fit."""
    if batch_size < k or epochs < 1 or len(x) < k:
        raise ValueError("Use a batch size and data size >= k and at least one epoch.")
    model = MiniBatchKMeans(n_clusters=k, n_init=1, batch_size=batch_size,
                            random_state=42, compute_labels=False)
    rng = np.random.default_rng(42)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            model.partial_fit(x[order[start:start + batch_size]])
    return model


def evaluate(model, x):
    labels = model.predict(x)
    # Direct residuals also work for partial_fit models without full-data inertia_.
    residuals = x - model.cluster_centers_[labels]
    return dict(mean_squared_distance=float((residuals ** 2).sum() / len(x)),
                silhouette=float(silhouette_score(x, labels)))


def run_experiment():
    x = make_customers()
    train, validation, test, development = partition_indices(len(x))
    scaler = StandardScaler().fit(x[train])
    x_train = scaler.transform(x[train])
    x_validation = scaler.transform(x[validation])
    selection, k = select_k(x_train, x_validation)
    # Same validation rows, independently seeded single fits; no true labels.
    stability_labels = [
        new_model(k, seed=seed, n_init=1).fit(x_train).predict(x_validation)
        for seed in range(5)
    ]
    stability = np.array([
        adjusted_rand_score(a, b) for a, b in combinations(stability_labels, 2)
    ])
    final_scaler = StandardScaler().fit(x[development])
    x_development = final_scaler.transform(x[development])
    x_test = final_scaler.transform(x[test])
    final = new_model(k).fit(x_development)
    mini = stream_minibatches(x_development, k)
    test_results = [
        dict(model="full_kmeans", **evaluate(final, x_test)),
        dict(model="streamed_minibatch", **evaluate(mini, x_test)),
    ]
    return dict(
        x=x, train=train, validation=validation, test=test, development=development,
        scaler=scaler, selection=selection, k=k, stability=stability,
        final_scaler=final_scaler, final=final, mini=mini,
        x_development=x_development, x_test=x_test, test_results=test_results,
    )


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, destination):
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "k_selection.csv", result["selection"])
    save_csv(output / "test_metrics.csv", result["test_results"])
    centers = result["final_scaler"].inverse_transform(result["final"].cluster_centers_)
    assigned = result["final"].predict(result["x_development"])
    profiles = [
        dict(cluster=j, development_count=int((assigned == j).sum()),
             orders_per_month=float(center[0]), average_basket_dollars=float(center[1]))
        for j, center in enumerate(centers)
    ]
    save_csv(output / "cluster_profiles.csv", profiles)
    trace = lloyd_trace(TOY, INITIAL_CENTERS)
    save_csv(output / "toy_iterations.csv", [
        dict(iteration=t["iteration"], inertia_before=t["inertia_before"],
             inertia_after=t["inertia_after"], center_shift=t["shift"],
             stopped=t["stopped"]) for t in trace
    ])
    fig, axes = plt.subplots(1, len(trace), figsize=(4 * len(trace), 3.7), squeeze=False)
    for ax, step in zip(axes[0], trace):
        for j in range(2):
            rows = TOY[step["labels"] == j]
            ax.scatter(rows[:, 0], rows[:, 1], color=COLORS[j])
        ax.scatter(*step["centers_before"].T, marker="X", s=150, c="black")
        for name, row in zip(TOY_IDS, TOY):
            ax.annotate(name, row + .12)
        ax.set(title=f"Assign: iteration {step['iteration']}",
               xlabel="Orders per month", ylabel="Average items per order",
               xlim=(0, 10), ylim=(0, 10))
    save_figure(fig, output / "toy_iterations.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ks = [r["k"] for r in result["selection"]]
    for key, label in [
        ("train_mean_squared_distance", "Training"),
        ("validation_mean_squared_distance", "Validation"),
    ]:
        axes[0].plot(ks, [r[key] for r in result["selection"]], "o-", label=label)
    axes[0].set(xlabel="k (centroids)", ylabel="Mean squared distance (scaled units)",
                title="Compactness: lower is not automatically better")
    axes[0].legend()
    axes[1].plot(ks, [r["validation_silhouette"] for r in result["selection"]], "o-")
    axes[1].set(xlabel="k (centroids)", ylabel="Validation silhouette",
                title=f"Predeclared rule selects k = {result['k']}", ylim=(-.1, 1))
    save_figure(fig, output / "k_selection.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model, name in zip(
        axes, [result["final"], result["mini"]], ["Full K-means", "Streamed MiniBatchKMeans"]
    ):
        labels = model.predict(result["x_test"])
        raw_test = result["x"][result["test"]]
        for j in range(result["k"]):
            rows = raw_test[labels == j]
            ax.scatter(rows[:, 0], rows[:, 1], color=COLORS[j], label=f"Cluster {j}")
        raw_centers = result["final_scaler"].inverse_transform(model.cluster_centers_)
        ax.scatter(*raw_centers.T, c="black", marker="X", s=150, label="Development centers")
        ax.set(xlabel="Orders per month", ylabel="Average basket ($)", title=name)
        ax.legend(fontsize=7)
    save_figure(fig, output / "test_clusters.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    raw = result["x"][result["development"]]
    raw_model = new_model(result["k"]).fit(raw)
    for ax, labels, title in zip(
        axes, [raw_model.labels_, assigned],
        ["Without scaling (development only)", "With training-fitted scaling"],
    ):
        for j in range(result["k"]):
            points = raw[labels == j]
            ax.scatter(*points.T, s=12, color=COLORS[j])
        ax.set(title=title, xlabel="Orders per month", ylabel="Average basket ($)")
    save_figure(fig, output / "scaling_comparison.png")
    print(f"Selected k: {result['k']}; five-seed pairwise ARI mean: {result['stability'].mean():.6f}")
    print(f"Final full K-means iterations: {result['final'].n_iter_}")
    for row in result["test_results"]:
        print(row)
    print(f"Wrote 4 CSV files and 4 PNG figures to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    write_outputs(run_experiment(), args.output)


if __name__ == "__main__":
    main()
