from __future__ import annotations

import csv
import math
import shutil
from collections import Counter
from pathlib import Path

import kagglehub

DATASET_ID = "noordeen/insurance-premium-prediction"
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATA_DIR = PROJECT_ROOT / "data" / "insurance-premium-prediction"
CSV_PATH = PROJECT_DATA_DIR / "insurance.csv"


def ensure_local_dataset() -> Path:
	"""Download from kagglehub cache and keep a project-local copy."""
	cached_path = Path(kagglehub.dataset_download(DATASET_ID))

	if PROJECT_DATA_DIR.exists():
		shutil.rmtree(PROJECT_DATA_DIR)
	shutil.copytree(cached_path, PROJECT_DATA_DIR)
	return cached_path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		return list(reader)


def describe_numeric(values: list[float]) -> dict[str, float]:
	count = len(values)
	mean = sum(values) / count
	variance = sum((v - mean) ** 2 for v in values) / count
	std = math.sqrt(variance)
	return {
		"count": float(count),
		"min": min(values),
		"max": max(values),
		"mean": mean,
		"std": std,
	}


def pearson_correlation(x: list[float], y: list[float]) -> float:
	n = len(x)
	x_mean = sum(x) / n
	y_mean = sum(y) / n
	num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
	den_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
	den_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
	if den_x == 0 or den_y == 0:
		return 0.0
	return num / (den_x * den_y)


def correlation_strength_label(r: float) -> str:
	"""Map |r| to a plain-language strength label for students."""
	abs_r = abs(r)
	if abs_r < 0.10:
		return "very weak"
	if abs_r < 0.30:
		return "weak"
	if abs_r < 0.50:
		return "moderate"
	if abs_r < 0.70:
		return "strong"
	return "very strong"


def correlation_direction_label(r: float) -> str:
	if r > 0:
		return "positive"
	if r < 0:
		return "negative"
	return "no linear"


def print_ascii_histogram(values: list[float], title: str, bins: int = 8, width: int = 28) -> None:
	"""Print a simple text histogram so students can visualize spread in terminal-only environments."""
	if not values:
		print(f"{title}: <no values>")
		return

	v_min = min(values)
	v_max = max(values)
	if v_min == v_max:
		print(f"{title}: all values are {v_min:.2f}")
		return

	step = (v_max - v_min) / bins
	counts = [0 for _ in range(bins)]
	for v in values:
		idx = min(int((v - v_min) / step), bins - 1)
		counts[idx] += 1

	peak = max(counts)
	print(title)
	for i, count in enumerate(counts):
		left = v_min + i * step
		right = left + step
		bar_len = int((count / peak) * width) if peak else 0
		bar = "#" * bar_len
		print(f"  [{left:6.2f}, {right:6.2f}) {bar} ({count})")


def print_sorted_sample(x: list[float], y: list[float], x_name: str, y_name: str, points: int = 12) -> None:
	"""Show a compact, sorted sample of paired points as a text-based scatter preview."""
	if not x or not y:
		return
	pairs = sorted(zip(x, y), key=lambda p: p[0])
	step = max(1, len(pairs) // points)
	print(f"Sampled data points ({x_name} -> {y_name}) [text-based scatter sample]:")
	for xi, yi in pairs[::step][:points]:
		print(f"  {x_name}={xi:6.2f} -> {y_name}={yi:10.2f}")


def print_pipeline_outline() -> None:
	"""Print an ordered lesson outline for script 1."""
	print("Pipeline outline (script 1):")
	print("  1) Read data: load insurance.csv so we know what we are modeling.")
	print("  2) Inspect data: shape, columns, summaries, and category counts.")
	print("  3) Relationship check: compute correlations with expenses.")
	print("  4) Visual intuition: show histograms and sampled points.")
	print("  5) Outcome: this script does not train yet; it prepares us for training scripts.")
	print("  6) If patterns look weak/noisy: add more features or transformations before training.")
	print()


def main() -> None:
	print_pipeline_outline()

	# --------------------------------------------------------------------------
	# Step 1: Ensure the dataset is copied from kagglehub cache into this repo.
	# --------------------------------------------------------------------------
	cached_path = ensure_local_dataset()
	rows = load_rows(CSV_PATH)

	# Convert selected columns to floats for descriptive stats and correlations.
	age = [float(r["age"]) for r in rows]
	bmi = [float(r["bmi"]) for r in rows]
	children = [float(r["children"]) for r in rows]
	expenses = [float(r["expenses"]) for r in rows]
	smoker_counts = Counter(r["smoker"] for r in rows)
	sex_counts = Counter(r["sex"] for r in rows)
	region_counts = Counter(r["region"] for r in rows)

	print("Cached dataset path:", cached_path)
	print("Project dataset path:", PROJECT_DATA_DIR)
	print("CSV path:", CSV_PATH)
	print()

	print("Dataset shape:", f"{len(rows)} rows x {len(rows[0]) if rows else 0} columns")
	print("Columns:", ", ".join(rows[0].keys()) if rows else "<none>")
	print()

	print("Numeric summary:")
	for name, values in {
		"age": age,
		"bmi": bmi,
		"children": children,
		"expenses": expenses,
	}.items():
		stats = describe_numeric(values)
		print(
			f"  {name:8s} count={int(stats['count'])}, min={stats['min']:.2f}, "
			f"max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}"
		)
	print()

	print("Category counts:")
	print("  sex:", dict(sex_counts))
	print("  smoker:", dict(smoker_counts))
	print("  region:", dict(region_counts))
	print()

	age_corr = pearson_correlation(age, expenses)
	bmi_corr = pearson_correlation(bmi, expenses)
	children_corr = pearson_correlation(children, expenses)

	print("Correlation with expenses:")
	print(f"  age      -> expenses: {age_corr:.4f}")
	print(f"  bmi      -> expenses: {bmi_corr:.4f}")
	print(f"  children -> expenses: {children_corr:.4f}")
	print()

	print("How to interpret these numbers:")
	print("  sign: positive means both tend to increase together; negative means opposite directions")
	print("  size: 0.00-0.09 very weak, 0.10-0.29 weak, 0.30-0.49 moderate, 0.50+ strong")
	print()

	print("Student interpretation summary:")
	print(
		f"  age: {age_corr:.4f} -> {correlation_strength_label(age_corr)} "
		f"{correlation_direction_label(age_corr)} linear relationship"
	)
	print(
		f"  bmi: {bmi_corr:.4f} -> {correlation_strength_label(bmi_corr)} "
		f"{correlation_direction_label(bmi_corr)} linear relationship"
	)
	print(
		f"  children: {children_corr:.4f} -> {correlation_strength_label(children_corr)} "
		f"{correlation_direction_label(children_corr)} linear relationship"
	)
	print("  note: correlation is not causation, and weak single-feature correlation can still help in multi-feature models")
	print()

	# Text visuals help students "see" distributions without plotting libraries.
	print("ASCII stands for American Standard Code for Information Interchange.")
	print("Here, ASCII simply means the chart is drawn with plain keyboard characters (like #).")
	print()
	print("Histogram note: each bar is a count of data points inside an equal-width value interval (bin).")
	print("A taller bar means more observations fell into that bin range.")
	print("The final section is a sampled data table (not a real image plot).")
	print()
	print_ascii_histogram(bmi, "ASCII histogram for BMI")
	print()
	print_ascii_histogram(expenses, "ASCII histogram for Expenses")
	print()
	print_sorted_sample(bmi, expenses, "bmi", "expenses")


if __name__ == "__main__":
	main()