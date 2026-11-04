"""Build the printable KNN lesson with the project's Matplotlib dependency."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.textpath import TextPath
from sklearn.datasets import load_iris


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "KNN_Beginner_Guide.pdf"
INSTRUCTOR_OUTPUT = ROOT / "KNN_Beginner_Guide_Instructor_Copy.pdf"
NAVY = "#17324d"
TEAL = "#187a78"
ORANGE = "#c65324"
PALE = "#eef4f7"
IRIS = load_iris()
REFERENCE_ROWS = [54, 106, 50, 119, 133]
QUERY_ROW = 51
POINTS = IRIS.data[REFERENCE_ROWS, 2:]
LABELS = IRIS.target[REFERENCE_ROWS]
QUERY = IRIS.data[QUERY_ROW, 2:]
NAMES = list("ABCDE")
PAGE_COUNT = 24
CASE_ROWS = [
    ["F001", 1.4, 0.2, "setosa", "Training"],
    ["F002", 4.7, 1.4, "versicolor", "Training"],
    ["F003", 5.5, 2.0, "virginica", "Training"],
    ["F004", 5.1, 1.8, "virginica", "Validation"],
    ["F005", 4.9, 1.5, "versicolor", "Test"],
]
CASE_K = np.array([1, 3, 7, 15, 51])
CASE_TRAIN_ACCURACY = np.array([1.00, 0.98, 0.96, 0.94, 0.86])
CASE_VALIDATION_SIZE = 17_000_000
CASE_VALIDATION_CORRECT = np.array([13_940_000, 14_790_000, 15_300_000, 15_130_000, 14_280_000])
CASE_TEST_MATRIX = np.array([[7_650_000, 170_000, 0], [85_000, 4_080_000, 935_000],
                             [85_000, 595_000, 3_400_000]])


def page(pdf, number, title, subtitle):
    edition = pdf.infodict()["Subject"]
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.08, 0.96, f"CLASSIFICATION / KNN / {edition.upper()}", fontsize=9,
            color=ORANGE, weight="bold", va="top")
    ax.text(0.08, 0.915, title, fontsize=22, color=NAVY, weight="bold", va="top")
    paragraph(ax, 0.08, 0.865, subtitle, size=10, color=TEAL)
    ax.plot([0.08, 0.92], [0.06, 0.06], color="#d3dfe5")
    ax.text(0.08, 0.038, "KNN: A Beginner's Guide | Classroom use only",
            fontsize=8, color=NAVY)
    ax.text(0.92, 0.038, f"{number} / {PAGE_COUNT}", ha="right", fontsize=8, color=NAVY)
    return fig, ax


def paragraph(ax, x, y, content, width=7.0, size=11, color=NAVY, bold=False):
    font = FontProperties(family="DejaVu Sans", size=size,
                          weight="bold" if bold else "normal")
    lines = []
    for block in content.split("\n"):
        line = ""
        for word in block.split():
            candidate = f"{line} {word}".strip()
            if TextPath((0, 0), candidate, prop=font).get_extents().width > width * 72:
                if not line:
                    raise ValueError(f"Word is too wide: {word}")
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    height = len(lines) * size * 1.4 / 792
    if x + width / 8.5 > 0.94 or y - height < 0.075:
        raise ValueError(f"Text exceeds page content bounds: {content[:60]}")
    artist = ax.text(x, y, "\n".join(lines), va="top", fontsize=size,
                     color=color, weight="bold" if bold else "normal",
                     linespacing=1.4)
    return artist, y - height


def card(ax, top, heading, content, height=0.13):
    ax.add_patch(FancyBboxPatch((0.08, top - height), 0.84, height,
                               boxstyle="round,pad=0.008", facecolor=PALE,
                               edgecolor="#d3dfe5"))
    paragraph(ax, 0.10, top - 0.017, heading, size=11, bold=True, width=6.7)
    _, bottom = paragraph(ax, 0.10, top - 0.05, content, width=6.7, size=10.5)
    if bottom < top - height + 0.008:
        raise ValueError(f"Text does not fit its card: {heading}")


def equation(ax, y, expression, size=17):
    ax.text(0.10, y, f"${expression}$", va="top", fontsize=size, color=NAVY)


def case_table(fig, bottom, height, headers, rows, widths):
    ax = fig.add_axes([0.09, bottom, 0.82, height])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.85)
    return table


def finish(pdf, fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for artist in ax.texts:
            bounds = artist.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise ValueError(f"Text outside page: {artist.get_text()[:60]}")
    pdf.savefig(fig)
    plt.close(fig)


def labels_page(pdf):
    fig, ax = page(pdf, 1, "First: what are we predicting?",
                   "Identify an Iris flower's species from its petal measurements. A label is its known species.")
    paragraph(ax, 0.08, 0.79,
              "The famous Iris dataset has 150 flowers and three species: setosa, versicolor, and virginica. Scikit-learn includes it as load_iris(). For a manageable hand example, we select five real flowers from just versicolor and virginica.")
    card(ax, 0.67, "Inputs: two measurements in centimeters (cm)",
         "Petal length measures base to tip. Petal width measures across the widest part. A length of 4.6 means 4.6 cm, not a score. The next page shows exactly what we mean.", height=0.12)
    table_ax = fig.add_axes([0.09, 0.32, 0.82, 0.20])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=[[name, str(point[0]), str(point[1]), IRIS.target_names[label], str(row)]
                  for name, point, label, row in zip(NAMES, POINTS, LABELS, REFERENCE_ROWS)],
        colLabels=["Flower", "Length (cm)", "Width (cm)", "Known species", "Row ID"],
        colWidths=[0.12, 0.20, 0.20, 0.32, 0.16], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)
    card(ax, 0.29, "Output: the species name",
         "The known species is the target or class label, not a third measurement. Row IDs are zero-based positions in sklearn's dataset, not features. We select these rows to teach the arithmetic, not to measure model quality.", height=0.13)
    paragraph(ax, 0.08, 0.12,
              "Our query flower has length 4.5 cm and width 1.5 cm (row 51, excluded from the five references). Hide its known species until after prediction.", size=10)
    finish(pdf, fig)


def measurements_page(pdf):
    fig, ax = page(pdf, 2, "What exactly do we measure?",
                   "A petal is a colored part of a flower. This simplified sketch shows one petal, not a whole flower.")
    paragraph(ax, 0.08, 0.79,
              "For flower A, the dataset records a petal length of 4.6 cm and a petal width of 1.5 cm. Imagine placing a ruler along each direction below. These are physical sizes, not ratings or points.")
    sketch = fig.add_axes([0.19, 0.35, 0.62, 0.33])
    sketch.set(xlim=(-2, 2), ylim=(-0.7, 5.3))
    sketch.set_aspect("equal")
    sketch.axis("off")
    sketch.add_patch(Ellipse((0, 2.3), 1.5, 4.6, facecolor="#e5d7ed", edgecolor="#773d9b", linewidth=2))
    sketch.annotate("", (1.3, 4.6), (1.3, 0), arrowprops={"arrowstyle": "<->", "color": NAVY})
    sketch.text(1.48, 2.3, "Length: 4.6 cm", rotation=90, va="center", fontsize=11, color=NAVY)
    sketch.annotate("", (0.75, 2.3), (-0.75, 2.3), arrowprops={"arrowstyle": "<->", "color": NAVY})
    sketch.text(0, 2.7, "Width: 1.5 cm", ha="center", fontsize=10, color=NAVY)
    sketch.text(0, -0.35, "Base", ha="center", fontsize=10)
    sketch.text(0, 4.85, "Tip", ha="center", fontsize=10)
    card(ax, 0.28, "From a ruler to a point on a graph",
         "Flower A becomes the coordinate (4.6, 1.5): length first, width second. Our query flower becomes (4.5, 1.5). KNN compares these measurements to find similar reference flowers.", height=0.14)
    paragraph(ax, 0.08, 0.10, "Illustrative measurement sketch, not a botanical drawing of an actual specimen.", size=9)
    finish(pdf, fig)


def voting_page(pdf):
    fig, ax = page(pdf, 3, "A prediction by nearby examples",
                   "KNN means k-nearest neighbors. k is how many neighbors vote, not how many classes exist.")
    paragraph(ax, 0.08, 0.79,
              "Each colored point is a reference flower from page 1. Position shows petal length and width; color and shape show known species. The star is the query at (4.5, 1.5). The plot is zoomed in; both axes use equal centimeter spacing.")
    plot = fig.add_axes([0.17, 0.34, 0.68, 0.34])
    for label, color, marker, name in [(1, TEAL, "o", "versicolor"),
                                       (2, ORANGE, "^", "virginica")]:
        plot.scatter(*POINTS[LABELS == label].T, c=color, marker=marker, s=100, label=name)
    plot.scatter(*QUERY, marker="*", c=NAVY, s=230, label="Query: hide species")
    for name, point in zip(NAMES, POINTS):
        plot.annotate(name, point, xytext=(8, 5), textcoords="offset points")
    for point in POINTS[:3]:
        plot.plot([QUERY[0], point[0]], [QUERY[1], point[1]], "--", color="#78909c", zorder=0)
    plot.set(xlim=(4.3, 5.5), ylim=(1.1, 2.1), xlabel="Petal length (cm)",
             ylabel="Petal width (cm)")
    plot.set_aspect("equal")
    plot.grid(alpha=0.2)
    plot.legend(fontsize=8, loc="upper right")
    card(ax, 0.26, "With k = 3, use A, B, and C",
         "A and C are versicolor; B is virginica. Two versicolor votes beat one virginica vote, so predict versicolor. Do not average species codes as though they were measurements. Next, we calculate the distances.", height=0.14)
    finish(pdf, fig)


def distance_page(pdf):
    fig, ax = page(pdf, 4, "Watch two distance calculations",
                   "Euclidean distance means straight-line distance. We show every step before you try one.")
    paragraph(ax, 0.08, 0.79,
              "The query is (4.5, 1.5). Here x is a reference flower's petal length and y is its petal width, both in cm. d is distance in this two-measurement plane. Species is NOT used in this calculation.", size=10.5)
    equation(ax, 0.71, r"d=\sqrt{(x-4.5)^2+(y-1.5)^2}", size=20)
    paragraph(ax, 0.08, 0.65,
              "A square multiplies a number by itself: 0.1 squared is 0.01. A square root reverses squaring. Keep the calculator's full precision when ordering distances; round only for display.", size=10)
    paragraph(ax, 0.08, 0.57, "Worked example 1: A is at (4.6, 1.5)", bold=True)
    paragraph(ax, 0.10, 0.53, "Subtract: 4.6 minus 4.5 is 0.1; 1.5 minus 1.5 is 0.", size=10)
    equation(ax, 0.49, r"d_A=\sqrt{(4.6-4.5)^2+(1.5-1.5)^2}", size=17)
    equation(ax, 0.43, r"d_A=\sqrt{0.1^2+0^2}=\sqrt{0.01}=0.1", size=17)
    paragraph(ax, 0.10, 0.375, "A is 0.1 cm away in the length-width plane.", size=10)
    paragraph(ax, 0.08, 0.315, "Worked example 2: C is at (4.7, 1.4)", bold=True)
    paragraph(ax, 0.10, 0.275, "Subtract: 4.7 minus 4.5 is 0.2; 1.4 minus 1.5 is -0.1.", size=10)
    equation(ax, 0.235, r"d_C=\sqrt{0.2^2+(-0.1)^2}=\sqrt{0.04+0.01}", size=17)
    equation(ax, 0.175, r"d_C=\sqrt{0.05}\approx 0.224", size=17)
    paragraph(ax, 0.10, 0.115, "A negative difference becomes positive when squared. A is closer than C.", size=10)
    finish(pdf, fig)


def worked_vote_page(pdf):
    fig, ax = page(pdf, 5, "Watch the neighbors vote",
                   "First calculate distances. Then sort. Only then use the known species to vote.")
    paragraph(ax, 0.08, 0.79,
              "Here are all five distances to the query flower at (4.5, 1.5), sorted from nearest to farthest. Distances use raw centimeters for this small arithmetic demonstration.")
    table_ax = fig.add_axes([0.09, 0.49, 0.82, 0.21])
    table_ax.axis("off")
    distances = np.linalg.norm(POINTS - QUERY, axis=1)
    table = table_ax.table(
        cellText=[[name, IRIS.target_names[LABELS[index]], f"{distances[index]:.3f}",
                   "YES" if rank < 3 else "NO"]
                  for rank, index in enumerate(np.argsort(distances))
                  for name in [NAMES[index]]],
        colLabels=["Flower", "Known species", "Distance (cm)", "Votes at k = 3?"],
        colWidths=[0.12, 0.32, 0.27, 0.29], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    card(ax, 0.45, "Worked vote: choose k = 3",
         "Take the first three rows: A, B, C. Their species are versicolor, virginica, versicolor. Two versicolor votes beat one virginica vote. Predict versicolor. D and E do not vote.", height=0.13)
    paragraph(ax, 0.08, 0.27, "How much of the vote was for versicolor?", bold=True)
    equation(ax, 0.22, r"\mathrm{Versicolor\ vote\ fraction}=\frac{2}{3}\approx 0.667=66.7\%", size=17)
    paragraph(ax, 0.08, 0.14,
              "Two of three chosen neighbors are versicolor. This is a neighbor fraction, not guaranteed real-world confidence. We deliberately selected five references; this is not an accuracy study of all 150 flowers.", size=10)
    finish(pdf, fig)


def distance_to_vote_page(pdf):
    fig, ax = page(pdf, 6, "Why distances lead to votes",
                   "Distance chooses who participates. The selected neighbors' labels choose the prediction.")
    card(ax, 0.79, "1. Calculate, sort, and select",
         "Calculate distances to the reference flowers and sort smallest to largest. With k = 3, keep ONLY the first three: A, B, C. D and E are farther away and do not participate in this prediction.", height=0.13)
    card(ax, 0.63, "2. Count the known species",
         "We want a species, not a distance. 'Voting' means counting labels: A and C are versicolor; B is virginica. Two versicolor votes beat one virginica vote. Return versicolor, then stop.", height=0.13)
    card(ax, 0.47, "Why not use only the single closest flower?",
         "That is exactly k = 1. A small neighborhood can be less sensitive to one noisy or unusual reference. More neighbors do not guarantee better predictions; choose k using validation data.", height=0.12)
    card(ax, 0.32, "What is the 66.7% for?",
         "It is optional extra information: 2 of 3 chosen neighbors are versicolor. Counting 2 versus 1 already gives the prediction. Report: 'Predicted versicolor; 2 of its 3 nearest references are versicolor.'", height=0.12)
    paragraph(ax, 0.08, 0.16,
              "The percentage does NOT feed back into training, adjust measurements, or start another iteration in our workflow. It is not necessarily a calibrated real-world probability. With equal votes, A and C each count once despite their different distances. Distance-weighted KNN uses a different rule.", size=10)
    finish(pdf, fig)


def pseudocode_page(pdf):
    fig, ax = page(pdf, 7, "The algorithm before Python",
                   "Pseudocode describes the steps in plain language. This is classification with equal-weight votes.")
    card(ax, 0.79, "Fit: prepare the reference information",
         "INPUT: training measurements, known labels, and a chosen k.\n1. Learn any preprocessing statistics from training rows only.\n2. Apply that preprocessing to the training measurements.\n3. Store the processed measurements and their labels.\n4. Fitting is complete; a library may also build a search index.", height=0.20)
    card(ax, 0.54, "Predict: process one new row",
         "1. Apply the saved preprocessing to the new measurements.\n2. Calculate distances to the stored training examples.\n3. Sort from smallest distance to largest.\n4. Select the first k examples.\n5. Count how many selected examples belong to each class.\n6. Return the class with the most votes. Prediction is complete.\n7. Optionally return each class's count divided by k.", height=0.25)
    paragraph(ax, 0.08, 0.24,
              "This describes a straightforward distance search. Libraries can use optimized searches without sorting every reference explicitly. The prediction still depends on the selected k nearest neighbors.", size=10)
    paragraph(ax, 0.08, 0.16,
              "The hand example uses raw centimeters; the main notebook saves and applies a scaler. The unknown row's true species is never an input to prediction. A complete implementation also needs rules for tied votes and equal-distance neighbors; our worked example has neither tie.", size=10)
    finish(pdf, fig)


def training_and_selection_page(pdf):
    fig, ax = page(pdf, 8, "When does KNN stop training?",
                   "Basic KNN has no repeated coefficient updates and no loss-minimization loop like logistic regression.")
    card(ax, 0.79, "Fitting is not the same as selecting k",
         "Logistic regression repeatedly adjusts coefficients to reduce a loss until its solver stops. Basic KNN retains examples and labels; fitting ends after preparing/storing this reference data and any search structure. It is often called lazy learning: much of the work happens at prediction time.", height=0.16)
    card(ax, 0.59, "Model selection: a separate loop in our notebook",
         "FOR each k in the specified candidate list:\n    Fit preprocessing and KNN using TRAINING rows.\n    Predict VALIDATION rows and measure validation accuracy.\nChoose the k with the highest validation accuracy.\nIf scores tie, our notebook chooses the smaller k.\nSTOP after all specified candidate values have been evaluated.", height=0.21)
    paragraph(ax, 0.08, 0.33,
              "Maximizing validation accuracy is equivalent to minimizing its fraction of incorrect predictions. This selects a hyperparameter, not coefficients. It does not keep changing the reference measurements until every error disappears.", size=10)
    card(ax, 0.24, "Freeze settings, refit, then test",
         "After choosing k, fit a fresh model on training + validation rows. Evaluate it once on untouched TEST rows. Do not select k using test results. Test performance estimates new-data performance only when future data is sufficiently similar.", height=0.14)
    finish(pdf, fig)


def practice_page(pdf):
    instructor = pdf.infodict()["Subject"] == "Instructor Copy"
    fig, ax = page(pdf, 9, "Your turn: calculate, then vote",
                   "Follow the worked examples on pages 4-5. Show your reasoning before checking with your instructor.")
    card(ax, 0.78, "1. Calculate two distances to (4.5, 1.5)",
         "B is at (4.5, 1.7). D is at (5.0, 1.5). For each one: subtract coordinates, square the differences, add them, and take the square root. Show all steps.", height=0.13)
    paragraph(ax, 0.10, 0.61, "B: ______________________________________________________\n\nD: ______________________________________________________", size=11)
    card(ax, 0.49, "2. Change the number of voters",
         "Use the sorted table on page 5. Predict for k = 1 and k = 5. Count each species's votes. Does the prediction change from the worked k = 3 example?", height=0.12)
    paragraph(ax, 0.10, 0.33, "My votes and predictions: __________________________________", size=11)
    if instructor:
        paragraph(ax, 0.08, 0.265, "Answer key - Instructor Copy", bold=True, color=TEAL)
        equation(ax, 0.225, r"d_B=\sqrt{0^2+0.2^2}=\sqrt{0.04}=0.2", size=15)
        equation(ax, 0.175, r"d_D=\sqrt{0.5^2+0^2}=\sqrt{0.25}=0.5", size=15)
        paragraph(ax, 0.10, 0.12,
                  "k = 1: A votes versicolor. k = 3: versicolor wins 2 to 1. k = 5: virginica wins 3 to 2. Changing k changes the voters. The hidden true species of row 51 is versicolor; k = 5 is wrong here.", size=9.5)
    else:
        paragraph(ax, 0.08, 0.265, "Explain your reasoning", bold=True, color=TEAL)
        paragraph(ax, 0.10, 0.22,
                  "Which reference flowers vote for each choice of k?\n\n________________________________________________________\n\nWhy can the prediction change when the flower has not changed?\n\n________________________________________________________", size=10.5)
    finish(pdf, fig)


def model_page(pdf):
    fig, ax = page(pdf, 10, "How is this a trained model?",
                   "Basic KNN retains training examples; it does not learn logistic coefficients or decision-tree rules.")
    cards = [
        ("1. Split before preprocessing",
         "Training rows build the model. Validation rows help select k. Reserve test rows for one final evaluation."),
        ("2. Fit the scaler on training rows",
         "The scaler puts features on comparable scales using training means and standard deviations. The next page shows the arithmetic. Never fit it on test rows."),
        ("3. Fit KNN",
         "Retain the scaled training features and known labels, with settings such as k, distance metric, and vote weighting."),
        ("4. Predict a new row",
         "Use the saved scaler, find k training neighbors, and vote. Do not refit the scaler on the new row or add its unknown label to training."),
        ("5. Interpret the vote fraction carefully",
         "With equal weights, 2 versicolor votes out of 3 gives a vote fraction of two thirds. It is not proof of the species or a guaranteed calibrated probability."),
    ]
    for top, (heading, content) in zip([0.79, 0.65, 0.51, 0.37, 0.23], cards):
        card(ax, top, heading, content, height=0.12)
    finish(pdf, fig)


def scaling_page(pdf):
    fig, ax = page(pdf, 11, "Watch a scaling calculation",
                   "A separate invented example: these numbers are NOT the course dataset's fitted statistics.")
    paragraph(ax, 0.08, 0.79,
              "Features can have different units and spreads. Standardization measures how far a value is above or below the training mean, in training standard-deviation units.")
    equation(ax, 0.69, r"z=\frac{\mathrm{value}-\mathrm{training\ mean}}{\mathrm{training\ standard\ deviation}}", size=18)
    card(ax, 0.59, "Given: summaries already learned from training rows",
         "Feature 1: training mean = 10, training standard deviation = 2.\nFeature 2: training mean = 20, training standard deviation = 5.\nNew row: feature 1 = 14; feature 2 = 25.", height=0.14)
    paragraph(ax, 0.08, 0.40, "Substitute, subtract, then divide", bold=True)
    equation(ax, 0.35, r"z_1=\frac{14-10}{2}=\frac{4}{2}=2", size=20)
    equation(ax, 0.27, r"z_2=\frac{25-20}{5}=\frac{5}{5}=1", size=20)
    paragraph(ax, 0.08, 0.18,
              "The raw row (14, 25) becomes (2, 1). Feature 1 is two standard deviations above its training mean; feature 2 is one above. KNN calculates distances between scaled rows using these same saved training summaries.", size=10.5)
    finish(pdf, fig)


def case_objective_page(pdf):
    fig, ax = page(pdf, 12, "Case study: 85 million flowers",
                   "HYPOTHETICAL: all large-scale counts, records, scores, and results in this case are invented.")
    paragraph(ax, 0.08, 0.79,
              "Imagine a flower-monitoring company has 85,000,000 labeled records. For this story, experts have confirmed three known species: setosa, versicolor, virginica. We predict the species of a new flower from petal length and width in centimeters.", size=10.5)
    case_table(fig, 0.49, 0.19,
               ["Record", "Length (cm)", "Width (cm)", "Expert species"],
               [row[:4] for row in CASE_ROWS], [0.16, 0.23, 0.23, 0.38])
    paragraph(ax, 0.08, 0.46,
              "These five fake records are a window into the larger data, not the whole training set. They are separate from the real Iris rows used earlier in the guide.", size=10)
    card(ax, 0.38, "What KNN IS for",
         "Supervised classification: assign a new flower to a class present in labeled references. Ordinary KNN still chooses a known class for an unfamiliar species; it does not automatically discover or reject new species, or explain biological causes.", height=0.12)
    card(ax, 0.23, "What if nobody knows how many species exist?",
         "KNN classification cannot discover the number of species. HDBSCAN clustering can suggest groups without fixing their count in advance, but groups are NOT proof of species. Experts must examine biology and confirm labels. Explore a representative sample first; 85 million rows still requires a scaling plan.", height=0.15)
    finish(pdf, fig)


def case_splits_page(pdf):
    fig, ax = page(pdf, 13, "Split before choosing anything",
                   "Training fits the model. Validation selects settings. Test is the final independent check.")
    paragraph(ax, 0.08, 0.79,
              "Assume each record is a different plant and the intended task is predicting similar new plants. Use a species-stratified 60/20/20 split. If plants have repeat records, keep each plant in one split; for future dates or new farms, use a matching time/group holdout instead.", size=10.5)
    case_table(fig, 0.53, 0.16, ["Pool", "Rows", "Purpose"],
               [["Training", "51,000,000", "Fit scaler + references"],
                ["Validation", "17,000,000", "Choose k; diagnose errors"],
                ["Test", "17,000,000", "Final evaluation only"]],
               [0.22, 0.25, 0.53])
    case_table(fig, 0.29, 0.19, ["Record", "Assigned pool", "How its label is used"],
               [[row[0], row[4], "Fit reference" if row[4] == "Training" else "Check prediction only"]
                for row in CASE_ROWS], [0.18, 0.24, 0.58])
    card(ax, 0.23, "Use ALL rows in this hypothetical, in their proper roles",
         "All 51 million training rows are references during tuning. All 17 million validation rows help choose k. All 17 million test rows stay locked until final evaluation. We display only five records; they do not replace the full data. Process queries in batches rather than loading every query-to-reference distance at once.", height=0.15)
    finish(pdf, fig)


def case_fit_page(pdf):
    fig, ax = page(pdf, 14, "What did fitting actually learn?",
                   "Prepare all 51 million training references; the five shown rows are only a preview.")
    paragraph(ax, 0.08, 0.79,
              "Check units, impossible values, duplicate plants, and label quality first. If imputation is needed, fit it using training rows only. Here all shown measurements are present. Suppose the training scaler learns these invented summaries:", size=10.5)
    equation(ax, 0.69, r"z_L=\frac{L-4.0}{1.5}\qquad z_W=\frac{W-1.0}{0.5}", size=20)
    paragraph(ax, 0.08, 0.62,
              "L and W are raw length and width in cm. Means are 4.0 and 1.0; standard deviations are 1.5 and 0.5. These summaries come from all 51 million training rows, NOT from the three displayed training rows.", size=10)
    scaled_rows = [[row[0], f"{(row[1] - 4.0) / 1.5:.3f}", f"{(row[2] - 1.0) / 0.5:.3f}", row[4]]
                   for row in CASE_ROWS]
    case_table(fig, 0.33, 0.20, ["Record", "Scaled length", "Scaled width", "Pool"],
               scaled_rows, [0.18, 0.26, 0.26, 0.30])
    paragraph(ax, 0.08, 0.31,
              "F005's values above are an illustration only; actual test transformation waits for the final frozen pipeline.", size=9)
    card(ax, 0.26, "What is learned, stored, and NOT learned?",
         "The scaler learns training means and spreads. Basic KNN stores scaled training measurements and labels, plus k and distance/voting rules; it may build a search index. It does not learn coefficients, a species-discovery rule, or a loss curve. F004 is transformed for validation, not stored as a neighbor during tuning. F005 stays locked.", height=0.17)
    finish(pdf, fig)


def case_select_k_page(pdf):
    fig, ax = page(pdf, 15, "Choose k using validation",
                   "HYPOTHETICAL scores: this is a worked selection example, not a result of running 85 million rows.")
    paragraph(ax, 0.08, 0.79,
              "Before comparing results, specify uniform votes, Euclidean distance, scaling, candidate k values, and validation accuracy as the selection metric. Smaller k breaks score ties. Each candidate uses the SAME 51 million training and 17 million validation rows.", size=10.5)
    case_table(fig, 0.47, 0.22, ["k", "Train accuracy", "Validation correct", "Val accuracy"],
               [[str(k), f"{train:.0%}", f"{correct:,}", f"{correct / CASE_VALIDATION_SIZE:.0%}"]
                for k, train, correct in zip(CASE_K, CASE_TRAIN_ACCURACY, CASE_VALIDATION_CORRECT)],
               [0.10, 0.27, 0.36, 0.27])
    card(ax, 0.40, "Decision: select k = 7",
         "15,300,000 of 17,000,000 validation flowers are correct: 90%. This is the highest validation accuracy among these candidates. Equivalently, 1,700,000 are wrong: a 10% validation error rate. Do NOT choose k = 1 just because its training score is 100%.", height=0.15)
    paragraph(ax, 0.08, 0.20,
              "Validation labels act as an answer sheet AFTER prediction, never as voting neighbors. F004 participates in validation scoring; F005's test label remains untouched. Choosing k is hyperparameter tuning, not coefficient training.", size=10)
    paragraph(ax, 0.08, 0.125,
              "Search once for 51 neighbors per query; evaluate the first 1, 3, 7, 15, or 51. This reuses distances. Accuracy is our predeclared objective; rare-species projects may preselect recall or macro-F1 instead.", size=9.5)
    finish(pdf, fig)


def case_overfitting_page(pdf):
    fig, ax = page(pdf, 16, "Overfitting and knowing when to stop",
                   "KNN can overfit even though it does not repeatedly adjust coefficients.")
    plot = fig.add_axes([0.15, 0.49, 0.72, 0.28])
    positions = np.arange(len(CASE_K))
    plot.plot(positions, CASE_TRAIN_ACCURACY, "o-", label="Training")
    plot.plot(positions, CASE_VALIDATION_CORRECT / CASE_VALIDATION_SIZE, "s-", label="Validation")
    plot.set(xticks=positions, xticklabels=CASE_K, xlabel="Number of neighbors (k)",
             ylabel="Accuracy", ylim=(0.75, 1.02), title="Invented full-data results (not a measured run)")
    plot.legend(fontsize=9)
    plot.grid(alpha=0.2)
    card(ax, 0.43, "Read the gap, not just the training score",
         "k = 1: 100% training vs 82% validation is an 18-percentage-point gap, a warning of noise sensitivity. A training row can be its own nearest neighbor. k = 7: 96% vs 90% is a smaller 6-point gap and better validation performance.", height=0.14)
    paragraph(ax, 0.08, 0.25,
              "k = 51 has a small 2-point gap but lower scores (86% / 84%): that may be underfitting. A small gap alone is not success. Gaps are clues, not proof; check group leakage, repeated splits, learning curves, and class-specific errors using development data.", size=10)
    paragraph(ax, 0.08, 0.15,
              "Fitting finishes when preprocessing and reference storage/index construction finish. Tuning finishes after all planned k values are scored and k = 7 is frozen. There is no 'keep reducing KNN loss until convergence' step. Further experiments use validation, not test results.", size=10)
    finish(pdf, fig)


def case_prediction_page(pdf):
    fig, ax = page(pdf, 17, "Follow F004 through a prediction",
                   "During validation, F004 is a query only. Its true species is hidden until after the vote.")
    equation(ax, 0.79, r"(5.1,\ 1.8)\ \longrightarrow\ \left(\frac{5.1-4.0}{1.5},\frac{1.8-1.0}{0.5}\right)\approx(0.733,\ 1.600)", size=15)
    paragraph(ax, 0.08, 0.71,
              "Search among ALL 51 million scaled training references. Keep the SEVEN closest; only those seven vote. An index may find them without scanning every reference. Here is an invented report of the neighbors selected for this query:", size=10)
    case_table(fig, 0.36, 0.29, ["Rank", "Reference ID", "Scaled distance", "Known species"],
               [[1, "N11", "0.03", "virginica"], [2, "N12", "0.07", "versicolor"],
                [3, "N13", "0.13", "virginica"], [4, "N14", "0.20", "virginica"],
                [5, "N15", "0.30", "versicolor"], [6, "N16", "0.33", "virginica"],
                [7, "N17", "0.45", "versicolor"]],
               [0.12, 0.25, 0.27, 0.36])
    card(ax, 0.29, "Return the winning species, then stop",
         "Virginica wins 4 votes to 3; no setosa reference is selected. Predict virginica. Its vote fraction is 4/7, about 57.1%: an optional description of neighbor agreement, not a training update or reliable calibrated probability. Odd k does not prevent every multiclass tie; this vote has no tie.", height=0.15)
    paragraph(ax, 0.08, 0.12,
              "Reveal F004's label: virginica. Count one correct prediction. Score ALL 17 million validation queries, not just this convenient example.", size=9.5)
    finish(pdf, fig)


def case_test_page(pdf):
    fig, ax = page(pdf, 18, "Freeze, refit, then open the test set",
                   "The final evaluation model has 68 million references. Test labels never become neighbors.")
    paragraph(ax, 0.08, 0.79,
              "Freeze k = 7 and all other settings. Fit a NEW scaler and KNN on the combined 51 million training + 17 million validation rows. F004 can now be a reference. The scaler's summaries may change; do not reuse the previous page's invented values blindly. F005 remains test-only.", size=10)
    paragraph(ax, 0.08, 0.68,
              "Apply the final saved pipeline to all 17 million locked test flowers, then compare predictions with expert labels once. F005 is just one of these rows. Here is an invented confusion matrix for the final model:", size=10)
    plot = fig.add_axes([0.24, 0.30, 0.54, 0.31])
    plot.imshow(CASE_TEST_MATRIX, cmap="Blues")
    species = ["setosa", "versicolor", "virginica"]
    for row in range(3):
        for col in range(3):
            plot.text(col, row, f"{CASE_TEST_MATRIX[row, col]:,}", ha="center", va="center",
                      color="white" if CASE_TEST_MATRIX[row, col] > 4_000_000 else NAVY, fontsize=8)
    plot.set(xticks=range(3), yticks=range(3), xticklabels=species, yticklabels=species,
             xlabel="Predicted species", ylabel="Expert (actual) species")
    plot.tick_params(labelsize=9)
    card(ax, 0.22, "Read one row: actual virginica",
         "Of 4,080,000 actual virginica flowers, 3,400,000 are predicted virginica, 595,000 versicolor, and 85,000 setosa. So 680,000 are missed. Off-diagonal cells are errors; diagonal cells are correct.", height=0.13)
    finish(pdf, fig)


def case_metrics_page(pdf):
    fig, ax = page(pdf, 19, "What do these scores mean?",
                   "Hypothetical test results: the denominators tell you which question each metric answers.")
    card(ax, 0.79, "Accuracy: 89.0% overall",
         "15,130,000 of 17,000,000 flowers are correct; 1,870,000 are wrong. This beats a 46% always-setosa baseline on this test set, but does not tell us which species is being missed.", height=0.12)
    card(ax, 0.64, "Virginica precision: 3,400,000 / 4,335,000 = 78.4%",
         "Of flowers CALLED virginica, about 78 in 100 really are virginica. There are 935,000 false positives. This matters when assigning the wrong species is costly. It does NOT mean 78% of all virginica flowers were found.", height=0.12)
    card(ax, 0.49, "Virginica recall: 3,400,000 / 4,080,000 = 83.3%",
         "Of actual virginica flowers, about 83 in 100 are detected; 680,000 are missed (false negatives). This matters when failing to find a species is costly. Precision and recall use different denominators.", height=0.12)
    card(ax, 0.34, "Virginica F1: 6,800,000 / 8,415,000 = 80.8%",
         "F1 balances precision and recall using their harmonic mean; it is not overall accuracy. Report every species. Macro-F1 averages the species F1 scores equally so common species do not dominate.", height=0.12)
    paragraph(ax, 0.08, 0.18,
              "Analyst statement: 'Overall accuracy is 89%, but we miss 680,000 virginica flowers and incorrectly call 935,000 other flowers virginica. Whether this is acceptable depends on the costs of those errors, not a universal passing score.'", size=10)
    paragraph(ax, 0.08, 0.115,
              "Do not change k because you dislike the test score. Improve using development data, then obtain a fresh final evaluation if test findings influenced your decisions.", size=9.5)
    finish(pdf, fig)


def case_scale_page(pdf):
    fig, ax = page(pdf, 20, "Do we really scan 85 million rows?",
                   "No fixed universal row limit. Practical limits are memory, search time, hardware, and implementation.")
    card(ax, 0.79, "Eligible references are not the same as rows examined",
         "During tuning, 51 million training rows are eligible neighbors. After refitting, 68 million are eligible. Brute force checks EVERY eligible reference per query. An exact tree-based index can prune candidates and still return exact neighbors, but may approach a full scan in difficult data.", height=0.15)
    card(ax, 0.61, "Why brute force gets expensive",
         "51 million references x 17 million validation queries = 867 trillion distance comparisons for one exhaustive pass. Reuse the nearest-51 results for all candidate k values. Batch queries and stream data; never allocate the full query-by-reference distance matrix.", height=0.14)
    paragraph(ax, 0.08, 0.43,
              "85 million rows x 2 float64 features x 8 bytes = 1.36 GB (decimal) for raw features alone. Labels, IDs, scaler copies, indexes, and search buffers add memory. Fitting in memory does NOT prove prediction will meet a response-time target.", size=10)
    card(ax, 0.32, "A realistic engineering decision, not a magic setting",
         "Benchmark exact indexes (especially plausible for only two features) against brute force. Approximate-neighbor indexes trade exactness for speed and can change votes: measure both neighbor-retrieval recall and classification metrics on validation. A smaller representative reference sample is another option, but changes the model and must be validated.", height=0.16)
    paragraph(ax, 0.08, 0.125,
              "After evaluation, you may refit on all 85 million rows for deployment. The old score evaluated 68 million references, not this new model. Those rows are no longer an independent test. Assess deployment on fresh labeled data and monitor drift and latency.", size=9)
    finish(pdf, fig)


def case_batches_page(pdf):
    fig, ax = page(pdf, 21, "Yes, use batches for large data",
                   "Batching controls memory and scheduling. It is NOT a KNN gradient-descent training loop.")
    card(ax, 0.79, "Load and preprocess in chunks",
         "Read the training pool in chunks, for example 1 million rows at a time. Training-only running statistics can produce one GLOBAL scaler (StandardScaler supports partial_fit). Then transform chunks with that saved scaler. Do not fit a different scaler for each chunk.", height=0.14)
    paragraph(ax, 0.08, 0.61,
              "Build or store a searchable representation of ALL 51 million scaled training references. Ordinary sklearn KNeighborsClassifier has no partial_fit: repeatedly calling fit on chunks replaces the references; it does not accumulate them. Large-scale storage/index construction requires suitable tooling.", size=10)
    case_table(fig, 0.36, 0.16, ["Validation batch", "Query rows", "Eligible reference set"],
               [["1", "1 - 10,000", "All 51 million training rows"],
                ["2", "10,001 - 20,000", "All 51 million training rows"],
                ["...", "...", "Same reference set"],
                ["1,700", "16,990,001 - 17,000,000", "Same reference set"]],
               [0.23, 0.30, 0.47])
    paragraph(ax, 0.08, 0.32,
              "For each query batch, find up to 51 neighbors per flower and score each candidate k. Add correct-prediction counts to running totals. Only after ALL 1,700 batches, divide each total by 17 million and select k. Do not choose a different k per batch or update weights.", size=10)
    paragraph(ax, 0.08, 0.20,
              "Exact brute-force search can also scan reference blocks and merge the nearest candidates across ALL blocks. Searching only the current block gives the wrong neighborhood. An index can reduce scanning; batching alone does not reduce the number of exhaustive comparisons.", size=10)
    paragraph(ax, 0.08, 0.12,
              "Bound internal search buffers too; query batch size alone is not a memory guarantee. After refitting, batch the test queries against all 68 million references without tuning on them.", size=9.5)
    finish(pdf, fig)


def code_page(pdf):
    fig, ax = page(pdf, 22, "Now connect the idea to code",
                   "The companion notebook runs the full experiment. This excerpt assumes training and validation arrays exist.")
    paragraph(ax, 0.08, 0.78,
              "The Iris hand example ends here. For the main code comparison, return to the course breast-cancer data: mean radius and mean texture, with 0 = benign and 1 = malignant. This is not for patient care.")
    code = (
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neighbors import KNeighborsClassifier\n\n"
        "knn = make_pipeline(\n"
        "    StandardScaler(),\n"
        "    KNeighborsClassifier(n_neighbors=5,\n"
        "                         weights=\"uniform\"),\n"
        ")\n\n"
        "knn.fit(X_train, y_train)\n"
        "predictions = knn.predict(X_validation)\n"
        "vote_fractions = knn.predict_proba(X_validation)"
    )
    ax.text(0.10, 0.65, code, family="DejaVu Sans Mono", fontsize=10.5,
            color=NAVY, va="top", linespacing=1.6)
    card(ax, 0.30, "Read the code in plain English",
         "make_pipeline connects scaling to KNN. fit learns the scaler and stores the training examples. predict applies both steps to new raw measurements. predict_proba returns fractions for the classes in classes_ order.", height=0.16)
    finish(pdf, fig)


def metrics_page(pdf):
    fig, ax = page(pdf, 23, "How well did it perform?",
                   "Count mistakes on unseen rows. The example below is illustrative, not a measured notebook result.")
    paragraph(ax, 0.08, 0.80, "Here we switch to the medical-data labels: 0 = benign, 1 = malignant.", size=10)
    plot = fig.add_axes([0.27, 0.53, 0.46, 0.23])
    matrix = np.array([[40, 5], [8, 17]])
    plot.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            plot.text(col, row, str(matrix[row, col]), ha="center", va="center",
                      fontsize=18, color="white" if matrix[row, col] > 25 else NAVY)
    plot.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"],
             yticklabels=["0", "1"], xlabel="Predicted class", ylabel="Actual class")
    paragraph(ax, 0.08, 0.47,
              "Of 70 rows, 40 + 17 are correct. For class 1 there are 17 correct detections, 8 missed detections, and 5 false alarms.", size=10)
    equations = [
        (0.40, r"\mathrm{Accuracy}=\frac{\mathrm{correct}}{\mathrm{all}}=\frac{40+17}{70}\approx 81.4\%"),
        (0.33, r"\mathrm{Precision}=\frac{\mathrm{correct\ class\ 1}}{\mathrm{predicted\ class\ 1}}=\frac{17}{17+5}\approx 77.3\%"),
        (0.26, r"\mathrm{Recall}=\frac{\mathrm{detected\ class\ 1}}{\mathrm{actual\ class\ 1}}=\frac{17}{17+8}=68.0\%"),
        (0.19, r"\mathrm{F1}=\frac{2\times17}{2\times17+5+8}\approx 72.3\%"),
    ]
    for y, expression in equations:
        equation(ax, y, expression, size=15)
    paragraph(ax, 0.08, 0.135, "F1 balances precision and recall using their harmonic mean.", size=9)
    paragraph(ax, 0.08, 0.105,
              'Interpretation: "The model missed 8 of 25 actual malignant rows. Accuracy alone hides these misses."', size=9)
    finish(pdf, fig)


def cautions_page(pdf):
    fig, ax = page(pdf, 24, "What to watch for",
                   "KNN is easy to picture, but a meaningful idea of closeness is essential.")
    items = [
        ("Choosing k",
         "Small k is sensitive to noise; k = 1 can memorize training rows. Very large k can blur patterns and favor the majority class. Select k using validation, not test scores."),
        ("Missing values and categories",
         "sklearn KNN does not accept NaN directly. Fit imputation inside a pipeline using training data only. Category codes such as 1, 2, 3 do not automatically represent meaningful distances."),
        ("Ties, weighting, and boundaries",
         "Odd k helps avoid binary vote ties with uniform weights, but does not prevent equal-distance ties at the neighbor cutoff. Distance-weighted voting is a different rule from the equal votes taught here."),
        ("Many features and large datasets",
         "Irrelevant features can distort closeness. High-dimensional distances can be less informative. Retaining and searching training examples requires memory and prediction time."),
        ("What the notebook's plots tell you",
         "Neighbor plots explain one vote; decision boundaries show predicted regions. Compare training and validation curves for overfitting, then confusion matrices for error types."),
    ]
    for top, (heading, content) in zip([0.79, 0.65, 0.51, 0.37, 0.23], items):
        card(ax, top, heading, content, height=0.12)
    paragraph(ax, 0.08, 0.09, "Sources: scikit-learn estimator and dataset documentation; links in the notebook.", size=7.5)
    finish(pdf, fig)


def build_guide(output, instructor):
    edition = "Instructor Copy" if instructor else "Student Copy"
    temporary = output.with_suffix(".tmp.pdf")
    try:
        with PdfPages(temporary) as pdf:
            pdf.infodict()["Title"] = f"KNN: A Beginner's Guide - {edition}"
            pdf.infodict()["Subject"] = edition
            for build_page in [labels_page, measurements_page, voting_page, distance_page, worked_vote_page,
                               distance_to_vote_page, pseudocode_page, training_and_selection_page,
                               practice_page, model_page, scaling_page,
                               case_objective_page, case_splits_page, case_fit_page, case_select_k_page,
                               case_overfitting_page, case_prediction_page, case_test_page,
                               case_metrics_page, case_scale_page, case_batches_page, code_page,
                               metrics_page, cautions_page]:
                build_page(pdf)
            if pdf.get_pagecount() != PAGE_COUNT:
                raise ValueError(f"The guide must contain {PAGE_COUNT} pages.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {output}")


def main():
    build_guide(OUTPUT, instructor=False)
    build_guide(INSTRUCTOR_OUTPUT, instructor=True)


if __name__ == "__main__":
    main()
