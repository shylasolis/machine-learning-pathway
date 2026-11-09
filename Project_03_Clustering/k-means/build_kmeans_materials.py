"""Build student/instructor PDFs and optionally regenerate the guided notebook."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from kmeans_lab import (
    INITIAL_CENTERS, ROOT, TOY, TOY_IDS, lloyd_trace, run_experiment,
)


INSTRUCTOR = ROOT.parent.parent / "Project_03_Clustering_Instructor"
BLACK = "#231f20"
ORANGE = "#e65300"
PALE = "#f4f4f4"
CLUSTER_COLORS = [ORANGE, "#666666"]


class Document:
    def __init__(self, path, title, instructor=False, flow=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix(".tmp.pdf")
        self.pdf = PdfPages(self.temporary)
        self.pdf.infodict().update(Title=title, Subject="Instructor Copy" if instructor else "Student Copy")
        self.instructor = instructor
        self.flow = flow
        self.number = 0
        self.fig = None
        self.y = 0
        self.width = 6.9

    def page(self, title, subtitle=""):
        if self.fig is not None:
            self.finish_page()
        self.number += 1
        self.fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 8.5), ylim=(0, 11))
        self.ax.axis("off")
        self.ax.add_patch(Rectangle((0, 10.42), 8.5, .58, facecolor=ORANGE, edgecolor="none"))
        self.ax.text(.8, 10.78, "K-MEANS / " +
                     ("INSTRUCTOR COPY" if self.instructor else "STUDENT COPY"),
                     color="white", fontsize=8.5, weight="bold", va="top")
        self.y = 10.12
        self.text(title, size=21, bold=True, gap=.13)
        if subtitle:
            self.text(subtitle, size=10, color=BLACK, gap=.20)
        else:
            self.y -= .10

    def room(self, height):
        if self.y - height < .85:
            if not self.flow:
                raise ValueError(f"Page {self.number} overflows: need {height:.2f} at {self.y:.2f}")
            self.page("K-means homework" if not self.instructor else "K-means: instructor notes",
                      "Continued")

    def wrap(self, text, width, size=11, bold=False, family="DejaVu Sans"):
        font = FontProperties(family=family, size=size, weight="bold" if bold else "normal")
        renderer = self.fig.canvas.get_renderer()
        lines = []
        for block in str(text).split("\n"):
            line = ""
            for word in block.split():
                candidate = f"{line} {word}".strip()
                length = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
                if length > width * self.fig.dpi:
                    if not line:
                        raise ValueError(f"Unbreakable text is too wide: {word}")
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        return lines

    def text(self, text, size=11, bold=False, color=BLACK, gap=.13, width=None):
        lines = self.wrap(text, width or self.width, size, bold)
        height = len(lines) * size * 1.38 / 72
        self.room(height + gap)
        self.ax.text(.8, self.y, "\n".join(lines), fontsize=size, color=color,
                     weight="bold" if bold else "normal", va="top", linespacing=1.38)
        self.y -= height + gap

    def heading(self, text):
        self.room(.75)
        self.text(text, size=12, bold=True, color=BLACK, gap=.09)

    def math(self, text, size=18, height=.65):
        expression = f"${text}$"
        font = FontProperties(family="DejaVu Sans", size=size)
        _, pixels, _ = self.fig.canvas.get_renderer().get_text_width_height_descent(
            expression, font, ismath=True,
        )
        height = max(height, pixels / self.fig.dpi + .24)
        self.room(height)
        self.ax.text(.95, self.y - .05, expression, fontsize=size, color=BLACK, va="top")
        self.y -= height

    def table(self, headers, rows, widths=None, size=9):
        if widths is None:
            widths = [self.width / len(headers)] * len(headers)
        for i, row in enumerate([headers] + list(rows)):
            wrapped = [self.wrap(str(cell), width - .16, size, bold=i == 0)
                       for cell, width in zip(row, widths)]
            height = max(map(len, wrapped)) * size * 1.35 / 72 + .16
            if self.y - height < .85 and self.flow:
                self.page("K-means: continued")
                if i > 0:
                    self.table(headers, [], widths, size)
            self.room(height)
            x = .8
            for lines, width in zip(wrapped, widths):
                self.ax.add_patch(Rectangle((x, self.y - height), width, height,
                                           facecolor=BLACK if i == 0 else PALE,
                                           edgecolor="white"))
                self.ax.text(x + .08, self.y - .07, "\n".join(lines), fontsize=size,
                             color="white" if i == 0 else BLACK, va="top",
                             weight="bold" if i == 0 else "normal", linespacing=1.35)
                x += width
            self.y -= height
        self.y -= .18

    def plot(self, draw, height=2.7):
        self.room(height + .28)
        axes = self.fig.add_axes([.14, (self.y - height + .45) / 11, .75, (height - .8) / 11])
        draw(axes)
        axes.tick_params(labelsize=8)
        self.y -= height + .28

    def finish_page(self):
        self.ax.plot([.8, 7.7], [.62, .62], color=ORANGE)
        self.ax.text(.8, .38, "K-means | From Customers to Centroids", fontsize=8, color=BLACK)
        self.ax.text(7.7, .38, str(self.number), fontsize=8, color=BLACK, ha="right")
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for axes in self.fig.axes:
            for artist in axes.texts:
                bbox = artist.get_window_extent(renderer)
                if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > self.fig.bbox.x1 or bbox.y1 > self.fig.bbox.y1:
                    raise ValueError(f"Text outside PDF page {self.number}: {artist.get_text()[:70]}")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def close(self):
        if self.fig is not None:
            self.finish_page()
        self.pdf.close()
        self.temporary.replace(self.path)
        print(f"Created {self.path} ({self.number} pages)")


def toy_plot(ax, centers, labels=None):
    if labels is None:
        labels = np.zeros(len(TOY), dtype=int)
    for j in np.unique(labels):
        points = TOY[labels == j]
        ax.scatter(*points.T, color=CLUSTER_COLORS[j], s=45, marker=["o", "s"][j])
    for name, point in zip(TOY_IDS, TOY):
        ax.annotate(name, point + .12, fontsize=9)
    ax.scatter(*centers.T, c="black", marker="X", s=150, label="Centers")
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="Monthly orders",
           ylabel="Average items/order")
    ax.legend(fontsize=8)


def build_guide(result, instructor=False):
    filename = "KMeans_Beginner_Guide_Instructor_Copy.pdf" if instructor else "KMeans_Beginner_Guide.pdf"
    p = Document(INSTRUCTOR / filename if instructor else ROOT / filename,
                 "K-means: From Customers to Centroids", instructor)
    trace = lloyd_trace(TOY, INITIAL_CENTERS)
    p.page("Learning without an answer column", "Our first unsupervised learning algorithm")
    p.text("A retailer has shopping measurements but no trusted customer-segment labels. Can we summarize patterns as groups of similar shopping behavior?")
    p.heading("Supervised versus unsupervised")
    p.table(["Supervised classification", "Unsupervised clustering"], [
        ["Features plus known targets; learn to predict a class.", "Features without training targets; search for structure."],
        ["KNN asks nearby labeled examples to vote.", "K-means assigns each point to a learned center."],
        ["A held-out label allows accuracy.", "No true segment labels means no ordinary accuracy."],
    ])
    p.text("K-means is a useful first clustering method, not a way to prove real customer types. It will partition data even when there are no meaningful natural groups.")
    p.heading("The whole learning journey")
    p.text("Read measurements -> prepare the geometry -> choose starting centers -> assign -> average -> repeat -> assess k and usefulness -> freeze -> assign future records.")
    p.text("All examples in this lesson are invented. No real customer records or external data are used.")

    p.page("What do these numbers mean?", "Six invented customers; no segment names are supplied")
    p.table(["Customer", "Monthly orders", "Average items/order"],
            [[name, int(row[0]), int(row[1])] for name, row in zip(TOY_IDS, TOY)])
    p.text("A = (1,1): one order per month, averaging one item per order. E = (9,7): nine orders per month, averaging seven items per order. These are counts, not ratings.")
    p.plot(lambda ax: toy_plot(ax, INITIAL_CENTERS), height=3.0)
    p.text("We use raw count coordinates only for this first arithmetic lesson. The larger coding case uses monthly orders and dollar baskets, with training-only scaling. IDs are never features.")

    p.page("Connect four algorithms you have learned", "What does each method learn, and what does a good result mean?")
    p.text("An OBJECTIVE or LOSS is a quantity used to guide fitting. An EVALUATION METRIC assesses a fitted model. Sometimes the same formula serves both purposes, but not every algorithm minimizes a training loss.")
    p.table(["Method", "What it uses and learns", "Fitting objective / quality check"], [
        ["Linear regression", "Known numerical targets; learns coefficients to predict values.",
         "Mean Squared Error (MSE): average squared difference between actual and predicted targets."],
        ["Binary logistic regression", "Known binary targets; learns coefficients to predict probabilities.",
         "Binary Cross-Entropy (BCE): penalizes probabilities inconsistent with observed binary labels."],
        ["KNN classification", "Known class labels; stores training examples and labels for neighbor voting.",
         "No iterative training-loss minimization in ordinary KNN. Choose settings using validation metrics."],
        ["K-means clustering", "Feature measurements, NO target labels; learns representative cluster centers.",
         "Within-Cluster Sum of Squares (WCSS): total squared distance to assigned centers."],
    ], widths=[1.35, 2.6, 2.95], size=9)
    p.heading("Three objectives, not four optimization loops")
    p.text("Regression, logistic regression, and K-means have fitting objectives here. Ordinary KNN does not keep adjusting coefficients or centers until a loss becomes small: fitting mainly retains examples (and may build a search structure).")
    p.text("For KNN, fit preprocessing on training rows, try neighbor counts, distance weights, or other declared settings, and compare validation accuracy, precision, recall, or F1 as appropriate. These evaluate labeled predictions; they do not become an iterative KNN fitting loop.", size=10)
    p.text("k means DIFFERENT things: KNN's k counts nearby training examples used to vote; K-means' k counts centers/groups. Neither k is automatically learned by ordinary fitting.", size=10)
    p.text("Supervised errors compare with known targets. K-means has no correct segment labels supplied: a compact group is not automatically a true category. Silhouette and stability assess geometry/agreement, not classification accuracy.", size=10)

    p.page("Name the K-means objective: WCSS", "Total squared distance to assigned centers; J is only its mathematical symbol")
    p.text("WITHIN-CLUSTER: compare each point with its OWN assigned center. SUM: add contributions from all points. SQUARES: square each coordinate difference before adding. WCSS is the name we will use for this total.")
    p.math(r"J=\mathrm{WCSS}=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2", size=18, height=.8)
    p.text("J is a variable name, NOT joules, energy, or a percentage. Other sources call this sum of squared errors (SSE). Scikit-learn calls the fitted total 'inertia' and stores it in inertia_. These are names for the same squared-distance total here.", size=10)
    p.heading("Compare what is being squared or penalized")
    p.math(r"\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2", size=17, height=.7)
    p.text("Regression: if the actual value is 10 and the prediction is 8, that record's squared error is (10-8)^2=4. Average all records' squared errors to obtain MSE.", size=10)
    p.math(r"\mathrm{BCE}=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\ln(p_i)+(1-y_i)\ln(1-p_i)\right]", size=15, height=.8)
    p.text("Binary logistic regression: y is 0 or 1; p is the predicted probability of class 1; ln is the natural logarithm. For y=1, p=0.9 contributes -ln(0.9)=0.1054, while p=0.1 contributes 2.3026. A confident wrong probability is penalized more.", size=10)
    p.text("K-means: for point (3,2) assigned to center (1,1), the contribution is (3-1)^2+(2-1)^2=5. Add ALL points' contributions to obtain WCSS; there is no true target y.", size=10)
    p.heading("Total versus mean: do not call WCSS 'MSE'")
    p.math(r"\mathrm{mean\ squared\ distance}=\frac{\mathrm{WCSS}}{n}"
           r"\qquad \frac{246}{6}=41", size=17, height=.6)
    p.text("Our first six-customer assignment has WCSS=246 and average squared distance=41 per customer. Each distance already includes both feature columns; we do NOT also divide by the number of columns.", size=10)
    p.text("For fixed data and k, dividing by n preserves the best centers. WCSS alone does not choose k. Do not compare numerical losses across different objectives, units, or scales.", size=10)

    p.page("What K-means tries to minimize", "k counts centers, not neighbors and not a discovered truth")
    p.text("A feature is an input column. Here the two features are monthly orders and average items per order. The customer ID is not a feature.")
    p.heading("Two centers, each with a pair of numbers")
    p.text("Choose k = 2: ask for TWO centers, not four. Each center needs one number for each input column. That pair locates one point on our two-axis graph.")
    p.table(["One center (one point)", "Column 1: orders/month", "Column 2: items/order"], [
        ["Center 0 at the start", "1", "1"],
        ["Center 1 at the start", "1", "3"],
    ], widths=[2.5, 2.2, 2.2])
    p.text("Center 0 is the point (1,1); center 1 is the point (1,3). Two pairs contain four numbers altogether. We do NOT have four centers. Training moves these two points by updating their numbers.")
    p.heading("The goal: make each group compact")
    p.math(r"J=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2")
    p.text("Read this as: add every customer's squared distance to its assigned center. J denotes Within-Cluster Sum of Squares (WCSS), called inertia in scikit-learn. x means a customer's measurements; mu means a center; c identifies its assigned center; n counts customers.")
    p.text("You choose the number of centers and the input columns. K-means learns where those centers end up. Assignments follow from the nearest center; meaningful group names are not supplied as training answers.")

    p.page("Step 1: assign every customer", "Start at center 0 = A (1,1), center 1 = B (1,3)")
    p.text("D has 7 monthly orders and 8 items per order: D = (7,8). Subtract matching columns, square each difference, then add. Here d squared means squared distance.")
    p.heading("Compare D with center 0 = (1,1)")
    p.math(r"d_0^2=(7-1)^2+(8-1)^2=36+49=85", size=17, height=.55)
    p.heading("Compare D with center 1 = (1,3)")
    p.math(r"d_1^2=(7-1)^2+(8-3)^2=36+25=61", size=17, height=.55)
    p.text("61 is smaller than 85, so D goes to center 1. Repeat that same comparison for every row.", gap=.10)
    p.table(["Customer", "d squared to 0", "d squared to 1", "Assignment"],
            [[name, f"{d[0]:g}", f"{d[1]:g}", int(label)]
             for name, d, label in zip(TOY_IDS, trace[0]["distances"], trace[0]["labels"])])
    p.text("C ties at 5 and 5. Our teaching implementation assigns distance ties to the smaller center index, so C goes to 0. Declare a rule rather than hiding the tie.")
    p.text("Square roots are unnecessary for ranking nonnegative distances: the smaller squared distance also has the smaller distance. K-means optimizes the squared values, not the roots.")
    p.text("Keep just the distance to each row's assigned center, not both distances. The next page shows exactly where all six contributions to J come from.")

    p.page("Where does J = 246 come from?", "Use the starting centers, BEFORE moving either center")
    p.text("J is one total: add the squared distance from EVERY customer to its assigned center. For two input columns, each customer's contribution is:")
    p.math(r"d_i^2=(x_{i,1}-\mu_{c_i,1})^2+(x_{i,2}-\mu_{c_i,2})^2", size=16, height=.55)
    p.text("In words: (orders minus center orders) squared, plus (items minus center items) squared. Use the center chosen on the previous page.")
    p.text("A and C use center 0 = (1,1). B, D, E, and F use center 1 = (1,3). Below, each line substitutes the customer's coordinates and its assigned center's coordinates, then evaluates both squares.")
    for name, row, label in zip(TOY_IDS, TOY, trace[0]["labels"]):
        center = INITIAL_CENTERS[label]
        differences = row - center
        squares = differences ** 2
        p.math(
            rf"d_{name}^2=({row[0]:g}-{center[0]:g})^2+({row[1]:g}-{center[1]:g})^2"
            rf"={differences[0]:g}^2+{differences[1]:g}^2"
            rf"={squares[0]:g}+{squares[1]:g}={squares.sum():g}",
            size=14, height=.43,
        )
    p.text("The first zero belongs to A: A is exactly at center 0. The second belongs to B: B is exactly at center 1. Zero means no distance, not a missing value.", size=10)
    p.math(r"J=d_A^2+d_B^2+d_C^2+d_D^2+d_E^2+d_F^2", size=15, height=.45)
    p.math(r"J=0+0+5+61+80+100=246", size=17, height=.45)
    p.text("Each customer contributes once. Do NOT add both candidate distances, square these results again, or divide by six. J is a total squared error, not an average or a percentage. Updating centers next changes the distances and therefore J.", size=10)

    p.page("Step 2: average assigned records", "Hold assignments fixed; calculate a mean for each coordinate")
    p.text("The STARTING centers (1,1) and (1,3) were chosen by using A and B's coordinates. They were not calculated as averages. Now we calculate NEW centers from the assigned customers.")
    p.math(r"\mathrm{new\ center\ value}=\frac{\mathrm{sum\ of\ values\ in\ one\ column}}{\mathrm{number\ of\ customers\ in\ this\ cluster}}", size=14, height=.65)
    p.text("Repeat this calculation separately for each input column. Do not mix orders with items, divide by k=2, or divide every cluster by all six customers.")
    p.heading("Center 0: A and C, so divide each column total by 2")
    p.text("A = (1,1), C = (3,2). Their orders are 1 and 3; their items/order values are 1 and 2.")
    p.math(r"\mathrm{orders}=\frac{1+3}{2}=\frac{4}{2}=2", size=17, height=.58)
    p.math(r"\mathrm{items/order}=\frac{1+2}{2}=\frac{3}{2}=1.5", size=17, height=.58)
    p.text("Put those two averages back in column order: new center 0 = (2,1.5). This means 2 orders/month and 1.5 items/order.")
    p.heading("Center 1: B, D, E, F, so divide each total by 4")
    p.text("B = (1,3), D = (7,8), E = (9,7), F = (9,9). The first numbers are orders; the second numbers are items/order.")
    p.math(r"\mathrm{orders}=\frac{1+7+9+9}{4}=\frac{26}{4}=6.5", size=17, height=.58)
    p.math(r"\mathrm{items/order}=\frac{3+8+7+9}{4}=\frac{27}{4}=6.75", size=17, height=.58)
    p.text("New center 1 = (6.5,6.75). A center is an average location: it need not match any actual customer, and fractional values are allowed.")
    p.text("This is why it is called k-MEANS. For fixed assignments, these column averages minimize total squared error.")
    p.text("Using these updated centers with the same assignments lowers J from 246 to 66.25. We are not done: moving centers can change which center is closest.")

    p.page("Step 3: reassign, then average again", "Iteration means one assignment followed by one center update")
    p.table(["ID", "d squared to (2,1.5)", "d squared to (6.5,6.75)", "Group"],
            [[name, f"{d[0]:.4f}", f"{d[1]:.4f}", int(label)]
             for name, d, label in zip(TOY_IDS, trace[1]["distances"], trace[1]["labels"])],
            widths=[.85, 2.15, 2.65, 1.25])
    p.text("B now belongs to cluster 0. The groups become A,B,C versus D,E,F. The reassignment reduces J to 25.1875 using the old centers.")
    p.text("Each cluster now contains THREE customers, so both column totals are divided by 3. B's move changes both the values being added and the cluster counts.", size=10)
    p.math(r"\mu_0=\left(\frac{1+1+3}{3},\frac{1+3+2}{3}\right)=\left(\frac{5}{3},\frac{6}{3}\right)=\left(\frac{5}{3},2\right)", size=16)
    p.math(r"\mu_1=\left(\frac{7+9+9}{3},\frac{8+7+9}{3}\right)=\left(\frac{25}{3},\frac{24}{3}\right)=\left(\frac{25}{3},8\right)", size=16)
    p.text("Example A's contribution uses its assigned center 0:")
    p.math(r"d_A^2=\left(1-\frac{5}{3}\right)^2+(1-2)^2=\frac{13}{9}", size=17)
    p.math(r"J=\frac{28}{3}\approx9.3333", size=18, height=.55)
    p.text("J adds all six contributions after the update. It is an error total, not a percentage.")

    p.page("Read the error totals, not just the table", "Same customers A-F: one iteration = assign, THEN update")
    p.text("WCSS after assignment uses the current centers after choosing each customer's nearest group, BEFORE moving centers. WCSS after update keeps those memberships fixed and measures the total again AFTER moving centers to the group means. J is shorthand for WCSS.", size=10)
    p.heading("Iteration 1: why does 246 become 66.25?")
    p.text("Starting centers: (1,1) and (1,3). Assign A,C to 0 and B,D,E,F to 1. Then update centers to (2,1.5) and (6.5,6.75). Each table entry is one customer's squared distance to its assigned center.", size=10)
    first = trace[0]
    first_before = first["distances"][np.arange(len(TOY)), first["labels"]]
    first_after = ((TOY - first["centers_after"][first["labels"]]) ** 2).sum(axis=1)
    p.table(["Customer", "Before centers move", "After centers move"], [
        [name, f"{before:g}", f"{after:.4f}"]
        for name, before, after in zip(TOY_IDS, first_before, first_after)
    ] + [["TOTAL WCSS", f"{first_before.sum():g}", f"{first_after.sum():.4f}"]],
        widths=[1.3, 2.8, 2.8])
    p.text("Example: B stays assigned to center 1 DURING this update. Its new contribution is:")
    p.math(r"d_B^2=(1-6.5)^2+(3-6.75)^2=30.25+14.0625=44.3125", size=14, height=.55)
    p.text("B's distance INCREASES from zero, but the TOTAL falls to 66.25. Averaging minimizes the sum for fixed memberships, not every individual's distance.")
    p.text("Iteration 1 ends at J=66.25. Next we reassign customers BEFORE averaging centers again. The next page calculates the new error total; it is not an assumed starting value.")

    p.page("Iteration 2: reassign, then calculate J", "Start from J=66.25; the reassignment produces a NEW total")
    p.text("Keep centers fixed at (2,1.5) and (6.5,6.75). Compare both centers for EVERY customer again. Only B changes its nearest center; the other five customers keep their groups.")
    p.heading("B = (1,3): compare both current centers")
    p.math(r"d_{B,1}^2=(1-6.5)^2+(3-6.75)^2=30.25+14.0625=44.3125", size=14, height=.55)
    p.math(r"d_{B,0}^2=(1-2)^2+(3-1.5)^2=1+2.25=3.25", size=15, height=.5)
    p.text("3.25 is smaller than 44.3125, so B switches from group 1 to group 0. The centers have NOT moved again yet. Only B's contribution to the total changes:")
    second_before = trace[1]["distances"][np.arange(len(TOY)), trace[1]["labels"]]
    p.table(["Customer", "Before reassignment", "After reassignment"], [
        [name, f"{before:.4f}", f"{after:.4f}"]
        for name, before, after in zip(TOY_IDS, first_after, second_before)
    ] + [["TOTAL WCSS", f"{first_after.sum():.4f}", f"{second_before.sum():.4f}"]],
        widths=[1.3, 2.8, 2.8])
    p.heading("Add the six selected squared distances")
    p.math(r"J=1.25+3.25+1.25+1.8125+6.3125+11.3125", size=15, height=.5)
    p.math(r"J=25.1875", size=18, height=.5)
    p.text("Equivalently, subtract B's old contribution from 66.25 and add its new one:")
    p.math(r"J=66.25-44.3125+3.25=25.1875", size=17, height=.5)
    p.text("Thus 25.1875 is the total AFTER iteration 2's assignment step, not a number supplied before it. Next, average the new groups A,B,C and D,E,F to move the centers.")

    p.page("Finish the totals; measure center movement", "Error and movement are DIFFERENT quantities")
    p.text("After B switches, the groups are A,B,C and D,E,F. Their means become (5/3,2) and (25/3,8). Using those NEW centers and those fixed groups gives these squared-distance contributions:")
    second_after = ((TOY - trace[1]["centers_after"][trace[1]["labels"]]) ** 2).sum(axis=1)
    p.table(list(TOY_IDS), [[f"{value:.4f}" for value in second_after]])
    p.math(r"J=\frac{13+13+16+16+13+13}{9}=\frac{84}{9}=\frac{28}{3}\approx9.3333", size=16, height=.6)
    p.text("This is why iteration 2 shows 25.1875 after assignment and 9.3333 after update. The table displays rounded decimals; the total is calculated with full precision.", size=10)
    p.heading("Where does movement = 6.75 come from?")
    p.text("Our teaching code combines ALL center-coordinate changes into one number: square each change, add them, then take the square root. It is not J, a percentage, a sum of the individual centers' travel distances, or the movement of just one center.", size=10)
    p.table(["Center", "Old position", "New position", "New minus old"], [
        [0, "(1,1)", "(2,1.5)", "(1,0.5)"],
        [1, "(1,3)", "(6.5,6.75)", "(5.5,3.75)"],
    ], widths=[.8, 1.7, 1.9, 2.5])
    p.math(r"\mathrm{movement}_1=\sqrt{1^2+0.5^2+5.5^2+3.75^2}=\sqrt{45.5625}=6.75", size=15, height=.6)
    p.text("For iteration 2, the coordinate changes are (-1/3,1/2) for center 0 and (11/6,5/4) for center 1. Apply the SAME calculation:", size=10)
    p.math(r"\mathrm{movement}_2=\sqrt{\left(-\frac{1}{3}\right)^2+\left(\frac{1}{2}\right)^2+\left(\frac{11}{6}\right)^2+\left(\frac{5}{4}\right)^2}\approx2.2989", size=16, height=.7)
    p.text("Error reductions in order:", size=10, bold=True, gap=.05)
    p.math(r"246\ \longrightarrow\ 66.25\ \longrightarrow\ 25.1875\ \longrightarrow\ 9.3333", size=17, height=.5)
    p.text("Iteration 3: same groups and centers, J=9.3333, movement=0. This run has settled; global optimality is not proven.", size=10, gap=.05)

    p.page("Step 4: check before stopping", "A third assignment/update confirms the centers do not change")
    p.table(["Iteration", "WCSS after assignment", "WCSS after update", "Center movement"], [
        [t["iteration"], f"{t['inertia_before']:.4f}", f"{t['inertia_after']:.4f}", f"{t['shift']:.4f}"]
        for t in trace
    ])
    p.plot(lambda ax: toy_plot(ax, trace[-1]["centers_after"], trace[-1]["labels"]), height=2.8)
    p.text("Iteration 3 returns the same assignments and means. Our small implementation stops when center movement is at most 1e-10, or stops at its iteration budget. Here movement is exactly zero.")
    p.text("A maximum-iteration cap is a safety limit, NOT evidence of convergence. Small center movement or unchanged assignments indicates a fixed point of this run, not a globally optimal k or business answer.")

    p.page("Predicting is not training", "Use the learned centers and the same feature transformation")
    p.text("New customer G = (2,2). Reuse the two final centers; do not calculate new ones.")
    p.math(r"d_0^2=\left(2-\frac{5}{3}\right)^2+(2-2)^2=\frac{1}{9}", size=17)
    p.math(r"d_1^2=\left(2-\frac{25}{3}\right)^2+(2-8)^2=\frac{685}{9}", size=17)
    p.text("Assign G to cluster 0. A call to predict does not move either center, refit the scaler, or ask neighbors to vote. The assignment is a nearest-center decision, not a 0-to-1 confidence estimate.")
    p.heading("Pseudocode: training K-means step by step")
    p.text("Pseudocode means plain-language algorithm instructions, not executable Python. Fitting means learning the cluster-center locations from training data.", size=10)
    p.text("This standard assign-and-average procedure is called Lloyd's algorithm, named after Stuart Lloyd. It is the SAME procedure we have been using, not a new model or an extra step.", size=10)
    p.text("Choose k starting centers.\nREPEAT:\n  Assign EVERY training row to its nearest current center.\n  AFTER assigning all rows, average each group's rows to update its center.\n  Check the stopping condition and iteration budget; stop if either is met.\nKeep final centers and preprocessing for later assignments.", size=10)
    p.text("'Full' means a complete training-data assignment pass BEFORE updating centers. Mini-batch K-means instead updates centers using smaller batches, without waiting for a full assignment pass. Full-data processing can still read rows in blocks while holding centers fixed until the pass ends.", size=10)
    p.text("If a cluster receives no rows, its mean is undefined. Production implementations handle this explicitly; our hand-code raises an error and asks for different seeds.", size=10)
    p.text("Cluster numbers are only IDs: swapping 0 and 1 changes no grouping. Suggest descriptive names only after reviewing profiles in original units.", size=10)

    p.page("Now you try: four new customers", "Work first; do not copy the six-customer answer")
    p.table(["Customer", "Monthly orders", "Average items/order"], [
        ["P", 2, 1], ["Q", 2, 3], ["R", 8, 7], ["S", 8, 9]
    ])
    p.text("Use k=2, starting centers (2,1) and (8,7). Calculate all squared distances, assign P-S, update both means, then reassign/update until unchanged. Show J before and after each update.")
    p.text("Then assign T=(3,2) without refitting. What was learned? What was chosen? Why is the final distance not accuracy?")
    if instructor:
        p.heading("Instructor solution")
        p.text("Initial squared-distance pairs: P (0,72), Q (4,52), R (72,0), S (100,4). Groups P,Q and R,S. Updated centers (2,2), (8,8). Initial J=8; updated J=4. Next assignment/update leaves centers unchanged, J=4.")
        p.text("T has squared distances 1 and 61: assign to center 0, without an update. Full hand work is required in homework; the solution is withheld in the student edition.")
    else:
        p.text("Distance table and assignments:\n\n________________________________________________________\n\nUpdated means and J:\n\n________________________________________________________\n\nStop check and new-customer explanation:\n\n________________________________________________________")

    p.page("Why scale before finding clusters?", "NEW four-customer example: not customers A-F or another iteration")
    p.text("We are preparing the measurements BEFORE running K-means. Scaling changes the ruler used for distances; it does not assign groups, move centers, or create buckets.")
    p.text("Each row below is one invented training customer. These FOUR rows are the whole scaling example: no hidden extra rows are needed. The later 600-customer lab is a separate dataset.")
    p.table(["Customer", "Orders/month", "Average spending/order ($)"], [
        [1, 2, 20], [2, 2, 40], [3, 6, 20], [4, 6, 40]
    ])
    p.text("'Average basket' means the average amount spent on one shopping order. It does NOT mean a bucket, bin, or cluster. For customer 4, (6,40) means six orders/month and average spending of 40 dollars/order.")
    p.heading("Why raw units can accidentally set the priorities")
    p.text("Compare customers 1 and 4. Their orders differ by 4, while spending differs by 20 dollars. With raw numbers:")
    p.math(r"d^2=(6-2)^2+(40-20)^2=16+400=416", size=17, height=.60)
    p.text("Spending contributes 400 of the 416 squared-distance units; orders contribute only 16. Spending dominates this comparison. That may be intended, but we should not give it extra influence accidentally because of its unit.")
    p.text("If spending were recorded in cents, the same customers would be (2,2000) and (6,4000): the spending contribution would be 2000 squared = 4,000,000. The purchasing behavior did not change; only the unit changed.")
    p.heading("Our choice: express both columns on a common scale")
    p.text("Standardization describes each value relative to its training column's average and spread. It removes the arbitrary dollar-versus-cent scale. It does not guarantee equally useful features or meaningful clusters.")
    p.text("Workflow: raw rows -> learn training means/SDs -> scale rows -> fit K-means.", size=10)

    p.page("SD means standard deviation", "A measure of spread, calculated separately for each input column")
    p.text("The MEAN is a column's average. STANDARD DEVIATION (SD) measures how spread out its values are around that average, in the same units as that column. It is not a center, a cluster count, or the average absolute distance.")
    p.text("Use the four training customers from the previous page. First calculate a mean; subtract it from each value; square each difference; average those squares; then take the square root.")
    p.heading("Orders/month: values 2, 2, 6, 6")
    p.math(r"\mathrm{Orders\ mean}=\frac{2+2+6+6}{4}=4", size=16, height=.55)
    p.math(r"\mathrm{Orders\ SD}=\sqrt{\frac{(2-4)^2+(2-4)^2+(6-4)^2+(6-4)^2}{4}}", size=16, height=.7)
    p.math(r"=\sqrt{\frac{4+4+4+4}{4}}=\sqrt{\frac{16}{4}}=\sqrt{4}=2", size=16, height=.65)
    p.text("The mean is 4 orders/month; the SD is 2 orders/month. Here every value happens to be 2 orders from the mean. In general, values need not be equally far from their mean.", size=10)
    p.heading("Average spending/order: values 20, 40, 20, 40 dollars")
    p.math(r"\mathrm{Spending\ mean}=\frac{20+40+20+40}{4}=30", size=16, height=.55)
    p.math(r"\mathrm{Spending\ SD}=\sqrt{\frac{(20-30)^2+(40-30)^2+(20-30)^2+(40-30)^2}{4}}", size=14, height=.7)
    p.math(r"=\sqrt{\frac{100+100+100+100}{4}}=\sqrt{100}=10", size=16, height=.65)
    p.text("The spending mean is 30 dollars/order; its SD is 10 dollars/order. The average squared difference INSIDE each root is called population variance. SD is its square root.", size=10)
    p.text("Why divide by 4, not 3? We follow StandardScaler's population-variance convention: divide by the number of training rows. The sample-SD formula taught in statistics often divides by n-1; it is not the convention used here.", size=10)

    p.page("What does a scaled value mean?", "Same four customers; change the ruler, not the original shopping behavior")
    p.text("For each column, subtract its TRAINING mean and divide by its TRAINING SD. The result z tells us how many standard deviations above or below the training average that measurement lies.")
    p.math(r"z=\frac{x-\mathrm{training\ mean}}{\mathrm{training\ standard\ deviation}}")
    p.heading("Customer 4: 6 orders/month and 40 dollars/order")
    p.math(r"z_{\mathrm{orders}}=\frac{6-4}{2}=\frac{2}{2}=1", size=17, height=.6)
    p.math(r"z_{\mathrm{spending}}=\frac{40-30}{10}=\frac{10}{10}=1", size=17, height=.6)
    p.text("Thus (6,40) becomes (1,1). This does NOT mean one order and one dollar! Both new values are unitless: each original measurement is one SD above its training average.")
    p.table(["Customer", "Raw values", "Scaled values", "Meaning"], [
        [1, "(2,20)", "(-1,-1)", "Both measurements one SD below average."],
        [2, "(2,40)", "(-1,1)", "Orders below; spending above average."],
        [3, "(6,20)", "(1,-1)", "Orders above; spending below average."],
        [4, "(6,40)", "(1,1)", "Both measurements one SD above average."],
    ], widths=[1.0, 1.15, 1.25, 3.5])
    p.text("Zero means at the training average, NOT no orders or no spending. Negative values mean below that average; positive values mean above it.", size=10)
    p.math(r"d_{\mathrm{scaled}}^2=(1-(-1))^2+(1-(-1))^2=4+4=8", size=16, height=.55)
    p.text("Customers 1 and 4 now differ by two SDs in each column. Each column contributes 4 to their squared distance. Do not compare 8 with raw J as an improvement: we changed the geometry, not the fit.", size=10)
    p.text("Save the training means (4,30) and SDs (2,10). Reuse them for validation, test, and future customers; do not fit a new ruler to each pool or each new row. The scaler learns these summaries; K-means then learns centers in the scaled space.", size=10)
    p.text("Scaling changes feature weighting. Outliers can distort means, SDs, and centers. A constant column has SD=0; do not divide by zero by hand (StandardScaler uses scale 1 for that column).", size=10)

    p.page("What is learned? What do we choose?", "Do not confuse fitting parameters with hyperparameters")
    p.table(["Item", "Role and interpretation"], [
        ["cluster_centers_", "One row per center, one number per input column. With 2 centers and 2 input columns: 2 rows of 2 numbers."],
        ["labels_", "Assignments for fitted rows, not observed ground-truth labels."],
        ["inertia_", "WCSS: total squared distance to assigned centers; lower is more compact."],
        ["n_iter_", "Iterations used by the retained full-KMeans run."],
        ["n_clusters (k)", "You choose how many groups/centers to ask for."],
        ["init", "How the initial center locations are chosen."],
        ["n_init", "How many complete starting attempts to try; retain the smallest training error."],
        ["max_iter", "Maximum assignment/update rounds for one attempt."],
        ["tol", "How small center movement must be for the tolerance stop check."],
        ["random_state", "Seed to reproduce the random starts; not a quality score."],
        ["algorithm", "Lloyd or Elkan implementation; same squared-Euclidean objective."],
    ], widths=[1.9, 5.0])
    p.text("k-means++ spreads initial seeds using distance-based sampling; it does not prove a global optimum. n_init=10 runs full KMeans from ten starts and retains the smallest training inertia.")
    p.text("We use explicit settings instead of version-dependent defaults. Do not select a seed because it gives the most flattering held-out result.")

    p.page("Choosing k without an answer column", "One fit can converge even when the chosen number of groups is poor")
    p.text("Training inertia typically drops as k grows; k equal to the number of distinct rows can memorize them. That does not reveal the correct customer segmentation.")
    p.heading("Combine evidence")
    p.text("Elbow: look for diminishing compactness gains, not a guaranteed sharp corner.\nSilhouette: compare within-group versus other-group separation.\nStability: compare repeat fits using the same assessment rows.\nUsefulness: review original-unit profiles, group sizes, costs, and actions.")
    p.text("For this lesson, predeclare candidate k=2,3,4,5,6 and choose the highest validation silhouette; exact score ties choose smaller k. This makes the exercise reproducible, not universally optimal.")
    p.text("Do not search k until a test score looks good. Do not force a desired story, equate a cluster with a person type, or claim the number of real biological species has been discovered.")
    p.text("Three different decisions: when ONE run stops, which initialization wins, and which k is selected. They happen at different levels.")

    p.page("Elbow: when do extra centers help less?", "Read a change in slope, not just the lowest point")
    ks = np.array([r["k"] for r in result["selection"]])
    errors = np.array([r["train_mean_squared_distance"] for r in result["selection"]])
    p.text("Executed lab: same training rows, training-fitted scaler, and fitting settings for every candidate. We plot J/n so the vertical axis is average squared distance. Lower means more compact, not more correct.")
    def elbow_plot(ax):
        ax.axis("off")
        for bounds, values, title, color in [
            ([0, 0, .41, 1], errors, "Executed lab", ORANGE),
            ([.59, 0, .41, 1], [1.0, .8, .64, .512, .4096],
             "Hypothetical: no clear elbow", "#666666"),
        ]:
            panel = ax.inset_axes(bounds)
            panel.plot(ks, values, "o-", color=color)
            panel.set(xlabel="Number of centers (k)", ylabel="Training J / n", xticks=ks,
                      ylim=(0, max(values) * 1.15))
            panel.set_title(title, fontsize=9)
            panel.tick_params(labelsize=8)
    p.plot(elbow_plot, height=3.0)
    p.table(["Change in k", "Reduction in training J/n", "Relative reduction"], [
        [f"{ks[i-1]} to {ks[i]}", f"{errors[i-1] - errors[i]:.4f}",
         f"{100 * (errors[i-1] - errors[i]) / errors[i-1]:.1f}%"]
        for i in range(1, len(ks))
    ], widths=[1.25, 3.2, 2.45])
    p.text("Read left to right: the large initial drop buys substantial compactness. Later smaller absolute drops buy less. This suggests examining k=3; the elbow alone does not certify it.")
    p.text("The gray hypothetical curve improves smoothly, by 20% each time. It does not show one obvious bend. Its scale is separate from the executed lab; compare the shape, not the heights between panels.")
    p.text("DO: report an ambiguous elbow and combine separation, stability, and usefulness. DO NOT invent a sharp corner, pick k=6 just because it has the lowest error, or compare curves made with different feature scaling.", size=10)

    p.page("What is a silhouette score?", "Return to our ORIGINAL six customers A-F, not a new dataset")
    p.text("Silhouette asks: is a customer closer to the OTHER customers in its own group than to customers in competing groups? It assesses an existing grouping. It does not move centers and is not the J that K-means minimizes.")
    final_labels = trace[-1]["labels"]
    p.text("Use our final groups: group 0 = A,B,C; group 1 = D,E,F. The coordinates and assignments below come from the earlier worked example, not an imagined new scenario.")
    p.plot(lambda ax: toy_plot(ax, trace[-1]["centers_after"], final_labels), height=2.4)
    p.text("Read the axes: x = monthly orders; y = average items per order. A dot's height is its items/order measurement, NOT its silhouette score. Black X marks show centers, but silhouette uses distances BETWEEN CUSTOMERS.", size=10)
    p.heading("Start with A = (1,1): measure to each OTHER customer")
    p.text("Use ordinary Euclidean distance: subtract matching columns, square, add, then take the square root. Unlike J, silhouette here uses distances, NOT squared distances.", size=10)
    p.table(["From A to", "Expanded ordinary distance", "Distance"], [
        ["B (1,3): own group", "sqrt((1-1)^2 + (1-3)^2) = sqrt(0+4)", "2.0000"],
        ["C (3,2): own group", "sqrt((1-3)^2 + (1-2)^2) = sqrt(4+1)", f"{np.sqrt(5):.4f}"],
        ["D (7,8): other group", "sqrt((1-7)^2 + (1-8)^2) = sqrt(36+49)", f"{np.sqrt(85):.4f}"],
        ["E (9,7): other group", "sqrt((1-9)^2 + (1-7)^2) = sqrt(64+36)", "10.0000"],
        ["F (9,9): other group", "sqrt((1-9)^2 + (1-9)^2) = sqrt(64+64)", f"{np.sqrt(128):.4f}"],
    ], widths=[1.8, 4.1, 1.0], size=8.5)
    p.text("sqrt means square root. Exclude A-to-A: a zero self-distance would make its own group look artificially close. Next we average the appropriate distances, not center coordinates.", size=10)

    p.page("Calculate A's silhouette, step by step", "Same six customers; all numbers come from the preceding distance table")
    own_mean = (2 + np.sqrt(5)) / 2
    other_mean = (np.sqrt(85) + 10 + np.sqrt(128)) / 3
    toy_scores = silhouette_samples(TOY, final_labels)
    p.heading("1. Calculate a: average distance to A's own group")
    p.text("A's other group members are B and C: TWO distances, so divide by 2. Lowercase a is a distance average; it is not the customer ID A.", size=10)
    p.math(r"a_A=\frac{d(A,B)+d(A,C)}{2}=\frac{2+\sqrt{5}}{2}"
           rf"\approx{own_mean:.4f}", size=16, height=.65)
    p.heading("2. Calculate b: closest competing group's average distance")
    p.text("There is only one other group: D,E,F. Average all THREE distances, not just the nearest record. With more groups, calculate a separate average for each competing group and choose the smallest average.", size=10)
    p.math(r"b_A=\frac{d(A,D)+d(A,E)+d(A,F)}{3}"
           r"=\frac{\sqrt{85}+10+\sqrt{128}}{3}"
           rf"\approx{other_mean:.4f}", size=15, height=.65)
    p.heading("3. Compare those averages")
    p.math(r"s_A=\frac{b_A-a_A}{\max(a_A,b_A)}"
           rf"\approx\frac{{{other_mean:.4f}-{own_mean:.4f}}}{{{other_mean:.4f}}}"
           rf"\approx{toy_scores[0]:.4f}", size=16, height=.65)
    p.text("max means use the larger average as the denominator; here b is larger. A is much closer to its own group, giving a positive score near 1. It is NOT a probability or a percentage correct. Calculations retain full precision; displayed distances are rounded.", size=10)
    p.table(["Score", "Meaning"], [
        ["Near +1", "Own group is much closer than competing groups."],
        ["Near 0", "Own and competing groups are similarly close; inspect overlap."],
        ["Negative", "A competing group is closer on average; inspect the assignment."],
    ], widths=[1.2, 5.7])
    p.text("Repeat for every customer, using that customer's own and competing groups:", size=10)
    p.table(list(TOY_IDS) + ["Mean of all 6"],
            [[f"{score:.4f}" for score in toy_scores] + [f"{toy_scores.mean():.4f}"]],
            widths=[.85] * 6 + [1.8])
    p.text("This mean summarizes the six-customer grouping, not just A. Next we switch explicitly to a NEW eight-point illustration to contrast separated and mixed assignments.", size=10)

    p.page("NEW illustration: eight invented points", "Not customers A-F; not the 600-customer lab")
    example_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                               [5, 0], [5, 1], [6, 0], [6, 1]], dtype=float)
    separated = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mixed = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    def silhouette_example(ax):
        ax.axis("off")
        for left, title, labels in [
            (0, "Separated assignments", separated), (.59, "Mixed assignments", mixed),
        ]:
            panel = ax.inset_axes([left, 0, .41, 1])
            score = silhouette_score(example_points, labels)
            for group in range(2):
                points = example_points[labels == group]
                panel.scatter(*points.T, color=CLUSTER_COLORS[group],
                           marker=["o", "s"][group], s=65, label=f"Group {group}")
            panel.set(xlim=(-1, 7), ylim=(-.6, 2), xlabel="Feature 1", ylabel="Feature 2")
            panel.set_title(f"{title}\nMean silhouette = {score:.3f}", fontsize=9)
            panel.tick_params(labelsize=8)
            panel.legend(fontsize=7, loc="upper center", ncol=2)
    p.plot(silhouette_example, height=3.0)
    p.text("The x-axis is an invented first measurement; the y-axis is an invented second measurement, NOT silhouette. Both panels use identical coordinates; only the assigned groups change. Orange circles and gray squares identify groups. We assign these groups by hand, not by running KMeans.")
    p.text("Left: each group occupies one nearby collection of points. Right: each group stretches across both distant locations. The title is the calculated MEAN silhouette over all eight points, not a value plotted on the y-axis.", size=10)
    p.table(["Example for ONE record", "Calculation", "Interpretation"], [
        ["a=0.4; b=1.0", "(1.0 - 0.4) / 1.0 = 0.6", "Own group is closer on average."],
        ["a=0.9; b=1.0", "(1.0 - 0.9) / 1.0 = 0.1", "Little separation; inspect boundaries."],
        ["a=1.0; b=0.4", "(0.4 - 1.0) / 1.0 = -0.6", "Another group is closer on average."],
    ], widths=[2.0, 2.3, 2.6], size=8.5)
    p.text("The table above is a SEPARATE hypothetical arithmetic illustration. Its a and b values are supplied, not calculated from either plot or customers A-F. It shows how the formula behaves as own-group and other-group distances change.", size=10)
    p.text("DO: inspect low/negative records and per-cluster scores. DO NOT call 0.6 '60% accurate', treat these examples as universal cutoffs, or assume a high average makes every group good.", size=10)

    p.page("Stability: would another start regroup people?", "Compare the SAME assessment customers in the SAME row order")
    reference = np.array([0, 0, 0, 1, 1, 1])
    renamed = np.array([1, 1, 1, 0, 0, 0])
    changed = np.array([0, 0, 1, 0, 1, 1])
    p.text("Hold training rows, preprocessing, k, and validation customers fixed. Change only the initialization seed, refit on training, then predict the same validation rows. Different customer samples would confound this initialization check.")
    def stability_plot(ax):
        for y, labels in enumerate([reference, renamed, changed]):
            for group in range(2):
                columns = np.flatnonzero(labels == group)
                ax.scatter(columns, np.full(len(columns), y), s=140,
                           color=CLUSTER_COLORS[group], marker=["o", "s"][group])
        ax.set(xticks=range(6), xticklabels=list("ABCDEF"), yticks=range(3),
               yticklabels=["Run 1", "Run 2", "Run 3"], ylim=(2.6, -.6),
               xlabel="Same validation customers (hypothetical assignments)")
    p.plot(stability_plot, height=2.4)
    p.table(["Comparison", "Adjusted Rand index (ARI)", "What changed?"], [
        ["Run 1 vs Run 2", f"{adjusted_rand_score(reference, renamed):.3f}",
         "Only cluster names switched. Group membership is identical."],
        ["Run 1 vs Run 3", f"{adjusted_rand_score(reference, changed):.3f}",
         "C and D switch membership; this is a genuinely different grouping."],
    ], widths=[1.55, 2.05, 3.3])
    p.text("ARI compares which pairs stay together. 1 means identical partitions even if IDs are renamed; around 0 means chance-level agreement under its adjustment. Negative values are possible. It is not accuracy against true customer types.")
    p.text(f"Executed lab: five single-start fits produce ten pairwise validation comparisons. Mean ARI = {result['stability'].mean():.4f}. This easy synthetic case is stable across these starts; that is not proof of natural customer categories.")
    p.text("DO: inspect all agreements, use more starts, and investigate competing solutions if agreement is weak. Check sensitivity to training samples separately, using resampled training sets and fixed assessment rows.", size=10)
    p.text("DO NOT compare raw cluster ID numbers, pick the seed with the prettiest story, or conclude stable groups must be useful. A stable but arbitrary split can still be a poor business model.", size=10)

    p.page("Usefulness: could we act on these groups?", "A good geometry score is a reason to investigate, not a business result")
    p.text("Hypothetical development profiles below: counts sum to 480. These are illustrative, NOT the executed lab's measured profiles. Values are in original units so an operations team can understand them.")
    def size_plot(ax):
        ax.bar(["0", "1", "2"], [230, 230, 20], color=[ORANGE, "#666666", "#aaaaaa"])
        ax.set(xlabel="Cluster ID (not a rank)", ylabel="Development customers",
               title="Hypothetical group sizes")
    p.plot(size_plot, height=2.1)
    p.table(["ID / count", "Orders/month; basket", "Possible next step, not a proven action"], [
        ["0 / 230", "2; $25", "Investigate occasional shopping and whether reminders would help."],
        ["1 / 230", "12; $35", "Investigate frequent small baskets and delivery costs."],
        ["2 / 20", "2; $250", "Inspect large baskets, outliers, and enough evidence for a distinct group."],
    ], widths=[1.05, 2.05, 3.8], size=8.5)
    p.text("DO: check within-group spread, missingness, outliers, coverage, and actionable differences. A 20-person group could be important or noise; size alone does not decide.", size=10)
    p.math(r"\mathrm{net\ incremental\ value}=230(0.40-1.00)=-138\ \mathrm{dollars}", size=14, height=.55)
    p.text("Hypothetical assumption: a proposed contact costs 1 dollar per customer and is expected to add only 0.40 dollars of margin per contacted customer. Contacting all 230 would lose 138 dollars. A high silhouette does not make that action worthwhile.", size=10)
    p.text("Treat expected benefit as a hypothesis. Use a controlled pilot to estimate incremental impact and uncertainty before scaling. Review privacy and potential harms.", size=10)
    p.text("DO NOT label a cluster 'wealthy' from basket size, infer a person's identity, or claim a campaign caused improvement just because its target cluster looks distinct.", size=10)

    p.page("Three decisions, three different checks", "Stopping a fit is NOT the same as selecting a model")
    p.table(["Level", "Keep fixed", "Decision and evidence"], [
        ["1. One starting attempt", "Training rows, scaler, k, initial centers",
         "Assign/update until convergence or budget exit. Track center movement, assignments, and J."],
        ["2. Several starting attempts", "Training rows, scaler, k",
         "For full KMeans, retain the attempt with the smallest training J. n_init sets the number of attempts."],
        ["3. Several candidate k values", "Training/validation split and scaler",
         "Compare validation silhouette and inspect elbow, stability, and usefulness. Apply a declared selection rule."],
    ], widths=[1.6, 2.0, 3.3], size=8.5)
    p.heading("When can ONE attempt stop?")
    p.table(["Rule", "What it means / what to do"], [
        ["Assignments unchanged", "Same customers stay together; averaging those same groups keeps the centers fixed."],
        ["Center movement is small", "Stop below a declared tolerance. Library-specific scaling matters."],
        ["Tiny improvement in J", "A custom monitor can stop when improvement is small under a declared rule. Standard KMeans does not expose a separate J-improvement threshold."],
        ["Iteration/time budget", "Stop because work is capped. Report the budget exit; do not call it convergence without evidence."],
    ], widths=[2.0, 4.9], size=8.5)
    p.text("Our hand-code uses center movement <= 1e-10, with max_iter=20. The full sklearn lab uses max_iter=300 and tol=1e-4; sklearn scales its shift tolerance with training-data variability, not a per-coordinate 0.0001 rule.", size=10)
    p.text("Example: a k=2 fit can stop perfectly while merging two useful patterns. A k=6 fit can converge with unnecessarily split groups. Convergence answers 'has this attempt settled?', not 'is this k useful?'.", size=10)
    p.text("DO: report convergence and model-selection evidence separately. DO NOT keep training merely because k is poor, or mistake n_init=10 for ten iterations.", size=10)

    p.page("Validation chooses; test checks once", "Ordinary KMeans does not use validation scores inside its assignment/update loop")
    p.table(["Stage", "Rows used", "Example in this lesson"], [
        ["Declare the search", "Before examining test results", "Candidates k=2,3,4,5,6; highest validation silhouette; smaller k on an exact tie."],
        ["Fit candidate models", "360 training customers", "Fit ONE training scaler; fit each k with 10 starts in that geometry."],
        ["Compare candidates", "120 validation customers", "Transform using that scaler; predict with each frozen candidate; calculate silhouette."],
        ["Select and freeze settings", "Training + validation evidence", f"The declared rule selects k={result['k']}. Elbow/stability/profiles support interpretation; they do not secretly override the rule."],
        ["Refit selected model", "480 development customers", "Fit a NEW scaler and selected-k model on training + validation."],
        ["Assess final model", "120 untouched test customers", "Transform, assign, and report distance/separation. Do not tune against these results."],
    ], widths=[1.6, 1.9, 3.4], size=8.5)
    p.text("Validation customers DO NOT move candidate centers. Validation helps you choose settings BETWEEN fits. Changing k means fitting another candidate; it is not another iteration of the same model.")
    p.text("Other settings and feature/scaling choices can also be compared on development evidence. Predeclare a bounded search; repeated informal tuning can overfit validation too.")
    p.heading("What if evidence disagrees?")
    p.text("If the elbow suggests 3 but silhouette favors 2, report the disagreement. For this reproducible exercise, follow the declared rule. For a real project, investigate geometry, profiles, and action costs; revise a justified development-only plan and seek fresh confirmation.")
    p.text("DO NOT try k values until the TEST score looks good, hide unsuccessful searches, or claim any selected k proves the number of biological species. Without known labels, these scores assess geometry, not species identification.", size=10)

    p.page("Our larger case: 600 synthetic records", "A real executed small-data lab, not a claim about actual customers")
    p.text("Features are orders per month and average basket in dollars. The generator creates 600 records with deliberately clear structure and returns no target. We do not fit or select using its hidden construction groups.")
    p.table(["Pool", "Rows", "Purpose"], [
        ["Training", 360, "Fit scaler and candidate centers"],
        ["Validation", 120, "Choose k and examine stability"],
        ["Test", 120, "Final frozen-model assessment"],
        ["Development", 480, "Train + validation refit after selection"],
    ], widths=[1.35, .85, 4.7])
    p.text("Split indices with seed 42: first 20% test, then 25% of development for validation. No target is available for stratification. Verify no overlaps and full coverage before scaling.")
    p.text("Fit the scaler only on training. Fit each candidate in that fixed scaled geometry. Then freeze choices, refit a NEW scaler and centers on development, and evaluate test once.")
    p.text("This split answers future-assignment questions. Descriptive clustering of a fixed dataset can use all rows, but then cannot claim independent generalization evidence. Repeated customers need group holdouts; future deployment may need time holdouts.")

    p.page("Read the k-selection results", "Numbers below are generated by the supplied lab")
    p.table(["k", "Train mean d squared", "Validation mean d squared", "Val silhouette"],
            [[r["k"], f"{r['train_mean_squared_distance']:.4f}",
              f"{r['validation_mean_squared_distance']:.4f}",
              f"{r['validation_silhouette']:.4f}"] for r in result["selection"]],
            widths=[.45, 2.15, 2.5, 1.8])
    def k_plot(ax):
        ax.plot([r["k"] for r in result["selection"]],
                [r["validation_silhouette"] for r in result["selection"]], "o-", color=ORANGE)
        ax.set(xlabel="Number of centers (k)", ylabel="Validation silhouette", ylim=(0, 1))
    p.plot(k_plot, height=2.7)
    p.text(f"The declared rule selects k={result['k']} in this run. Lowest training error is NOT the rule. A k with extra centers may split an already coherent pattern without improving separation.")
    p.text("Scores are in standardized geometry. Neither 0.8 silhouette nor 0.1 mean squared distance means 80% or 90% correct. There are no known segment labels here.")

    p.page("What do the metrics actually mean?", "Always identify the numerator, denominator, and coordinate system")
    p.math(r"s_i=\frac{b_i-a_i}{\max(a_i,b_i)}")
    p.text("a = mean distance from one record to other members of its own cluster. b = the smallest mean distance to the records of another cluster, not just the distance to its center.")
    p.math(r"a=0.4,\quad b=1.0:\qquad s=\frac{1.0-0.4}{1.0}=0.6", size=17)
    p.text("Average record silhouettes for the overall score. Near 1: separated; near 0: overlap; below 0: a different group may be closer on average.")
    p.text("Silhouette is defined when the assessed labels contain at least 2 groups and fewer groups than records. An isolated singleton has special handling in common implementations. Not every k/assignment yields a usable score.")
    p.table(["Metric", "Meaning and limitation"], [
        ["WCSS (J / inertia)", "Total squared distance to assigned centers. Depends on row count and scaling; smaller alone does not choose k."],
        ["Mean squared distance J/n", "Average squared distance to assigned center. Same geometry needed for comparisons."],
        ["Silhouette", "Within/between-group separation, not correctness or business value."],
        ["Stability ARI", "Agreement between partitions, invariant to swapped cluster IDs; not accuracy."],
    ], widths=[1.8, 5.1])
    p.text("sklearn KMeans.score is NEGATIVE total squared distance, not classification accuracy. This lab negates it and divides by the number of assessed rows.")

    p.page("Validate stability and interpret profiles", "Repeat fits can agree without proving the groups are real")
    p.text(f"Fit five single-initialization models using seeds 0-4 on training rows. Assign the SAME validation records with each model. Their ten pairwise ARI scores average {result['stability'].mean():.4f} in this run.")
    p.text("ARI=1 means identical partitions up to renaming. Around 0 means chance-level agreement under its adjustment; negative values are possible. High agreement on this easy simulation is not evidence that real customers have these exact types.")
    raw_centers = result["final_scaler"].inverse_transform(result["final"].cluster_centers_)
    counts = np.bincount(result["final"].labels_, minlength=result["k"])
    p.table(["Cluster ID", "Development count", "Orders/month", "Basket ($)"],
            [[j, int(counts[j]), f"{center[0]:.2f}", f"{center[1]:.2f}"]
             for j, center in enumerate(raw_centers)],
            widths=[1.1, 2.1, 1.9, 1.8])
    p.text("These are inverse-transformed final-development centers. Describe relative behavior such as 'frequent, smaller baskets', not moral or demographic identities. IDs may change after a refit.")
    p.text("A center is a representative average, not necessarily a customer. Inspect within-group spread and size, not only the average. Test a proposed action ethically rather than assuming clustering caused a business improvement.")

    p.page("When are we really done?", "Convergence, hyperparameter selection, and deployment are different")
    p.text("Within full KMeans: assignments may stop changing, or center shift falls below its tolerance threshold. max_iter limits work. scikit-learn scales its center-shift tolerance using data variability; it is not an accuracy target or a requested inertia.")
    p.text("n_iter_ reports the retained run. Hitting a cap merits checking movement/objective, more budget, and repeated starts on development data. Early stopping can leave centers not exactly equal to means of the final reassigned labels.")
    p.text("Across starts: n_init chooses the retained full-KMeans run by training inertia. Across k: our validation rule chooses k. No test score participates in either decision.")
    p.table(["Final test model", "Mean d squared", "Silhouette"],
            [[r["model"], f"{r['mean_squared_distance']:.6f}", f"{r['silhouette']:.6f}"]
             for r in result["test_results"]], widths=[3.3, 1.8, 1.8])
    p.text("The test scores assess geometry for similar new records, not true-segment correctness. The mini-batch result may equal the full result on this easy case; neither equality nor a speed advantage is guaranteed.")
    p.text("After evaluation, deployment may refit on more unlabeled feature rows. The old held-out score then describes the old model; obtain fresh evidence for the new model.")

    p.page("What changes at 85 million rows?", "Hypothetical arithmetic only; there is no huge dataset supplied")
    p.text("Suppose N=85 million customers, d=2 features, k=3. Full Lloyd compares each row with three centers per assignment pass. It does NOT compare every customer with all 85 million other customers.")
    p.table(["Quantity", "Calculation"], [
        ["Raw float64 features", "85,000,000 x 2 x 8 bytes = 1.36 GB decimal"],
        ["Point-center comparisons / pass", "85,000,000 x 3 = 255,000,000"],
        ["Across 20 passes", "255,000,000 x 20 = 5.1 billion comparisons"],
        ["Ten complete full-KMeans restarts", "5.1 billion x 10 = 51 billion comparisons"],
        ["Coordinate differences / pass", "255 million x 2 = 510 million; not the same unit"],
    ], widths=[2.6, 4.3])
    p.text("The 1.36 GB excludes IDs, copies, labels, buffers, scaler data, and initialization. Actual passes and convergence vary. More features, k candidates, restarts, and a poor storage layout add cost.")
    p.text("Standard full KMeans.fit is not an automatic out-of-core iterator. Elkan can allocate N-by-k bounds. Small center storage (k by d) does not mean the fitting process uses only that much memory.")
    p.text("No universal row ceiling: benchmark memory, IO, runtime, initialization, and assessment cost on representative data. Distributed or GPU systems can help, but do not change the objective or prove usefulness.")

    p.page("Blocks versus genuine mini-batch learning", "Unlike basic KNN, mini-batch K-means really updates learned parameters")
    p.heading("Full Lloyd in blocks")
    p.text("Keep centers fixed for the whole pass. Read blocks, assign rows, and accumulate a sum vector and count per center. After ALL blocks, divide global sums by counts and update centers. This reproduces the full assignment/update step, apart from rounding and explicit empty-cluster handling.")
    p.heading("MiniBatchKMeans")
    p.text("Updates centers after each mini-batch using running counts/averages, not a full-pass reassignment. It is an approximate optimization route with different convergence behavior. It is not running independent KMeans on each block.")
    p.text("At 100,000 rows per chunk, 85 million rows takes 850 chunks per pass. Mini-batch updates need not decrease full-data J after every step. Evaluate progress on a fixed development sample, not the test set.")
    p.text("MiniBatchKMeans has partial_fit; full KMeans does not. Repeated fit calls replace a model. batch_size in fit controls its batching; with partial_fit, YOU supply the batch and control passes/stopping. The first batch must support the requested k.")
    p.text("MiniBatchKMeans n_init selects initialization candidates, not ten full training restarts like KMeans. Its fit has max_iter and early-stop controls tol/max_no_improvement; caller-driven partial_fit does not enforce a global stopping loop.")
    p.text("Other mini-batch settings include batch_size (work per update) and reassignment_ratio (handling low-count centers). Their effects need development-only checks; defaults are not universal best settings. Standard KMeans uses squared Euclidean geometry, not an arbitrary distance-metric hyperparameter.")

    p.page("A realistic streaming plan", "Keep one feature geometry and bounded monitoring")
    p.text("1. Check units, missingness, duplicate customers, coverage, outliers, and privacy. Define a deployment-matching split or a clearly descriptive scope.")
    p.text("2. First training-only pass: use StandardScaler.partial_fit over chunks to accumulate GLOBAL means and variances. Do not give each block a different scaler.")
    p.text("3. Freeze that scaler. Second pass: transform chunks with the fixed statistics. Initialize centers from a representative development/training sample, not only the first site's rows.")
    p.text("4. Update MiniBatchKMeans with controlled passes and batches. Check distance trends and center movement on a fixed development-only monitor sample; predeclare patience and maximum work. A budget exit is not convergence.")
    p.text("5. Compute cluster counts and distance summaries by streaming predictions with fixed centers. For silhouette, use a representative bounded sample, for example 2,000 records, not a giant all-pairs matrix.")
    p.text("Silhouette involves pairwise distances within the assessed sample: roughly quadratic work. A full 85-million by 85-million matrix is infeasible. A 2,000-row sample contains about 4 million pairwise entries; report sampling uncertainty and per-cluster coverage.")
    p.text("6. Freeze choices, then evaluate held-out data once if future assignment is the goal. Monitor drift, empty/tiny groups, distance distributions, latency, subgroup consequences, and action usefulness. New data does not silently refit the model.")

    p.page("Connect the lesson to Python", "A fit learns centers; predict only assigns")
    p.text("The runnable script and notebook use these exact concepts. X_train contains two numeric features; there is no y argument with observed customer labels.")
    p.heading("Preprocessing and fitting")
    p.text("scaler = StandardScaler().fit(X_train)\nZ_train = scaler.transform(X_train)\nmodel = KMeans(n_clusters=k, init='k-means++',\n    n_init=10, max_iter=300, tol=1e-4,\n    random_state=42, algorithm='lloyd')\nmodel.fit(Z_train)", size=10)
    p.heading("Assignment and interpretation")
    p.text("Z_new = scaler.transform(X_new)\ngroups = model.predict(Z_new)\nraw_centers = scaler.inverse_transform(\n    model.cluster_centers_)", size=10)
    p.text("Save preprocessing and centers together. Never pass raw dollars to a model fitted on z-scores or report a z-score center as dollars. predict does not update either component.")
    p.text("Read kmeans_lab.py and run the notebook cells top to bottom. The script exports four CSVs and four figures. Make each prediction before the notebook shows its self-check.")
    p.text("Before final test evaluation, freeze the selected k and all geometry choices, then fit a NEW scaler and KMeans on the 480 development rows. The displayed code above illustrates one fitting stage, not permission to refit on test rows.")

    p.page("Limits, next steps, and your lab", "Useful groups require evidence beyond an attractive scatterplot")
    p.table(["Failure mode", "What to investigate"], [
        ["Curved or irregular groups", "Density-based methods such as DBSCAN/HDBSCAN; noise and parameter choices still matter."],
        ["Outliers and skew", "Audit records, justify transformations/robust preprocessing; do not delete inconvenient people."],
        ["Unequal densities or broad overlap", "Other objectives or mixture models; no universal winner."],
        ["Categorical codes or many irrelevant features", "Choose meaningful representations; numeric IDs are not distances. Consider validated feature reduction."],
        ["No meaningful separation", "Report uncertainty or decline to operationalize the segments."],
    ], widths=[2.5, 4.4])
    p.text("Open the guided notebook, make predictions before running cells, and complete the independent homework. You should be able to explain each update, stopping condition, setting, metric, and scale decision without reading an output number aloud.")
    p.text("Official references: scikit-learn KMeans, MiniBatchKMeans, silhouette_score, adjusted_rand_score, and clustering user guide. Exact URLs are in the folder README. The lab records measured small-data results; all large-scale costs are hypothetical.")
    if p.number != 38:
        raise ValueError(f"Expected 38 lesson pages, found {p.number}")
    p.close()


def clean_markdown(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text).replace("**", "").replace("`", "")
    return re.sub(r",(?=\S)", ", ", text)


def render_markdown(source, destination, instructor=False):
    """Printable companion; Markdown remains the clickable authoritative source."""
    doc = Document(destination, source.stem, instructor=instructor, flow=True)
    lines = source.read_text(encoding="utf-8").splitlines()
    i = 0
    started = False
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            doc.page(clean_markdown(line[2:]), "Printable companion; see the Markdown source for clickable links.")
            started = True
            i += 1
            continue
        if not started:
            doc.page("K-means")
            started = True
        if line.startswith("##"):
            doc.heading(clean_markdown(line.lstrip("# ")))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [clean_markdown(c.strip()) for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in row):
                    rows.append(row)
                i += 1
            count = len(rows[0])
            if count == 2:
                widths = [2.4, 4.5]
            elif count == 3 and rows[0][1] in ["Points", "Maximum"]:
                widths = [2., 1., 3.9]
            else:
                widths = None
            doc.table(rows[0], rows[1:], widths=widths, size=8.5)
            continue
        content = line.strip()
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|")):
            if re.match(r"^\d+\.|^- ", lines[i]):
                break
            content += " " + lines[i].strip()
            i += 1
        doc.text(clean_markdown(content), size=10)
    doc.close()


def build_notebook():
    cells = []
    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))
    def code(text):
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                          source=text.splitlines(keepends=True)))
    md("# K-means: From Customers to Centroids\n\n"
       "A guided, self-checking lab, not an answer-free assessment. All records are synthetic. "
       "Read the PDF first; answer each **Before running** prompt before opening the next output. "
       "No target labels are used to fit, choose k, or report 'accuracy'.")
    code("from pathlib import Path\nimport sys\nimport platform\n"
         "import numpy as np\nimport matplotlib\nimport matplotlib.pyplot as plt\nimport sklearn\n"
         "folder = Path.cwd()\n"
         "if not (folder / 'kmeans_lab.py').exists():\n"
         "    folder = folder / 'Project_03_Clustering' / 'k-means'\n"
         "if not (folder / 'kmeans_lab.py').exists():\n"
         "    raise FileNotFoundError('Start from the repository root or the k-means folder.')\n"
         "sys.path.insert(0, str(folder))\n"
         "from kmeans_lab import (TOY, TOY_IDS, INITIAL_CENTERS, lloyd_trace, squared_distances,\n"
         "                        run_experiment, write_outputs, make_customers, new_model)\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "print('Python', platform.python_version(), 'NumPy', np.__version__,\n"
         "      'sklearn', sklearn.__version__, 'Matplotlib', matplotlib.__version__)\n"
         "print('Features: monthly orders and average items/order. No target column.')\n"
         "for name, row in zip(TOY_IDS, TOY):\n    print(name, row)\n")
    md("## 1. Assign before updating\n\n"
       "A **feature** is an input column. We have two: monthly orders and average "
       "items per order. **k=2 means two centers**, each a point described by a pair "
       "of numbers. At the start, center 0 is (1,1) and center 1 is (1,3): two "
       "points, four stored numbers in total, NOT four centers.\n\n"
       "**Before running:** Initial centers are A=(1,1) and B=(1,3). "
       "Calculate D's squared distances. C ties: use the smaller center index. "
       "Which records belong together? k counts centers, not neighbors.")
    code("trace = lloyd_trace(TOY, INITIAL_CENTERS)\n"
         "first = trace[0]\n"
         "for name, distances, label in zip(TOY_IDS, first['distances'], first['labels']):\n"
         "    print(name, 'squared distances:', distances, 'cluster:', label)\n"
         "print('Means after update:', first['centers_after'])\n"
         "print('SSE before and after:', first['inertia_before'], first['inertia_after'])\n"
         "assert np.array_equal(first['labels'], [0,1,0,1,1,1])\n")
    md("## 2. Follow every iteration\n\n"
       "**Before running:** Is one update enough? Why might B change groups? "
       "Predict the final centers, not just their cluster IDs. Each mean minimizes "
       "squared error for its fixed assigned records.")
    code("for step in trace:\n"
         "    print('Iteration', step['iteration'], 'assignments', step['labels'])\n"
         "    print('New centers:', step['centers_after'])\n"
         "    print('J after assignment/update:', step['inertia_before'], step['inertia_after'],\n"
         "          'shift:', step['shift'], 'converged:', step['stopped'])\n"
         "assert trace[-1]['stopped']\n"
         "assert np.isclose(trace[-1]['inertia_after'], 28/3)\n"
         "assert all(s['inertia_after'] <= s['inertia_before'] + 1e-9 for s in trace)\n")
    md("## 3. Your turn before the check\n\n"
       "Use P=(2,1), Q=(2,3), R=(8,7), S=(8,9), initial P and R, k=2. "
       "Calculate both distance columns, update means, show J, and check a second iteration. "
       "Then assign T=(3,2) **without moving centers**. The next cell reveals the answer.")
    code("practice = np.array([[2.,1.],[2.,3.],[8.,7.],[8.,9.]])\n"
         "practice_trace = lloyd_trace(practice, practice[[0,2]])\n"
         "for step in practice_trace:\n"
         "    print(step['distances'], '\\nupdated centers:', step['centers_after'],\n"
         "          'J:', step['inertia_after'])\n"
         "centers = practice_trace[-1]['centers_after']\n"
         "distances = squared_distances(np.array([[3.,2.]]), centers)\n"
         "print('T distances / assigned group:', distances, distances.argmin(axis=1))\n"
         "assert np.allclose(centers, [[2,2],[8,8]])\n"
         "assert np.isclose(practice_trace[-1]['inertia_after'], 4)\n")
    md("## 4. Calculate scaling statistics, not just substitute them\n\n"
       "Separate invented dollar example: (2,20),(2,40),(6,20),(6,40). "
       "**Before running:** Compute mean, each squared deviation, population variance "
       "(divide by 4), and standard deviation. Scale query (6,40). "
       "Scaling learns a ruler, not the customer groups.")
    code("raw = np.array([[2.,20.],[2.,40.],[6.,20.],[6.,40.]])\n"
         "mean = raw.sum(axis=0) / len(raw)\n"
         "squared_deviations = (raw - mean)**2\n"
         "sd = np.sqrt(squared_deviations.sum(axis=0) / len(raw))\n"
         "print('Means:', mean, '\\nSquared deviations:', squared_deviations, '\\nSD:', sd)\n"
         "scaler = StandardScaler().fit(raw)\n"
         "assert np.allclose(mean, scaler.mean_) and np.allclose(sd, scaler.scale_)\n"
         "print('Scaled rows:', scaler.transform(raw))\n"
         "print('Query:', scaler.transform([[6.,40.]]))\n")
    md("## 5. Unlabeled experiment and selecting k\n\n"
       "600 synthetic retail records: order frequency and average basket dollars. "
       "The generator returns no target. Split 360/120/120 for training, validation, test. "
       "Training-only scaling; candidates 2-6; highest validation silhouette, smaller k "
       "on exact score ties. Freeze choices before a fresh 480-row development refit "
       "and one test evaluation. Inspect the script to trace these boundaries. "
       "This demonstration runs the predeclared whole workflow, not interactive test tuning.")
    code("result = run_experiment()\n"
         "print('Split counts:', *(len(result[key]) for key in ['train','validation','test']))\n"
         "for row in result['selection']:\n    print(row)\n"
         "print('Chosen k:', result['k'])\n"
         "print('Pairwise ARI across 5 independent single fits:', result['stability'])\n"
         "assert not set(result['development']) & set(result['test'])\n"
         "assert np.allclose(result['scaler'].mean_, result['x'][result['train']].mean(axis=0))\n"
         "write_outputs(result, folder / 'reports')\n")
    md("## 6. Read pictures and denominators\n\n"
       "**Before viewing:** Why does smaller training inertia not automatically choose k? "
       "Silhouette compares average within-group distance a with nearest other-group "
       "average b. It is not a percent correct. ARI here compares repeated partitions, "
       "not customer truth. Dollar-space scatterplots display assignments learned "
       "in scaled space.")
    code("from IPython.display import display, Image\n"
         "for filename in ['toy_iterations.png', 'scaling_comparison.png', 'k_selection.png']:\n"
         "    display(Image(filename=str(folder / 'reports' / filename)))\n")
    md("## 7. Refit, profile, and assign a new customer\n\n"
       "Read centers in original units using inverse_transform. Name patterns cautiously "
       "and inspect size/spread. Cluster IDs may swap between fits. "
       "**Before running:** Does predict change centers? Does its returned ID tell us "
       "a customer's real identity or a probability?")
    code("model = result['final']\n"
         "scaler = result['final_scaler']\n"
         "print('Original-unit centers:', scaler.inverse_transform(model.cluster_centers_))\n"
         "print('Development cluster sizes:', np.bincount(model.labels_))\n"
         "new_customer = np.array([[5.,50.]])\n"
         "saved = model.cluster_centers_.copy()\n"
         "group = model.predict(scaler.transform(new_customer))\n"
         "print('New customer (5 orders/month, $50 basket) assigned to:', group)\n"
         "assert np.array_equal(saved, model.cluster_centers_)\n"
         "print('Retained full-model iterations:', model.n_iter_)\n"
         "print('Test results, not accuracy:', result['test_results'])\n"
         "display(Image(filename=str(folder / 'reports' / 'test_clusters.png')))\n")
    md("## 8. Mini-batches update centers; KNN query batches do not\n\n"
       "Inspect stream_minibatches: caller supplies shuffled batches and five passes. "
       "Its StandardScaler is fixed on development before streaming. This is a work "
       "budget, not a convergence claim. partial_fit does not automatically implement "
       "a global max_iter or max_no_improvement loop. Full Lloyd instead holds centers "
       "fixed through a whole pass before averaging. Mini-batch full-data inertia need "
       "not fall after every update. Equal results on this easy case do not establish "
       "equal behavior everywhere.")
    code("for row in result['test_results']:\n"
         "    print(row)\n"
         "assert not hasattr(result['final'], 'partial_fit')\n"
         "assert hasattr(result['mini'], 'partial_fit')\n"
         "n, k, d = 85_000_000, 3, 2\n"
         "print('Hypothetical raw bytes:', n*d*8)\n"
         "print('Point-center comparisons/pass:', n*k)\n"
         "print('20 passes, 10 full KMeans starts:', n*k*20*10)\n"
         "print('100,000-row chunks/pass:', n//100_000)\n")
    md("## Reflect and extend without leaking test data\n\n"
       "1. Explain why convergence does not choose k or prove useful segments.\n"
       "2. Which feature might dominate without scaling, and what does scaling assume?\n"
       "3. Calculate silhouette for a=.4, b=1; explain its denominator.\n"
       "4. Why does sampled silhouette avoid an infeasible 85M-squared calculation?\n"
       "5. Explain n_init for full KMeans versus MiniBatchKMeans.\n"
       "6. Sketch a global-scaler, two-pass streaming workflow with a development monitor.\n"
       "7. Why are equal-density blobs an easy case? What if groups curve or overlap?\n"
       "8. Try a new seed, skewed/noisy generator, or k range only with a fresh "
       "development/evaluation plan; do not retune on this already-opened test set.\n\n"
       "Complete the separate homework and student rubric for independent evidence. "
       "Sources: scikit-learn KMeans, MiniBatchKMeans, silhouette_score and "
       "adjusted_rand_score documentation (linked in README).")
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.10"),
    ), nbformat=4, nbformat_minor=5)
    for i, cell in enumerate(cells):
        cell["id"] = f"kmeans-{i:02d}"
    path = ROOT / "KMeans_Customer_Segmentation.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Created {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true",
                        help="Regenerate notebook source, replacing saved execution outputs.")
    args = parser.parse_args()
    result = run_experiment()
    build_guide(result)
    build_guide(result, instructor=True)
    for name in ["homework_instructions", "homework_rubric"]:
        render_markdown(ROOT / f"{name}.md", ROOT / f"{name}.pdf")
    notes_source = INSTRUCTOR / "teaching_and_grading.md"
    if notes_source.exists():
        render_markdown(notes_source, INSTRUCTOR / "KMeans_Teaching_and_Grading_Instructor_Copy.pdf",
                        instructor=True)
    else:
        print("Private teaching/grading source is not included in the student repository; skipping that optional PDF.")
    if args.notebook:
        build_notebook()


if __name__ == "__main__":
    main()
