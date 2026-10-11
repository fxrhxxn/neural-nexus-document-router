
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "baseline" / "routing_confusion_matrix.csv"
OUTPUT = ROOT / "outputs" / "baseline" / "routing_confusion_matrix.png"

# Read the confusion matrix without pandas.
with INPUT.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    header = next(reader)

    labels = header[1:]
    rows = list(reader)

true_labels = [row[0] for row in rows]
matrix = [[int(value) for value in row[1:]] for row in rows]

if true_labels != labels:
    raise ValueError(
        "The true-label order and predicted-label order do not match."
    )

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
fig.colorbar(image, ax=ax, label="Number of validation documents")

ax.set(
    title="Validation Routing Confusion Matrix",
    xlabel="Predicted routing outcome",
    ylabel="True routing outcome",
    xticks=range(len(labels)),
    yticks=range(len(true_labels)),
)

ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_yticklabels(true_labels)

# Annotate each cell with its count.
threshold = max(max(row) for row in matrix) / 2

for i, row in enumerate(matrix):
    for j, value in enumerate(row):
        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color="white" if value > threshold else "black",
        )

fig.tight_layout()
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Routing confusion matrix saved to: {OUTPUT}")
