"""Plot mean LOSO baseline metrics with between-subject variability."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    input_path = REPOSITORY_ROOT / "artifacts" / "baseline" / "loso_summary.csv"
    with input_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    labels = [
        f"{'All' if row['data_variant'] == 'all_windows' else 'Good'}\n"
        f"{'Logistic' if row['model'] == 'logistic_regression' else 'Forest'}"
        for row in rows
    ]
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1 score"),
        ("roc_auc", "ROC AUC"),
    ]
    x_positions = np.arange(len(rows))
    colors = ["#4472C4", "#70AD47", "#5B9BD5", "#A5A5A5"]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, (column, title) in zip(axes, metrics):
        means = [float(row[f"{column}_mean"]) for row in rows]
        standard_deviations = [float(row[f"{column}_std"]) for row in rows]
        axis.bar(
            x_positions,
            means,
            yerr=standard_deviations,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        if column in {"balanced_accuracy", "roc_auc"}:
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(x_positions, labels, fontsize=8)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Mean LOSO score ± SD")
    figure.suptitle("MPD-DF baseline results across 10 held-out subjects")
    figure.tight_layout()

    output_path = REPOSITORY_ROOT / "artifacts" / "baseline" / "loso_summary.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
