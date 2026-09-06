"""Create the final 30-subject classical-vs-EEGNet comparison table and plot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "artifacts" / "channel_benchmark_30subjects"
METRICS = (
    ("balanced_accuracy", "Balanced accuracy"),
    ("sensitivity_recall", "Sensitivity"),
    ("specificity", "Specificity"),
    ("f1", "F1"),
    ("roc_auc", "ROC AUC"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    baseline_rows = read_csv(RESULTS_ROOT / "baseline" / "loso_summary.csv")
    eegnet_rows = read_csv(RESULTS_ROOT / "eegnet_incremental" / "loso_summary_all.csv")

    rows: list[dict[str, str | int | float]] = []
    for source_rows, family in ((baseline_rows, "Classical"), (eegnet_rows, "EEGNet")):
        for source in source_rows:
            if family == "Classical":
                model = (
                    "Logistic Regression"
                    if source["model"] == "logistic_regression"
                    else "Random Forest"
                )
                configuration = source["configuration"]
            else:
                model = "EEGNet"
                configuration = source["configuration"]

            row: dict[str, str | int | float] = {
                "family": family,
                "model": model,
                "configuration": configuration,
                "channels": source["channels"],
                "n_channels": int(source["n_channels"]),
                "folds": int(source["folds"]),
            }
            for metric, _ in METRICS:
                row[metric] = float(source[f"{metric}_mean"])
            rows.append(row)

    output_csv = RESULTS_ROOT / "model_comparison_30subjects.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [f"{row['configuration']}\n{row['model']}" for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, len(METRICS), figsize=(19, 5), sharey=True)
    colors = ["#5B9BD5" if row["family"] == "EEGNet" else "#A5A5A5" for row in rows]
    for axis, (metric, title) in zip(axes, METRICS):
        values = [float(row[metric]) for row in rows]
        bars = axis.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
        best = int(np.nanargmax(values))
        bars[best].set_edgecolor("#C00000")
        bars[best].set_linewidth(2)

    axes[0].set_ylabel("Mean LOSO score")
    figure.suptitle("MPD-DF: subject-independent comparison across 30 participants")
    figure.tight_layout()
    output_plot = RESULTS_ROOT / "model_comparison_30subjects.png"
    figure.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_plot}")


if __name__ == "__main__":
    main()
