"""Compare classical baselines with the four-channel EEGNet LOSO result."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    baseline_rows = read_rows(
        REPOSITORY_ROOT / "artifacts" / "baseline" / "loso_summary.csv"
    )
    eegnet_row = read_rows(
        REPOSITORY_ROOT / "artifacts" / "eegnet" / "all" / "loso_summary.csv"
    )[0]

    entries: list[tuple[str, dict[str, str]]] = []
    for row in baseline_rows:
        variant = "All" if row["data_variant"] == "all_windows" else "Good"
        model = "Logistic" if row["model"] == "logistic_regression" else "Forest"
        entries.append((f"{variant}\n{model}", row))
    entries.append(("All\nEEGNet", eegnet_row))

    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1 score"),
        ("roc_auc", "ROC AUC"),
    ]
    positions = np.arange(len(entries))
    colors = ["#4472C4", "#70AD47", "#5B9BD5", "#A5A5A5", "#ED7D31"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for axis, (metric, title) in zip(axes, metrics):
        means = [float(row[f"{metric}_mean"]) for _, row in entries]
        deviations = [float(row[f"{metric}_std"]) for _, row in entries]
        axis.bar(
            positions,
            means,
            yerr=deviations,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        if metric in {"balanced_accuracy", "roc_auc"}:
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(positions, [label for label, _ in entries], fontsize=8)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Mean LOSO score ± SD")
    figure.suptitle("MPD-DF model comparison across 10 held-out subjects")
    figure.tight_layout()
    output_path = REPOSITORY_ROOT / "artifacts" / "eegnet" / "model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
