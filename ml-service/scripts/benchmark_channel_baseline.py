"""Compare frequency-feature classifiers across low-channel EEG layouts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark_baseline import build_models, metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHANNEL_CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "O1": ("O1",),
    "O2": ("O2",),
    "O1-O2": ("O1", "O2"),
    "C3-C4": ("C3", "C4"),
    "C3-C4-O1-O2": ("C3", "C4", "O1", "O2"),
}
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity_recall",
    "specificity",
    "f1",
    "roc_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "features" / "frequency_features_4ch_5s.npz",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CHANNEL_CONFIGURATIONS),
        default=list(CHANNEL_CONFIGURATIONS),
    )
    parser.add_argument(
        "--quality",
        choices=("all", "good", "both"),
        default="both",
        help="Evaluate all windows, good-only windows, or both variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "channel_benchmark" / "baseline",
    )
    return parser.parse_args()


def select_feature_columns(feature_names: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    names = [str(name) for name in feature_names]
    indices = np.asarray(
        [index for index, name in enumerate(names) if any(name.startswith(f"{channel}_") for channel in channels)],
        dtype=np.int64,
    )
    expected = 9 * len(channels)
    if indices.size != expected:
        raise ValueError(
            f"Expected {expected} features for channels {channels}, found {indices.size}"
        )
    return indices


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with np.load(args.features, allow_pickle=False) as data:
        features = data["X"]
        labels = data["y"]
        subjects = data["subjects"]
        quality_flags = data["quality_flags"].astype(bool)
        feature_names = data["feature_names"]

    quality_variants: dict[str, np.ndarray] = {
        "all_windows": np.ones(labels.size, dtype=bool),
        "good_only": ~quality_flags,
    }
    if args.quality != "both":
        selected_name = "all_windows" if args.quality == "all" else "good_only"
        quality_variants = {selected_name: quality_variants[selected_name]}

    rows: list[dict[str, str | float | int]] = []
    for configuration in args.configurations:
        channels = CHANNEL_CONFIGURATIONS[configuration]
        column_indices = select_feature_columns(feature_names, channels)
        selected_features = features[:, column_indices]
        print(
            f"\nConfiguration {configuration}: {len(channels)} channel(s), "
            f"{selected_features.shape[1]} features",
            flush=True,
        )
        for data_variant, keep in quality_variants.items():
            for model_name, model in build_models().items():
                for test_subject in sorted(np.unique(subjects)):
                    test_mask = (subjects == test_subject) & keep
                    train_mask = (subjects != test_subject) & keep
                    if np.unique(labels[train_mask]).size < 2:
                        raise ValueError(f"Training fold for {test_subject} has only one class")

                    model.fit(selected_features[train_mask], labels[train_mask])
                    predictions = model.predict(selected_features[test_mask])
                    probabilities = model.predict_proba(selected_features[test_mask])[:, 1]
                    fold_metrics = metrics(labels[test_mask], predictions, probabilities)
                    row: dict[str, str | float | int] = {
                        "configuration": configuration,
                        "channels": ",".join(channels),
                        "n_channels": len(channels),
                        "n_features": int(selected_features.shape[1]),
                        "data_variant": data_variant,
                        "model": model_name,
                        "test_subject": str(test_subject),
                        "test_windows": int(test_mask.sum()),
                    }
                    row.update(fold_metrics)
                    rows.append(row)
                    print(
                        f"  {data_variant}, {model_name}, subject {test_subject}: "
                        f"balanced accuracy={fold_metrics['balanced_accuracy']:.3f}, "
                        f"F1={fold_metrics['f1']:.3f}",
                        flush=True,
                    )

    summary_rows: list[dict[str, str | float | int]] = []
    for configuration in args.configurations:
        for data_variant in quality_variants:
            for model_name in build_models():
                selected = [
                    row
                    for row in rows
                    if row["configuration"] == configuration
                    and row["data_variant"] == data_variant
                    and row["model"] == model_name
                ]
                if not selected:
                    continue
                summary: dict[str, str | float | int] = {
                    "configuration": configuration,
                    "channels": str(selected[0]["channels"]),
                    "n_channels": int(selected[0]["n_channels"]),
                    "n_features": int(selected[0]["n_features"]),
                    "data_variant": data_variant,
                    "model": model_name,
                    "folds": len(selected),
                }
                for metric_name in METRIC_NAMES:
                    values = np.asarray([float(row[metric_name]) for row in selected])
                    summary[f"{metric_name}_mean"] = float(np.nanmean(values))
                    summary[f"{metric_name}_std"] = float(np.nanstd(values))
                summary_rows.append(summary)

    folds_path = args.output_dir / "loso_fold_metrics.csv"
    summary_path = args.output_dir / "loso_summary.csv"
    write_csv(folds_path, rows)
    write_csv(summary_path, summary_rows)

    print("\nMean LOSO channel results")
    for row in summary_rows:
        print(
            f"{row['configuration']}, {row['data_variant']}, {row['model']}: "
            f"balanced accuracy={row['balanced_accuracy_mean']:.3f}, "
            f"sensitivity={row['sensitivity_recall_mean']:.3f}, "
            f"specificity={row['specificity_mean']:.3f}, "
            f"F1={row['f1_mean']:.3f}, ROC AUC={row['roc_auc_mean']:.3f}"
        )
    print(f"Saved: {folds_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
