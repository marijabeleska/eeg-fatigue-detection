"""Evaluate classical fatigue classifiers with leave-one-subject-out testing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "features" / "frequency_features_4ch_5s.npz",
    )
    return parser.parse_args()


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = roc_auc_score(y_true, probabilities) if np.unique(y_true).size == 2 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.features, allow_pickle=False) as data:
        features = data["X"]
        labels = data["y"]
        subjects = data["subjects"]
        quality_flags = data["quality_flags"].astype(bool)

    rows: list[dict[str, str | float | int]] = []
    for data_variant, keep in {
        "all_windows": np.ones(labels.size, dtype=bool),
        "good_only": ~quality_flags,
    }.items():
        for model_name, model in build_models().items():
            for test_subject in sorted(np.unique(subjects)):
                test_mask = (subjects == test_subject) & keep
                train_mask = (subjects != test_subject) & keep
                if np.unique(labels[train_mask]).size < 2:
                    raise ValueError(f"Training fold for {test_subject} has only one class")

                model.fit(features[train_mask], labels[train_mask])
                predictions = model.predict(features[test_mask])
                probabilities = model.predict_proba(features[test_mask])[:, 1]
                fold_metrics = metrics(labels[test_mask], predictions, probabilities)
                row: dict[str, str | float | int] = {
                    "data_variant": data_variant,
                    "model": model_name,
                    "test_subject": str(test_subject),
                    "test_windows": int(test_mask.sum()),
                }
                row.update(fold_metrics)
                rows.append(row)
                print(
                    f"{data_variant}, {model_name}, subject {test_subject}: "
                    f"balanced accuracy={fold_metrics['balanced_accuracy']:.3f}, "
                    f"F1={fold_metrics['f1']:.3f}",
                    flush=True,
                )

    output_dir = REPOSITORY_ROOT / "artifacts" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / "loso_fold_metrics.csv"
    with folds_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, str | float]] = []
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    for variant in sorted({str(row["data_variant"]) for row in rows}):
        for model_name in sorted({str(row["model"]) for row in rows}):
            selected = [
                row for row in rows
                if row["data_variant"] == variant and row["model"] == model_name
            ]
            summary: dict[str, str | float] = {
                "data_variant": variant,
                "model": model_name,
            }
            for metric_name in metric_names:
                values = np.asarray([float(row[metric_name]) for row in selected])
                summary[f"{metric_name}_mean"] = float(np.nanmean(values))
                summary[f"{metric_name}_std"] = float(np.nanstd(values))
            summary_rows.append(summary)

    summary_path = output_dir / "loso_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nMean LOSO results")
    for row in summary_rows:
        print(
            f"{row['data_variant']}, {row['model']}: "
            f"balanced accuracy={row['balanced_accuracy_mean']:.3f}, "
            f"F1={row['f1_mean']:.3f}, ROC AUC={row['roc_auc_mean']:.3f}"
        )
    print(f"Saved: {folds_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
