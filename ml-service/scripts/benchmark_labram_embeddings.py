"""Evaluate frozen LaBraM embeddings with a subject-independent linear probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{number:02d}" for number in range(1, 31)],
    )
    parser.add_argument("--configuration", default="O1")
    return parser.parse_args()


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, y_pred) if both_classes else float("nan")
        ),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if both_classes else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    for subject in args.subjects:
        path = (
            args.embeddings_root
            / args.configuration
            / f"labram_embeddings_subject_{subject}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Embeddings not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            features = data["X"].copy()
            labels = data["y"].copy()
            if str(data["subject_id"]) != subject:
                raise ValueError(f"Subject mismatch in {path}")
            if str(data["configuration"]) != args.configuration:
                raise ValueError(f"Configuration mismatch in {path}")
        feature_blocks.append(features)
        label_blocks.append(labels)
        subject_blocks.append(np.full(labels.size, subject, dtype="U2"))

    features = np.vstack(feature_blocks)
    labels = np.concatenate(label_blocks)
    subjects = np.concatenate(subject_blocks)
    rows: list[dict[str, str | float | int]] = []

    for position, test_subject in enumerate(args.subjects, start=1):
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        )
        model.fit(features[train_mask], labels[train_mask])
        prediction = model.predict(features[test_mask])
        probability = model.predict_proba(features[test_mask])[:, 1]
        fold_metrics = calculate_metrics(labels[test_mask], prediction, probability)
        row: dict[str, str | float | int] = {
            "model": "LaBraM frozen encoder + logistic regression",
            "configuration": args.configuration,
            "test_subject": test_subject,
            "test_windows": int(test_mask.sum()),
        }
        row.update(fold_metrics)
        rows.append(row)
        # Save immediately so fold results survive a disconnected Colab runtime.
        write_csv(args.output_dir / "loso_fold_metrics.csv", rows)
        print(
            f"[{position}/{len(args.subjects)}] Subject {test_subject}: "
            f"balanced accuracy={fold_metrics['balanced_accuracy']:.3f}, "
            f"F1={fold_metrics['f1']:.3f}",
            flush=True,
        )

    summary: dict[str, str | float | int] = {
        "model": "LaBraM frozen encoder + logistic regression",
        "configuration": args.configuration,
        "folds": len(rows),
        "windows": labels.size,
        "embedding_dim": features.shape[1],
    }
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows])
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    write_csv(args.output_dir / "loso_summary.csv", [summary])
    print("\nMean 30-subject LOSO result", flush=True)
    print(
        f"Balanced accuracy={summary['balanced_accuracy_mean']:.3f}, "
        f"sensitivity={summary['sensitivity_recall_mean']:.3f}, "
        f"specificity={summary['specificity_mean']:.3f}, "
        f"F1={summary['f1_mean']:.3f}, ROC AUC={summary['roc_auc_mean']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
