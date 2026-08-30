"""Train and evaluate a compact EEGNet with subject-independent LOSO folds."""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBJECTS = [f"{number:02d}" for number in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--test-subjects",
        nargs="+",
        help="Optional subset of subjects to evaluate; defaults to all subjects.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality",
        choices=("all", "good"),
        default="all",
        help="Use all windows or exclude windows marked as possible artifacts.",
    )
    return parser.parse_args()


class EEGWindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
    ) -> None:
        self.windows = windows
        self.labels = labels
        self.indices = indices
        self.channel_mean = channel_mean[:, None]
        self.channel_std = channel_std[:, None]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[position]
        normalized = (self.windows[index] - self.channel_mean) / self.channel_std
        signal = torch.from_numpy(normalized.astype(np.float32, copy=False)).unsqueeze(0)
        label = torch.tensor(float(self.labels[index]), dtype=torch.float32)
        return signal, label


class EEGNet(nn.Module):
    """EEGNet-8,2 style network for input shaped (batch, 1, channels, samples)."""

    def __init__(
        self,
        channels: int,
        samples: int,
        dropout: float,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
    ) -> None:
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier
        pointwise_filters = spatial_filters

        self.features = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, 64),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 16),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, pointwise_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(pointwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            feature_size = self.features(torch.zeros(1, 1, channels, samples)).numel()
        self.classifier = nn.Linear(feature_size, 1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        features = self.features(signals)
        return self.classifier(features.flatten(start_dim=1)).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_windows(subject_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    window_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    quality_blocks: list[np.ndarray] = []
    expected_channels: list[str] | None = None
    expected_sampling_rate: float | None = None

    for subject in subject_ids:
        path = (
            REPOSITORY_ROOT
            / "data"
            / "processed"
            / f"participant_{subject}"
            / "eeg_windows_4ch_5s.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            channels = [str(channel) for channel in data["channels"]]
            sampling_rate = float(data["sampling_rate"])
            if expected_channels is None:
                expected_channels = channels
                expected_sampling_rate = sampling_rate
            elif channels != expected_channels or sampling_rate != expected_sampling_rate:
                raise ValueError(f"Inconsistent EEG layout for participant {subject}")
            windows = data["X"].astype(np.float32, copy=False)
            window_blocks.append(windows)
            label_blocks.append(data["y"].astype(np.int64, copy=False))
            quality_blocks.append(data["quality_flags"].astype(bool, copy=False))
            subject_blocks.append(np.full(windows.shape[0], subject, dtype="U2"))

    return (
        np.concatenate(window_blocks),
        np.concatenate(label_blocks),
        np.concatenate(subject_blocks),
        np.concatenate(quality_blocks),
        expected_channels or [],
        float(expected_sampling_rate or 0.0),
    )


def choose_validation_subject(
    subjects: np.ndarray,
    labels: np.ndarray,
    available_mask: np.ndarray,
    test_subject: str,
) -> str:
    candidates: list[tuple[float, str]] = []
    remaining = available_mask & (subjects != test_subject)
    target_ratio = float(labels[remaining].mean())
    for subject in sorted(np.unique(subjects[remaining])):
        mask = remaining & (subjects == subject)
        positives = int(labels[mask].sum())
        negatives = int(mask.sum()) - positives
        if positives >= 20 and negatives >= 20:
            candidates.append((abs(float(labels[mask].mean()) - target_ratio), str(subject)))
    if not candidates:
        raise ValueError(f"No suitable validation subject for test subject {test_subject}")
    return min(candidates)[1]


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for signals, batch_labels in loader:
            logits = model(signals.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(probabilities)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "sensitivity_recall": recall_score(labels, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities) if np.unique(labels).size == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_fold(
    windows: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    available_mask: np.ndarray,
    channels: list[str],
    test_subject: str,
    args: argparse.Namespace,
    device: torch.device,
    models_dir: Path,
) -> dict[str, str | float | int]:
    validation_subject = choose_validation_subject(subjects, labels, available_mask, test_subject)
    print(f"  validation subject: {validation_subject}", flush=True)
    train_mask = available_mask & (subjects != test_subject) & (subjects != validation_subject)
    validation_mask = available_mask & (subjects == validation_subject)
    test_mask = available_mask & (subjects == test_subject)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)

    channel_mean = windows[train_indices].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    channel_std = windows[train_indices].std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    channel_std = np.maximum(channel_std, np.finfo(np.float32).eps)

    train_dataset = EEGWindowDataset(windows, labels, train_indices, channel_mean, channel_std)
    validation_dataset = EEGWindowDataset(windows, labels, validation_indices, channel_mean, channel_std)
    test_dataset = EEGWindowDataset(windows, labels, test_indices, channel_mean, channel_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    seed_everything(args.seed)
    model = EEGNet(
        channels=len(channels),
        samples=windows.shape[-1],
        dropout=args.dropout,
    ).to(device)
    positive_count = int(labels[train_indices].sum())
    negative_count = int(train_indices.size - positive_count)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for signals, batch_labels in train_loader:
            signals = signals.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)
            seen += len(batch_labels)

        validation_labels, validation_probabilities = predict(model, validation_loader, device)
        validation_metrics = calculate_metrics(validation_labels, validation_probabilities)
        validation_score = float(validation_metrics["balanced_accuracy"])
        print(
            f"  epoch {epoch:02d}: loss={running_loss / seen:.4f}, "
            f"val balanced accuracy={validation_score:.3f}",
            flush=True,
        )
        if validation_score > best_score + 1e-4:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    test_labels, test_probabilities = predict(model, test_loader, device)
    test_metrics = calculate_metrics(test_labels, test_probabilities)

    model_path = models_dir / f"eegnet_test_subject_{test_subject}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "samples": windows.shape[-1],
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "test_subject": test_subject,
            "validation_subject": validation_subject,
            "quality": args.quality,
        },
        model_path,
    )
    result: dict[str, str | float | int] = {
        "test_subject": test_subject,
        "validation_subject": validation_subject,
        "train_windows": int(train_indices.size),
        "validation_windows": int(validation_indices.size),
        "test_windows": int(test_indices.size),
        "best_epoch": best_epoch,
        "validation_balanced_accuracy": best_score,
    }
    result.update(test_metrics)
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs, patience, and batch size must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows, labels, subjects, quality_flags, channels, sampling_rate = load_windows(args.subjects)
    available_mask = np.ones(labels.size, dtype=bool) if args.quality == "all" else ~quality_flags
    test_subjects = args.test_subjects or args.subjects
    unknown = sorted(set(test_subjects) - set(args.subjects))
    if unknown:
        raise ValueError(f"Unknown test subjects: {unknown}")

    output_dir = REPOSITORY_ROOT / "artifacts" / "eegnet" / args.quality
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    parameter_probe = EEGNet(len(channels), windows.shape[-1], args.dropout)
    parameter_count = sum(parameter.numel() for parameter in parameter_probe.parameters())
    print(f"Device: {device}")
    print(f"Data: {int(available_mask.sum())} windows, {len(channels)} channels, {sampling_rate:g} Hz")
    print(f"EEGNet trainable parameters: {parameter_count}")

    rows: list[dict[str, str | float | int]] = []
    for fold_index, test_subject in enumerate(test_subjects, start=1):
        print(f"\n[{fold_index}/{len(test_subjects)}] Test subject {test_subject}", flush=True)
        row = train_fold(
            windows,
            labels,
            subjects,
            available_mask,
            channels,
            test_subject,
            args,
            device,
            models_dir,
        )
        rows.append(row)
        print(
            f"  test balanced accuracy={row['balanced_accuracy']:.3f}, "
            f"F1={row['f1']:.3f}, ROC AUC={row['roc_auc']:.3f}",
            flush=True,
        )

    folds_path = output_dir / "loso_fold_metrics.csv"
    with folds_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    summary: dict[str, str | float | int] = {
        "model": "EEGNet-8,2",
        "quality": args.quality,
        "folds": len(rows),
        "parameters": parameter_count,
    }
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in rows])
        summary[f"{metric_name}_mean"] = float(np.nanmean(values))
        summary[f"{metric_name}_std"] = float(np.nanstd(values))

    summary_path = output_dir / "loso_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print("\nMean LOSO results")
    print(
        f"Balanced accuracy={summary['balanced_accuracy_mean']:.3f}, "
        f"F1={summary['f1_mean']:.3f}, ROC AUC={summary['roc_auc_mean']:.3f}"
    )
    print(f"Saved: {folds_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
