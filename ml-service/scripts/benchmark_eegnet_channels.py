"""Compare EEGNet across low-channel layouts with subject-independent LOSO."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from train_eegnet import (
    DEFAULT_SUBJECTS,
    EEGNet,
    load_windows,
    seed_everything,
    train_fold,
)


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
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--test-subjects",
        nargs="+",
        help="Optional subset of subjects to evaluate; defaults to all subjects.",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(CHANNEL_CONFIGURATIONS),
        default=list(CHANNEL_CONFIGURATIONS),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", choices=("all", "good"), default="all")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a configuration from its incrementally saved fold CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "channel_benchmark" / "eegnet",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs, patience, and batch size must be positive")

    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_windows, labels, subjects, quality_flags, source_channels, sampling_rate = load_windows(
        args.subjects
    )
    available_mask = np.ones(labels.size, dtype=bool) if args.quality == "all" else ~quality_flags
    test_subjects = args.test_subjects or args.subjects
    unknown = sorted(set(test_subjects) - set(args.subjects))
    if unknown:
        raise ValueError(f"Unknown test subjects: {unknown}")

    print(f"Device: {device}")
    print(f"Available windows: {int(available_mask.sum())}, sampling rate: {sampling_rate:g} Hz")
    combined_summaries: list[dict[str, str | float | int]] = []

    for configuration in args.configurations:
        selected_channels = list(CHANNEL_CONFIGURATIONS[configuration])
        missing = [channel for channel in selected_channels if channel not in source_channels]
        if missing:
            raise ValueError(f"Channels missing from processed data: {missing}")
        channel_indices = [source_channels.index(channel) for channel in selected_channels]
        windows = source_windows[:, channel_indices, :]

        configuration_dir = args.output_dir / configuration / args.quality
        models_dir = configuration_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        parameter_probe = EEGNet(len(selected_channels), windows.shape[-1], args.dropout)
        parameter_count = sum(parameter.numel() for parameter in parameter_probe.parameters())
        print(
            f"\nConfiguration {configuration}: {len(selected_channels)} channel(s), "
            f"{parameter_count} trainable parameters",
            flush=True,
        )

        folds_path = configuration_dir / "loso_fold_metrics.csv"
        if folds_path.exists() and not args.resume:
            raise FileExistsError(
                f"Fold results already exist: {folds_path}. "
                "Use --resume or select a new --output-dir."
            )

        rows: list[dict[str, str | float | int]] = (
            list(read_csv(folds_path)) if args.resume and folds_path.exists() else []
        )
        completed_subjects = [str(row["test_subject"]) for row in rows]
        if len(completed_subjects) != len(set(completed_subjects)):
            raise ValueError(f"Duplicate test subjects in existing fold CSV: {folds_path}")
        unknown_completed = sorted(set(completed_subjects) - set(test_subjects))
        if unknown_completed:
            raise ValueError(
                f"Existing fold CSV contains test subjects outside this run: {unknown_completed}"
            )
        if completed_subjects:
            print(
                f"Resuming {configuration}: {len(completed_subjects)}/{len(test_subjects)} "
                "fold(s) already complete",
                flush=True,
            )

        for fold_index, test_subject in enumerate(test_subjects, start=1):
            if test_subject in completed_subjects:
                print(
                    f"[{fold_index}/{len(test_subjects)}] Test subject {test_subject}: "
                    "already complete, skipping",
                    flush=True,
                )
                continue
            print(f"[{fold_index}/{len(test_subjects)}] Test subject {test_subject}", flush=True)
            row = train_fold(
                windows,
                labels,
                subjects,
                available_mask,
                selected_channels,
                test_subject,
                args,
                device,
                models_dir,
            )
            row = {
                "configuration": configuration,
                "channels": ",".join(selected_channels),
                "n_channels": len(selected_channels),
                "parameters": parameter_count,
                **row,
            }
            rows.append(row)
            # Save immediately after every complete fold. If training is
            # interrupted, --resume restarts only the unfinished fold.
            write_csv(folds_path, rows)
            print(
                f"  test balanced accuracy={row['balanced_accuracy']:.3f}, "
                f"sensitivity={row['sensitivity_recall']:.3f}, "
                f"specificity={row['specificity']:.3f}, "
                f"F1={row['f1']:.3f}, ROC AUC={row['roc_auc']:.3f}",
                flush=True,
            )

        summary: dict[str, str | float | int] = {
            "model": "EEGNet-8,2",
            "configuration": configuration,
            "channels": ",".join(selected_channels),
            "n_channels": len(selected_channels),
            "quality": args.quality,
            "folds": len(rows),
            "parameters": parameter_count,
        }
        for metric_name in METRIC_NAMES:
            values = np.asarray([float(row[metric_name]) for row in rows])
            summary[f"{metric_name}_mean"] = float(np.nanmean(values))
            summary[f"{metric_name}_std"] = float(np.nanstd(values))
        combined_summaries.append(summary)

        write_csv(configuration_dir / "loso_summary.csv", [summary])
        print(
            f"Mean {configuration}: balanced accuracy={summary['balanced_accuracy_mean']:.3f}, "
            f"sensitivity={summary['sensitivity_recall_mean']:.3f}, "
            f"specificity={summary['specificity_mean']:.3f}, "
            f"F1={summary['f1_mean']:.3f}, ROC AUC={summary['roc_auc_mean']:.3f}"
        )

    summary_path = args.output_dir / f"loso_summary_{args.quality}.csv"
    existing_summaries = read_csv(summary_path) if summary_path.exists() else []
    summaries_by_configuration: dict[str, dict[str, str | float | int]] = {
        str(row["configuration"]): row for row in existing_summaries
    }
    summaries_by_configuration.update(
        {str(row["configuration"]): row for row in combined_summaries}
    )
    ordered_summaries = [
        summaries_by_configuration[configuration]
        for configuration in CHANNEL_CONFIGURATIONS
        if configuration in summaries_by_configuration
    ]
    write_csv(summary_path, ordered_summaries)
    print(f"\nSaved combined summary: {summary_path}")


if __name__ == "__main__":
    main()
