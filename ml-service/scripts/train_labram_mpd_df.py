"""Fine-tune pretrained LaBraM once for MPD-DF alert-vs-fatigue detection.

This is intentionally a simple subject-independent train/validation/test run,
not a multi-fold benchmark. The best validation checkpoint is evaluated once on
held-out test subjects and can serve as the LaBraM deployment candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import OrderedDict
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
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_TRAIN_SUBJECTS = [f"{number:02d}" for number in range(1, 25)]
DEFAULT_VAL_SUBJECTS = ["25", "26", "27"]
DEFAULT_TEST_SUBJECTS = ["28", "29", "30"]
O1_POSITION_INDEX = [0, 81]  # class token plus O1 in official utils.standard_1020


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labram-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-subjects", nargs="+", default=DEFAULT_TRAIN_SUBJECTS)
    parser.add_argument("--val-subjects", nargs="+", default=DEFAULT_VAL_SUBJECTS)
    parser.add_argument("--test-subjects", nargs="+", default=DEFAULT_TEST_SUBJECTS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(labram_repo: Path, checkpoint_path: Path, device: torch.device):
    if not (labram_repo / "modeling_finetune.py").is_file():
        raise FileNotFoundError(f"Official LaBraM source not found: {labram_repo}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    sys.path.insert(0, str(labram_repo.resolve()))
    import modeling_finetune  # type: ignore[import-not-found]

    model = modeling_finetune.labram_base_patch200_200(
        pretrained=False,
        EEG_size=1000,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint
    if isinstance(checkpoint, dict):
        for model_key in ("model", "module"):
            if model_key in checkpoint and isinstance(checkpoint[model_key], dict):
                checkpoint_model = checkpoint[model_key]
                break
    if not isinstance(checkpoint_model, dict):
        raise ValueError("Unsupported LaBraM checkpoint structure")

    state = OrderedDict()
    has_student_prefix = any(str(key).startswith("student.") for key in checkpoint_model)
    for original_key, value in checkpoint_model.items():
        key = str(original_key)
        if has_student_prefix:
            if not key.startswith("student."):
                continue
            key = key[len("student.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("head.") or "relative_position_index" in key:
            continue
        state[key] = value

    model_state = model.state_dict()
    compatible_state = OrderedDict(
        (key, value)
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    )
    model_keys_without_head = {
        key for key in model_state if not key.startswith("head.")
    }
    matched = model_keys_without_head.intersection(compatible_state)
    if len(matched) < int(0.9 * len(model_keys_without_head)):
        raise RuntimeError(
            f"Checkpoint compatibility failed: matched {len(matched)} of "
            f"{len(model_keys_without_head)} encoder tensors"
        )
    incompatible = model.load_state_dict(compatible_state, strict=False)
    # The official pretraining checkpoint has a final ``norm`` layer, while
    # its own fine-tuning configuration uses a newly initialized ``fc_norm``
    # for mean pooling. Both fc_norm and the new binary head are therefore
    # expected to start from the downstream task initialization.
    expected_new_prefixes = ("head.", "fc_norm.")
    bad_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(expected_new_prefixes)
    ]
    bad_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("head.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={bad_missing[:10]}, "
            f"unexpected={bad_unexpected[:10]}"
        )
    model.to(device)
    return model, len(matched), len(model_keys_without_head)


def load_subjects(data_root: Path, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for subject in subjects:
        path = (
            data_root
            / f"participant_{subject}"
            / "eeg_windows_4ch_5s_200hz_uv.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            channels = [str(value) for value in data["channels"]]
            o1_index = channels.index("O1")
            x = data["X"][:, o1_index : o1_index + 1, :].astype(np.float32, copy=True)
            y = data["y"].astype(np.float32, copy=True)
            if float(data["sampling_rate"]) != 200.0 or str(data["unit"]) != "uV":
                raise ValueError(f"Unexpected sampling rate or unit: {path}")
            if x.shape[-1] != 1000:
                raise ValueError(f"Expected 1000 samples per window: {path}")
        windows.append(x)
        labels.append(y)

    all_windows = np.concatenate(windows)
    # Official engine_for_finetuning.py divides microvolt values by 100 and
    # reshapes each channel into 200-sample patches.
    all_windows /= 100.0
    all_windows = all_windows.reshape(all_windows.shape[0], 1, 5, 200)
    return all_windows, np.concatenate(labels)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
    cuda: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=False,
    )


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = (probabilities >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, predictions) if both_classes else float("nan")
        ),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "sensitivity_recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if both_classes else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, criterion):
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(x, input_chans=O1_POSITION_INDEX).squeeze(-1)
            loss = criterion(logits, y)
        losses.append(float(loss.item()) * y.size(0))
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(y.cpu().numpy())
    y_true = np.concatenate(targets)
    probability = np.concatenate(probabilities)
    result = metrics(y_true, probability)
    result["loss"] = sum(losses) / y_true.size
    return result


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    split_sets = [set(args.train_subjects), set(args.val_subjects), set(args.test_subjects)]
    if any(left.intersection(right) for i, left in enumerate(split_sets) for right in split_sets[i + 1 :]):
        raise ValueError("Train, validation and test subjects must not overlap")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Train subjects ({len(args.train_subjects)}): {args.train_subjects}", flush=True)
    print(f"Validation subjects ({len(args.val_subjects)}): {args.val_subjects}", flush=True)
    print(f"Test subjects ({len(args.test_subjects)}): {args.test_subjects}", flush=True)

    model, matched, total = make_model(args.labram_repo, args.checkpoint, device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Loaded pretrained encoder tensors: {matched}/{total}", flush=True)
    print(f"Trainable parameters: {trainable:,}", flush=True)

    train_x, train_y = load_subjects(args.data_root, args.train_subjects)
    val_x, val_y = load_subjects(args.data_root, args.val_subjects)
    test_x, test_y = load_subjects(args.data_root, args.test_subjects)
    print(
        f"Windows: train={train_y.size}, validation={val_y.size}, test={test_y.size}",
        flush=True,
    )
    print(
        f"Fatigue: train={int(train_y.sum())}, validation={int(val_y.sum())}, "
        f"test={int(test_y.sum())}",
        flush=True,
    )

    train_loader = make_loader(
        train_x, train_y, args.batch_size, True, args.num_workers, device.type == "cuda"
    )
    val_loader = make_loader(
        val_x, val_y, args.batch_size * 2, False, args.num_workers, device.type == "cuda"
    )
    test_loader = make_loader(
        test_x, test_y, args.batch_size * 2, False, args.num_workers, device.type == "cuda"
    )
    positive = float(train_y.sum())
    negative = float(train_y.size - positive)
    pos_weight = torch.tensor(negative / positive, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_path = args.output_dir / "labram_o1_best.pt"
    last_path = args.output_dir / "labram_o1_last.pt"
    history_path = args.output_dir / "training_history.csv"
    start_epoch = 1
    best_balanced_accuracy = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    if args.resume and last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best_balanced_accuracy = float(saved["best_balanced_accuracy"])
        epochs_without_improvement = int(saved["epochs_without_improvement"])
        history = list(saved["history"])
        print(f"Resuming at epoch {start_epoch}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(x, input_chans=O1_POSITION_INDEX).squeeze(-1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * y.size(0)
            seen += y.size(0)

        validation = evaluate(model, val_loader, device, criterion)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        write_history(history_path, history)
        score = float(validation["balanced_accuracy"])
        improved = score > best_balanced_accuracy + 1e-6
        if improved:
            best_balanced_accuracy = score
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "channels": ["O1"],
                    "input_chans": O1_POSITION_INDEX,
                    "sampling_rate": 200,
                    "window_seconds": 5,
                    "unit": "uV divided by 100",
                    "train_subjects": args.train_subjects,
                    "val_subjects": args.val_subjects,
                    "test_subjects": args.test_subjects,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_balanced_accuracy": best_balanced_accuracy,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_path,
        )
        print(
            f"Epoch {epoch:02d}: train loss={loss_sum / seen:.4f}, "
            f"val balanced accuracy={score:.3f}, val F1={validation['f1']:.3f}, "
            f"best={best_balanced_accuracy:.3f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_result = evaluate(model, test_loader, device, criterion)
    result = {
        "model": "pretrained LaBraM base fine-tuned",
        "configuration": "O1",
        "best_epoch": int(best["epoch"]),
        "train_subjects": args.train_subjects,
        "validation_subjects": args.val_subjects,
        "test_subjects": args.test_subjects,
        "test_windows": int(test_y.size),
        **{f"test_{key}": value for key, value in test_result.items()},
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    elapsed = (time.time() - started) / 60
    print("\nHeld-out test result", flush=True)
    print(
        f"Balanced accuracy={test_result['balanced_accuracy']:.3f}, "
        f"sensitivity={test_result['sensitivity_recall']:.3f}, "
        f"specificity={test_result['specificity']:.3f}, "
        f"F1={test_result['f1']:.3f}, ROC AUC={test_result['roc_auc']:.3f}",
        flush=True,
    )
    print(f"Completed in {elapsed:.1f} minutes; best model: {best_path}", flush=True)


if __name__ == "__main__":
    main()
