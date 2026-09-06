"""Extract resumable frozen-encoder LaBraM embeddings from MPD-DF windows."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


CHANNEL_CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "O1": ("O1",),
    "O1-O2": ("O1", "O2"),
    "C3-C4-O1-O2": ("C3", "C4", "O1", "O2"),
}

# Positions in the official LaBraM utils.standard_1020 list. Position zero is
# reserved for the class token, hence the +1 values used here.
LABRAM_POSITION_INDEX = {"C3": 40, "C4": 44, "O1": 81, "O2": 83}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labram-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{number:02d}" for number in range(1, 31)],
    )
    parser.add_argument(
        "--configuration",
        choices=tuple(CHANNEL_CONFIGURATIONS),
        default="O1",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_model(labram_repo: Path, checkpoint_path: Path, device: torch.device):
    if not (labram_repo / "modeling_finetune.py").is_file():
        raise FileNotFoundError(f"Official LaBraM source not found: {labram_repo}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(labram_repo.resolve()))
    import modeling_finetune  # type: ignore[import-not-found]

    model = modeling_finetune.labram_base_patch200_200(
        pretrained=False,
        EEG_size=1000,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )

    # This is the official, pinned LaBraM checkpoint. weights_only=False is
    # needed by some PyTorch versions because the checkpoint contains metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint
    if isinstance(checkpoint, dict):
        for model_key in ("model", "module"):
            if model_key in checkpoint and isinstance(checkpoint[model_key], dict):
                checkpoint_model = checkpoint[model_key]
                break
    if not isinstance(checkpoint_model, dict):
        raise ValueError("Unsupported LaBraM checkpoint structure")

    raw_state = OrderedDict()
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
        raw_state[key] = value

    model_state = model.state_dict()
    compatible_state = OrderedDict(
        (key, value)
        for key, value in raw_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    )
    model_keys = set(model_state)
    matched_keys = model_keys.intersection(compatible_state)
    if len(matched_keys) < int(0.9 * len(model_keys)):
        raise RuntimeError(
            f"Checkpoint compatibility check failed: matched {len(matched_keys)} "
            f"of {len(model_keys)} model tensors"
        )
    incompatible = model.load_state_dict(compatible_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected compatible checkpoint tensors: {unexpected[:10]}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return model, len(matched_keys), len(model_keys)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    os.replace(temporary, path)


def valid_embedding_output(path: Path, subject: str, configuration: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["X"]
            labels = data["y"]
            return bool(
                embeddings.ndim == 2
                and embeddings.shape == (labels.size, 200)
                and str(data["subject_id"]) == subject
                and str(data["configuration"]) == configuration
                and np.isfinite(embeddings).all()
            )
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    device = resolve_device(args.device)
    channels = CHANNEL_CONFIGURATIONS[args.configuration]
    output_dir = args.output_root / args.configuration
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Configuration: {args.configuration} ({', '.join(channels)})", flush=True)

    model, matched, total = load_model(args.labram_repo, args.checkpoint, device)
    print(f"Checkpoint tensors matched: {matched}/{total}", flush=True)
    input_chans = [0, *[LABRAM_POSITION_INDEX[channel] for channel in channels]]
    started = time.time()

    for position, subject in enumerate(args.subjects, start=1):
        output_path = output_dir / f"labram_embeddings_subject_{subject}.npz"
        if not args.overwrite and valid_embedding_output(
            output_path, subject, args.configuration
        ):
            print(f"[{position}/{len(args.subjects)}] Subject {subject}: valid output; skipping", flush=True)
            continue

        input_path = (
            args.data_root
            / f"participant_{subject}"
            / "eeg_windows_4ch_5s_200hz_uv.npz"
        )
        if not input_path.is_file():
            raise FileNotFoundError(f"Input not found: {input_path}")
        with np.load(input_path, allow_pickle=False) as data:
            all_channels = [str(value) for value in data["channels"]]
            indices = [all_channels.index(channel) for channel in channels]
            windows = data["X"][:, indices, :].astype(np.float32, copy=True)
            labels = data["y"].copy()
            quality_flags = data["quality_flags"].copy()
            starts = data["start_seconds"].copy()
            raw_labels = data["raw_labels"].copy()
            if float(data["sampling_rate"]) != 200.0 or str(data["unit"]) != "uV":
                raise ValueError(f"Unexpected sampling rate or unit in {input_path}")

        if windows.shape[-1] != 1000:
            raise ValueError(f"Expected 1000 samples per window, found {windows.shape[-1]}")
        # Official LaBraM fine-tuning code divides numeric microvolts by 100
        # before reshaping them into 200-sample channel patches.
        windows = windows.reshape(windows.shape[0], len(channels), 5, 200) / 100.0
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        blocks: list[np.ndarray] = []
        with torch.inference_mode():
            for (batch,) in loader:
                batch = batch.to(device, non_blocking=True)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = model.forward_features(batch, input_chans=input_chans)
                else:
                    features = model.forward_features(batch, input_chans=input_chans)
                blocks.append(features.float().cpu().numpy())
        embeddings = np.concatenate(blocks).astype(np.float32, copy=False)
        if embeddings.shape != (labels.size, 200) or not np.isfinite(embeddings).all():
            raise ValueError(f"Invalid embeddings generated for subject {subject}")

        atomic_save_npz(
            output_path,
            X=embeddings,
            y=labels,
            raw_labels=raw_labels,
            quality_flags=quality_flags,
            start_seconds=starts,
            subject_id=np.asarray(subject),
            configuration=np.asarray(args.configuration),
            channels=np.asarray(channels),
            embedding_dim=np.asarray(200, dtype=np.int16),
            encoder=np.asarray("labram-base frozen pretrained"),
        )
        print(
            f"[{position}/{len(args.subjects)}] Subject {subject}: "
            f"saved {embeddings.shape} -> {output_path}",
            flush=True,
        )

    elapsed = (time.time() - started) / 60
    print(f"Embedding extraction complete in {elapsed:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
