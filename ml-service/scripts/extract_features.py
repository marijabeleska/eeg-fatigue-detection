"""Extract interpretable frequency-domain features from processed EEG windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import welch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{number:02d}" for number in range(1, 11)],
    )
    return parser.parse_args()


def integrate_band(
    power_spectral_density: np.ndarray,
    frequencies: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    return np.trapezoid(power_spectral_density[..., mask], frequencies[mask], axis=-1)


def features_for_subject(path: Path) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        windows = data["X"]
        labels = data["y"].copy()
        quality_flags = data["quality_flags"].copy()
        starts = data["start_seconds"].copy()
        channels = [str(channel) for channel in data["channels"]]
        sampling_rate = float(data["sampling_rate"])

    frequencies, psd = welch(
        windows,
        fs=sampling_rate,
        nperseg=min(256, windows.shape[-1]),
        noverlap=128,
        axis=-1,
    )
    total_mask = (frequencies >= 1.0) & (frequencies < 40.0)
    total_power = np.trapezoid(psd[..., total_mask], frequencies[total_mask], axis=-1)
    epsilon = np.finfo(np.float32).eps

    band_power = {
        name: integrate_band(psd, frequencies, low, high)
        for name, (low, high) in BANDS.items()
    }

    columns: list[np.ndarray] = []
    names: list[str] = []
    for channel_index, channel in enumerate(channels):
        channel_total = total_power[:, channel_index]
        relative = {
            name: values[:, channel_index] / (channel_total + epsilon)
            for name, values in band_power.items()
        }
        for name in BANDS:
            columns.append(relative[name])
            names.append(f"{channel}_{name}_relative_power")

        columns.extend(
            [
                np.log10(channel_total + epsilon),
                relative["theta"] / (relative["alpha"] + epsilon),
                relative["theta"] / (relative["beta"] + epsilon),
                relative["alpha"] / (relative["beta"] + epsilon),
                (relative["theta"] + relative["alpha"])
                / (relative["beta"] + epsilon),
            ]
        )
        names.extend(
            [
                f"{channel}_log_total_power",
                f"{channel}_theta_alpha_ratio",
                f"{channel}_theta_beta_ratio",
                f"{channel}_alpha_beta_ratio",
                f"{channel}_theta_alpha_beta_ratio",
            ]
        )

    features = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features generated from {path}")
    metadata = {
        "labels": labels,
        "quality_flags": quality_flags,
        "start_seconds": starts,
    }
    return features, names, metadata


def main() -> None:
    args = parse_args()
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    subject_blocks: list[np.ndarray] = []
    quality_blocks: list[np.ndarray] = []
    start_blocks: list[np.ndarray] = []
    expected_names: list[str] | None = None

    for subject in args.subjects:
        path = (
            REPOSITORY_ROOT
            / "data"
            / "processed"
            / f"participant_{subject}"
            / "eeg_windows_4ch_5s.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Processed file not found: {path}")
        features, names, metadata = features_for_subject(path)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f"Feature columns differ for participant {subject}")

        feature_blocks.append(features)
        label_blocks.append(metadata["labels"])
        quality_blocks.append(metadata["quality_flags"])
        start_blocks.append(metadata["start_seconds"])
        subject_blocks.append(np.full(features.shape[0], subject, dtype="U2"))
        print(f"Participant {subject}: {features.shape[0]} windows", flush=True)

    output_dir = REPOSITORY_ROOT / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "frequency_features_4ch_5s.npz"
    np.savez_compressed(
        output_path,
        X=np.vstack(feature_blocks),
        y=np.concatenate(label_blocks),
        subjects=np.concatenate(subject_blocks),
        quality_flags=np.concatenate(quality_blocks),
        start_seconds=np.concatenate(start_blocks),
        feature_names=np.asarray(expected_names),
    )
    print(f"Feature matrix: {sum(len(block) for block in feature_blocks)} windows x {len(expected_names or [])} features")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
