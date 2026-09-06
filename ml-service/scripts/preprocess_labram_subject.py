"""Preprocess one MPD-DF subject for pretrained LaBraM fine-tuning.

The output keeps five-second labeled windows, but follows the official LaBraM
signal preparation: 0.1-75 Hz band-pass, 50 Hz notch, 200 Hz and microvolts.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import mne
import numpy as np

from preprocess_subject import build_windows, read_transitions, save_metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNELS = ("C3", "C4", "O1", "O2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--target-sfreq", type=float, default=200.0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=75.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--artifact-peak-to-peak-uv", type=float, default=200.0)
    parser.add_argument(
        "--input-scale-to-volts",
        type=float,
        default=1e-6,
        help="MPD-DF numeric microvolts to volts before MNE filtering.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "labram_processed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even when a valid output already exists.",
    )
    return parser.parse_args()


def paths(subject: str, output_root: Path) -> tuple[Path, Path, Path, Path]:
    raw_dir = REPOSITORY_ROOT / "data" / "raw" / f"participant_{subject}"
    eeg_path = raw_dir / f"MPDDF_raw_{subject}_EEG.edf"
    annotation_path = raw_dir / f"MPDDF_raw_{subject}_Annotation.txt"
    output_dir = output_root / f"participant_{subject}"
    stem = "eeg_windows_4ch_5s_200hz_uv"
    return (
        eeg_path,
        annotation_path,
        output_dir / f"{stem}.npz",
        output_dir / f"{stem}_metadata.csv",
    )


def valid_existing_output(
    npz_path: Path,
    metadata_path: Path,
    subject: str,
    channels: list[str],
    sampling_rate: float,
    window_seconds: float,
) -> bool:
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    required = {
        "X",
        "y",
        "raw_labels",
        "start_seconds",
        "peak_to_peak_uv",
        "quality_flags",
        "channels",
        "sampling_rate",
        "window_seconds",
        "subject_id",
        "unit",
    }
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                return False
            x = data["X"]
            y = data["y"]
            expected_samples = int(round(sampling_rate * window_seconds))
            return bool(
                x.ndim == 3
                and x.shape[0] == y.size
                and x.shape[1:] == (len(channels), expected_samples)
                and [str(value) for value in data["channels"]] == channels
                and np.isclose(float(data["sampling_rate"]), sampling_rate)
                and np.isclose(float(data["window_seconds"]), window_seconds)
                and str(data["subject_id"]) == subject
                and str(data["unit"]) == "uV"
                and np.isfinite(x).all()
            )
    except (OSError, ValueError, KeyError):
        return False


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not 0 < args.l_freq < args.notch_freq < args.h_freq < 250:
        raise ValueError("Expected l_freq < notch_freq < h_freq below raw Nyquist")
    if args.target_sfreq != 200.0:
        raise ValueError("Pretrained LaBraM preparation requires 200 Hz")
    if args.window_seconds * args.target_sfreq % 200:
        raise ValueError("Window must contain a whole number of 200-sample LaBraM patches")

    eeg_path, annotation_path, npz_path, metadata_path = paths(
        args.subject, args.output_root
    )
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    if not args.overwrite and valid_existing_output(
        npz_path,
        metadata_path,
        args.subject,
        args.channels,
        args.target_sfreq,
        args.window_seconds,
    ):
        print(f"Participant {args.subject}: valid output already exists; skipping", flush=True)
        return

    raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose="ERROR")
    raw_start = raw.info["meas_date"]
    if raw_start is None:
        raise ValueError("EDF recording start timestamp is missing")
    missing_channels = sorted(set(args.channels) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Channels missing from EDF: {missing_channels}")

    transitions = read_transitions(annotation_path, raw_start)
    raw.pick(args.channels)
    raw.load_data(verbose="ERROR")
    raw.apply_function(
        lambda values: values * args.input_scale_to_volts,
        picks="eeg",
        channel_wise=False,
        verbose="ERROR",
    )
    raw.filter(args.l_freq, args.h_freq, n_jobs=1, verbose="ERROR")
    raw.notch_filter(args.notch_freq, n_jobs=1, verbose="ERROR")
    raw.resample(args.target_sfreq, npad="auto", n_jobs=1, verbose="ERROR")

    sampling_rate = float(raw.info["sfreq"])
    recording_duration = raw.n_times / sampling_rate
    windows_v, labels, raw_labels, start_seconds, skipped_intervals = build_windows(
        data=raw.get_data(),
        sampling_rate=sampling_rate,
        transitions=transitions,
        recording_duration=recording_duration,
        window_seconds=args.window_seconds,
        label_mode="binary",
    )
    # LaBraM's official dataset preparation returns numeric microvolts.
    windows_uv = (windows_v * 1_000_000.0).astype(np.float32, copy=False)
    if not np.isfinite(windows_uv).all():
        raise ValueError("Processed EEG contains NaN or infinite values")

    peak_to_peak_uv = np.max(np.ptp(windows_uv, axis=2), axis=1).astype(np.float32)
    quality_flags = (peak_to_peak_uv > args.artifact_peak_to_peak_uv).astype(np.uint8)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        npz_path,
        X=windows_uv,
        y=labels,
        raw_labels=raw_labels,
        start_seconds=start_seconds,
        peak_to_peak_uv=peak_to_peak_uv,
        quality_flags=quality_flags,
        channels=np.asarray(args.channels),
        sampling_rate=np.asarray(sampling_rate, dtype=np.float32),
        window_seconds=np.asarray(args.window_seconds, dtype=np.float32),
        subject_id=np.asarray(args.subject),
        unit=np.asarray("uV"),
        l_freq=np.asarray(args.l_freq, dtype=np.float32),
        h_freq=np.asarray(args.h_freq, dtype=np.float32),
        notch_freq=np.asarray(args.notch_freq, dtype=np.float32),
        patch_samples=np.asarray(200, dtype=np.int16),
    )

    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".part")
    save_metadata(
        temporary_metadata,
        subject=args.subject,
        start_seconds=start_seconds,
        labels=labels,
        raw_labels=raw_labels,
        peak_to_peak_uv=peak_to_peak_uv,
        quality_flags=quality_flags,
        raw_start=raw_start,
        window_seconds=args.window_seconds,
    )
    os.replace(temporary_metadata, metadata_path)

    print(
        f"Participant {args.subject}: complete | X={windows_uv.shape} | "
        f"labels={dict(sorted(Counter(labels.tolist()).items()))} | "
        f"artifacts={int(quality_flags.sum())} | saved={npz_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
