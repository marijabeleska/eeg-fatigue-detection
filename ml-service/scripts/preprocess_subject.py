"""Preprocess one MPD-DF subject into labeled low-channel EEG windows."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import mne
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBJECT = "01"
DEFAULT_CHANNELS = ("C3", "C4", "O1", "O2")


@dataclass(frozen=True)
class Transition:
    onset_seconds: float
    segment: int
    raw_label: int | None
    label_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--l-freq", type=float, default=0.5)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument(
        "--artifact-peak-to-peak-uv",
        type=float,
        default=200.0,
        help=(
            "Mark, but do not remove, windows whose maximum channel peak-to-peak "
            "amplitude exceeds this threshold."
        ),
    )
    parser.add_argument(
        "--input-scale-to-volts",
        type=float,
        default=1e-6,
        help=(
            "Scale applied to EDF samples before filtering. MPD-DF stores values in "
            "microvolts but its EDF physical-unit text is corrupted, so 1e-6 converts "
            "the samples to volts."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("binary", "five-class"),
        default="binary",
        help="binary maps fatigue levels 1-4 to 1; five-class preserves 0-4.",
    )
    return parser.parse_args()


def subject_paths(subject: str) -> tuple[Path, Path, Path]:
    participant_dir = REPOSITORY_ROOT / "data" / "raw" / f"participant_{subject}"
    eeg_path = participant_dir / f"MPDDF_raw_{subject}_EEG.edf"
    annotation_path = participant_dir / f"MPDDF_raw_{subject}_Annotation.txt"
    output_dir = REPOSITORY_ROOT / "data" / "processed" / f"participant_{subject}"
    return eeg_path, annotation_path, output_dir


def annotation_datetime(clock_text: str, raw_start: datetime) -> datetime:
    parsed_time = time.fromisoformat(clock_text)
    result = datetime.combine(raw_start.date(), parsed_time, tzinfo=raw_start.tzinfo)
    if result < raw_start:
        result += timedelta(days=1)
    return result


def read_transitions(annotation_path: Path, raw_start: datetime) -> list[Transition]:
    transitions: list[Transition] = []
    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise ValueError(f"Invalid annotation at line {line_number}: {line!r}")

        clock_text, segment_text, label_text = parts
        onset = (annotation_datetime(clock_text, raw_start) - raw_start).total_seconds()
        try:
            raw_label = int(label_text)
        except ValueError:
            raw_label = None

        transitions.append(
            Transition(
                onset_seconds=onset,
                segment=int(segment_text),
                raw_label=raw_label,
                label_text=label_text,
            )
        )

    if not transitions:
        raise ValueError(f"No annotations found in {annotation_path}")
    if any(current.onset_seconds >= following.onset_seconds for current, following in zip(transitions, transitions[1:])):
        raise ValueError("Annotation timestamps must be strictly increasing")
    return transitions


def model_label(raw_label: int, label_mode: str) -> int:
    if raw_label not in range(5):
        raise ValueError(f"Unsupported numeric fatigue label: {raw_label}")
    if label_mode == "binary":
        return 0 if raw_label == 0 else 1
    return raw_label


def build_windows(
    data: np.ndarray,
    sampling_rate: float,
    transitions: list[Transition],
    recording_duration: float,
    window_seconds: float,
    label_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    window_samples = int(round(window_seconds * sampling_rate))
    if window_samples <= 0:
        raise ValueError("Window length must contain at least one sample")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    raw_labels: list[int] = []
    start_seconds: list[float] = []
    skipped_artifact_intervals = 0

    for index, transition in enumerate(transitions):
        interval_start = max(0.0, transition.onset_seconds)
        interval_end = (
            transitions[index + 1].onset_seconds
            if index + 1 < len(transitions)
            else recording_duration
        )
        interval_end = min(interval_end, recording_duration)
        if interval_end <= interval_start:
            continue
        if transition.raw_label is None:
            skipped_artifact_intervals += 1
            continue

        start_sample = int(np.ceil(interval_start * sampling_rate))
        end_sample = int(np.floor(interval_end * sampling_rate))
        available_samples = end_sample - start_sample
        number_of_windows = available_samples // window_samples
        assigned_label = model_label(transition.raw_label, label_mode)

        for window_index in range(number_of_windows):
            window_start = start_sample + window_index * window_samples
            window_end = window_start + window_samples
            windows.append(data[:, window_start:window_end].astype(np.float32, copy=True))
            labels.append(assigned_label)
            raw_labels.append(transition.raw_label)
            start_seconds.append(window_start / sampling_rate)

    if not windows:
        raise ValueError("No complete labeled windows were created")

    return (
        np.stack(windows),
        np.asarray(labels, dtype=np.int8),
        np.asarray(raw_labels, dtype=np.int8),
        np.asarray(start_seconds, dtype=np.float32),
        skipped_artifact_intervals,
    )


def save_metadata(
    path: Path,
    subject: str,
    start_seconds: np.ndarray,
    labels: np.ndarray,
    raw_labels: np.ndarray,
    peak_to_peak_uv: np.ndarray,
    quality_flags: np.ndarray,
    raw_start: datetime,
    window_seconds: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "subject_id",
                "window_index",
                "start_seconds",
                "end_seconds",
                "start_timestamp",
                "raw_label",
                "model_label",
                "peak_to_peak_uv",
                "quality_flag",
            ),
        )
        writer.writeheader()
        for index, (start, label, raw_label, peak_to_peak, quality_flag) in enumerate(
            zip(start_seconds, labels, raw_labels, peak_to_peak_uv, quality_flags)
        ):
            start_value = float(start)
            writer.writerow(
                {
                    "subject_id": subject,
                    "window_index": index,
                    "start_seconds": f"{start_value:.3f}",
                    "end_seconds": f"{start_value + window_seconds:.3f}",
                    "start_timestamp": (
                        raw_start + timedelta(seconds=start_value)
                    ).isoformat(),
                    "raw_label": int(raw_label),
                    "model_label": int(label),
                    "peak_to_peak_uv": f"{float(peak_to_peak):.3f}",
                    "quality_flag": (
                        "possible_artifact" if quality_flag else "good"
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    if args.input_scale_to_volts <= 0:
        raise ValueError("Input scale must be positive")
    if args.artifact_peak_to_peak_uv <= 0:
        raise ValueError("Artifact threshold must be positive")
    eeg_path, annotation_path, output_dir = subject_paths(args.subject)
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

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
    raw.resample(args.target_sfreq, npad="auto", n_jobs=1, verbose="ERROR")

    data = raw.get_data()
    sampling_rate = float(raw.info["sfreq"])
    recording_duration = raw.n_times / sampling_rate
    windows, labels, raw_labels, start_seconds, skipped_artifact_intervals = (
        build_windows(
            data=data,
            sampling_rate=sampling_rate,
            transitions=transitions,
            recording_duration=recording_duration,
            window_seconds=args.window_seconds,
            label_mode=args.label_mode,
        )
    )

    if not np.isfinite(windows).all():
        raise ValueError("Processed EEG contains NaN or infinite values")

    peak_to_peak_uv = (
        np.max(np.ptp(windows, axis=2), axis=1) * 1_000_000
    ).astype(np.float32)
    quality_flags = (
        peak_to_peak_uv > args.artifact_peak_to_peak_uv
    ).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eeg_windows_{len(args.channels)}ch_{args.window_seconds:g}s"
    npz_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}_metadata.csv"

    np.savez_compressed(
        npz_path,
        X=windows,
        y=labels,
        raw_labels=raw_labels,
        start_seconds=start_seconds,
        peak_to_peak_uv=peak_to_peak_uv,
        quality_flags=quality_flags,
        channels=np.asarray(args.channels),
        sampling_rate=np.asarray(sampling_rate, dtype=np.float32),
        window_seconds=np.asarray(args.window_seconds, dtype=np.float32),
        subject_id=np.asarray(args.subject),
    )
    save_metadata(
        metadata_path,
        subject=args.subject,
        start_seconds=start_seconds,
        labels=labels,
        raw_labels=raw_labels,
        peak_to_peak_uv=peak_to_peak_uv,
        quality_flags=quality_flags,
        raw_start=raw_start,
        window_seconds=args.window_seconds,
    )

    label_counts = dict(sorted(Counter(labels.tolist()).items()))
    peak_microvolts = float(np.max(np.abs(windows)) * 1_000_000)
    print("MPD-DF preprocessing complete")
    print(f"Subject: {args.subject}")
    print(f"Channels: {', '.join(args.channels)}")
    print(f"Input scale to volts: {args.input_scale_to_volts:g}")
    print(f"Filter: {args.l_freq:g}-{args.h_freq:g} Hz")
    print(f"Sampling rate: {sampling_rate:g} Hz")
    print(f"Window length: {args.window_seconds:g} seconds")
    print(f"X shape: {windows.shape}")
    print(f"y shape: {labels.shape}")
    print(f"Label counts: {label_counts}")
    print(f"Skipped non-numeric/artifact intervals: {skipped_artifact_intervals}")
    print(f"All values finite: {np.isfinite(windows).all()}")
    print(f"Peak absolute amplitude: {peak_microvolts:.2f} microvolts")
    print(
        f"Possible artifact windows (peak-to-peak > "
        f"{args.artifact_peak_to_peak_uv:g} microvolts): "
        f"{int(quality_flags.sum())}"
    )
    print(f"Saved: {npz_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
