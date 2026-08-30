"""Inspect the MPD-DF pilot recording without preloading it into RAM."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mne


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EEG_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "participant_01"
    / "MPDDF_raw_01_EEG.edf"
)
DEFAULT_ANNOTATION_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "participant_01"
    / "MPDDF_raw_01_Annotation.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", type=Path, default=DEFAULT_EEG_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION_PATH)
    return parser.parse_args()


def read_label_counts(annotation_path: Path) -> Counter[str]:
    labels: Counter[str] = Counter()
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        timestamp, segment, label = (part.strip() for part in line.split(",", 2))
        labels[label] += 1
        print(f"  {timestamp} | segment {segment:>3} | label {label}")
    return labels


def main() -> None:
    args = parse_args()
    if not args.eeg.is_file():
        raise FileNotFoundError(f"EEG file not found: {args.eeg}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")

    raw = mne.io.read_raw_edf(args.eeg, preload=False, verbose="ERROR")
    sampling_rate = float(raw.info["sfreq"])
    duration_seconds = raw.n_times / sampling_rate

    print("MPD-DF EDF metadata")
    print(f"File: {args.eeg}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {sampling_rate:g} Hz")
    print(f"Samples per channel: {raw.n_times}")
    print(f"Duration: {duration_seconds / 60:.2f} minutes")
    print("Channel names and inferred types:")
    for index, (name, channel_type) in enumerate(
        zip(raw.ch_names, raw.get_channel_types()), start=1
    ):
        print(f"  {index:02d}. {name} ({channel_type})")

    print("Annotation transitions:")
    label_counts = read_label_counts(args.annotations)
    print(f"Transition-label counts: {dict(sorted(label_counts.items()))}")


if __name__ == "__main__":
    main()

