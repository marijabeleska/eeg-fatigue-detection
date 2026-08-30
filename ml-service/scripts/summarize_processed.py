"""Create a compact per-subject summary of preprocessed MPD-DF windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{number:02d}" for number in range(1, 11)],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, int | float | str]] = []

    for subject in args.subjects:
        path = (
            REPOSITORY_ROOT
            / "data"
            / "processed"
            / f"participant_{subject}"
            / "eeg_windows_4ch_5s.npz"
        )
        if not path.is_file():
            print(f"Skipping participant {subject}: processed file not found")
            continue

        with np.load(path, allow_pickle=False) as data:
            labels = data["y"]
            flags = data["quality_flags"].astype(bool)
            alert = int(np.sum(labels == 0))
            fatigue = int(np.sum(labels == 1))
            total = int(labels.size)
            rows.append(
                {
                    "subject_id": subject,
                    "total_windows": total,
                    "alert_windows": alert,
                    "fatigue_windows": fatigue,
                    "fatigue_percent": round(100 * fatigue / total, 2),
                    "possible_artifacts": int(flags.sum()),
                    "artifact_percent": round(100 * flags.mean(), 2),
                }
            )

    if not rows:
        raise SystemExit("No processed participant files were found")

    output_path = REPOSITORY_ROOT / "artifacts" / "processed_subject_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summarized {len(rows)} participant(s)")
    for row in rows:
        print(
            f"{row['subject_id']}: {row['total_windows']} windows, "
            f"fatigue {row['fatigue_percent']}%, "
            f"possible artifacts {row['artifact_percent']}%"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
