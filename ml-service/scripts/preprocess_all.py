"""Run MPD-DF preprocessing sequentially for multiple subjects."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{number:02d}" for number in range(1, 11)],
        help="Subject IDs, for example: 01 02 03",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed: list[str] = []

    for position, subject in enumerate(args.subjects, start=1):
        print(
            f"\n[{position}/{len(args.subjects)}] Preprocessing participant {subject}",
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "preprocess_subject.py"), "--subject", subject],
            check=False,
        )
        if result.returncode != 0:
            failed.append(subject)

    if failed:
        raise SystemExit(
            "Preprocessing failed for participant(s): " + ", ".join(failed)
        )
    print("\nAll requested participants were preprocessed successfully.")


if __name__ == "__main__":
    main()
