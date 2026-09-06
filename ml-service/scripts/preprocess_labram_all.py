"""Run resumable LaBraM preprocessing sequentially for MPD-DF subjects."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{number:02d}" for number in range(1, 31)],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "labram_processed",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    failed: list[str] = []
    print("LaBraM preprocessing started", flush=True)
    print(f"Subjects: {', '.join(args.subjects)}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print("Internet is not required for this process.", flush=True)

    for position, subject in enumerate(args.subjects, start=1):
        print(
            f"\n[{position}/{len(args.subjects)}] Participant {subject}",
            flush=True,
        )
        command = [
            sys.executable,
            str(SCRIPT_DIR / "preprocess_labram_subject.py"),
            "--subject",
            subject,
            "--output-root",
            str(args.output_root),
        ]
        if args.overwrite:
            command.append("--overwrite")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failed.append(subject)
            print(f"Participant {subject}: FAILED (continuing)", flush=True)

    elapsed_minutes = (time.time() - started) / 60
    if failed:
        raise SystemExit(
            f"Finished in {elapsed_minutes:.1f} min; failed: {', '.join(failed)}"
        )
    print(
        f"\nAll {len(args.subjects)} participants completed successfully "
        f"in {elapsed_minutes:.1f} minutes.",
        flush=True,
    )


if __name__ == "__main__":
    main()
