"""Download selected MPD-DF raw EEG and annotation files from Figshare."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/28455737"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"{subject:02d}" for subject in range(1, 11)],
        help="Two-digit participant IDs, for example: 01 02 03.",
    )
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file_catalog() -> dict[str, dict[str, object]]:
    response = requests.get(FIGSHARE_API_URL, timeout=60)
    response.raise_for_status()
    return {file_info["name"]: file_info for file_info in response.json()["files"]}


def download_file(file_info: dict[str, object], destination: Path) -> None:
    expected_md5 = str(file_info["supplied_md5"]).lower()
    expected_size = int(file_info["size"])
    if destination.is_file():
        if destination.stat().st_size == expected_size and md5sum(destination) == expected_md5:
            print(f"Verified existing: {destination.name}")
            return
        raise ValueError(
            f"Existing file failed size/checksum validation: {destination}. "
            "Move it aside before retrying."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()
    bytes_written = 0
    try:
        with requests.get(str(file_info["download_url"]), stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial_path.open("wb") as file_handle:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    file_handle.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
        if bytes_written != expected_size:
            raise ValueError(
                f"Size mismatch for {destination.name}: {bytes_written} != {expected_size}"
            )
        if digest.hexdigest() != expected_md5:
            raise ValueError(f"MD5 mismatch for {destination.name}")
        partial_path.replace(destination)
        print(f"Downloaded and verified: {destination.name} ({bytes_written / 1e6:.1f} MB)")
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    subjects = [f"{int(subject):02d}" for subject in args.subjects]
    catalog = fetch_file_catalog()

    for subject in subjects:
        participant_dir = REPOSITORY_ROOT / "data" / "raw" / f"participant_{subject}"
        required_names = (
            f"MPDDF_raw_{subject}_EEG.edf",
            f"MPDDF_raw_{subject}_Annotation.txt",
        )
        print(f"Participant {subject}")
        for file_name in required_names:
            if file_name not in catalog:
                raise KeyError(f"File not listed by Figshare: {file_name}")
            download_file(catalog[file_name], participant_dir / file_name)


if __name__ == "__main__":
    main()

