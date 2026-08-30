"""Create quality-control plots and band-power summaries for processed EEG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = {0: "Alert", 1: "Fatigue"}
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
BAND_COLORS = {
    "delta": "#dbeafe",
    "theta": "#dcfce7",
    "alpha": "#fef3c7",
    "beta": "#fee2e2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="01")
    return parser.parse_args()


def processed_path(subject: str) -> Path:
    return (
        REPOSITORY_ROOT
        / "data"
        / "processed"
        / f"participant_{subject}"
        / "eeg_windows_4ch_5s.npz"
    )


def representative_index(windows: np.ndarray, labels: np.ndarray, label: int) -> int:
    indices = np.flatnonzero(labels == label)
    if not len(indices):
        raise ValueError(f"No windows found for class {label}")
    rms = np.sqrt(np.mean(np.square(windows[indices], dtype=np.float64), axis=(1, 2)))
    median_rms = np.median(rms)
    return int(indices[np.argmin(np.abs(rms - median_rms))])


def plot_representative_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    channels: list[str],
    sampling_rate: float,
    output_path: Path,
) -> tuple[int, int]:
    selected = {
        label: representative_index(windows, labels, label) for label in CLASS_NAMES
    }
    time_seconds = np.arange(windows.shape[-1]) / sampling_rate
    figure, axes = plt.subplots(
        len(channels),
        len(CLASS_NAMES),
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    for column, (label, class_name) in enumerate(CLASS_NAMES.items()):
        window = windows[selected[label]] * 1_000_000
        for row, channel in enumerate(channels):
            axis = axes[row, column]
            axis.plot(time_seconds, window[row], color="#2563eb" if label == 0 else "#dc2626", linewidth=0.8)
            axis.axhline(0, color="#9ca3af", linewidth=0.5)
            axis.grid(alpha=0.2)
            axis.set_ylabel(f"{channel}\nµV")
            if row == 0:
                axis.set_title(f"{class_name} — window {selected[label]}")
            if row == len(channels) - 1:
                axis.set_xlabel("Time (seconds)")

    figure.suptitle("MPD-DF participant 01: representative 5-second EEG windows")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return selected[0], selected[1]


def compute_psd(
    windows: np.ndarray, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, psd = welch(
        windows,
        fs=sampling_rate,
        nperseg=min(256, windows.shape[-1]),
        noverlap=min(128, windows.shape[-1] // 2),
        axis=-1,
        scaling="density",
    )
    return frequencies, psd


def plot_class_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for band_name, (low, high) in BANDS.items():
        axis.axvspan(low, high, color=BAND_COLORS[band_name], alpha=0.65, label=band_name)

    line_styles = {
        0: ("#1d4ed8", "-"),
        1: ("#b91c1c", "-"),
    }
    for label, class_name in CLASS_NAMES.items():
        class_psd = psd[labels == label].mean(axis=(0, 1)) * 1e12
        color, style = line_styles[label]
        axis.semilogy(
            frequencies,
            class_psd,
            color=color,
            linestyle=style,
            linewidth=2,
            label=f"{class_name} mean PSD",
        )

    axis.set_xlim(1, 40)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density (µV²/Hz, log scale)")
    axis.set_title("MPD-DF participant 01: alert vs fatigue spectrum")
    handles, legend_labels = axis.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    axis.legend(by_label.values(), by_label.keys(), ncols=3, fontsize=8)
    axis.grid(alpha=0.25, which="both")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_band_power_summary(
    frequencies: np.ndarray,
    psd: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> list[dict[str, str | int]]:
    total_mask = (frequencies >= 1.0) & (frequencies <= 40.0)
    total_power = np.trapezoid(psd[..., total_mask], frequencies[total_mask], axis=-1)
    rows: list[dict[str, str | int]] = []

    for label, class_name in CLASS_NAMES.items():
        class_mask = labels == label
        for band_name, (low, high) in BANDS.items():
            band_mask = (frequencies >= low) & (frequencies < high)
            band_power = np.trapezoid(
                psd[..., band_mask], frequencies[band_mask], axis=-1
            )
            relative_power = np.divide(
                band_power,
                total_power,
                out=np.zeros_like(band_power),
                where=total_power > 0,
            )[class_mask]
            rows.append(
                {
                    "class_id": label,
                    "class_name": class_name,
                    "band": band_name,
                    "low_hz": f"{low:g}",
                    "high_hz": f"{high:g}",
                    "mean_relative_power": f"{relative_power.mean():.6f}",
                    "std_relative_power": f"{relative_power.std():.6f}",
                    "window_count": int(class_mask.sum()),
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    args = parse_args()
    input_path = processed_path(args.subject)
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {input_path}")

    with np.load(input_path) as dataset:
        windows = dataset["X"]
        labels = dataset["y"]
        channels = dataset["channels"].tolist()
        sampling_rate = float(dataset["sampling_rate"])

    output_dir = REPOSITORY_ROOT / "artifacts" / "qc" / f"participant_{args.subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "eeg_representative_windows.png"
    psd_path = output_dir / "psd_alert_vs_fatigue.png"
    band_summary_path = output_dir / "band_power_summary.csv"

    alert_index, fatigue_index = plot_representative_windows(
        windows, labels, channels, sampling_rate, examples_path
    )
    frequencies, psd = compute_psd(windows, sampling_rate)
    plot_class_psd(frequencies, psd, labels, psd_path)
    rows = save_band_power_summary(frequencies, psd, labels, band_summary_path)

    print("QC artifacts created")
    print(f"Representative alert window: {alert_index}")
    print(f"Representative fatigue window: {fatigue_index}")
    print(f"Saved: {examples_path}")
    print(f"Saved: {psd_path}")
    print(f"Saved: {band_summary_path}")
    print("Mean relative band powers:")
    for row in rows:
        print(
            f"  {row['class_name']:>7} | {row['band']:<5} | "
            f"{float(row['mean_relative_power']) * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()

