"""Create a clean presentation figure for all MPD-DF model results."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


RESULTS = [
    {"name": "LR · O1",          "family": "Класични", "bacc": .658, "sens": .522, "spec": .807, "f1": .341, "auc": .722},
    {"name": "RF · O1",          "family": "Класични", "bacc": .655, "sens": .387, "spec": .933, "f1": .344, "auc": .693},
    {"name": "LR · O1/O2",       "family": "Класични", "bacc": .654, "sens": .513, "spec": .808, "f1": .333, "auc": .717},
    {"name": "RF · O1/O2",       "family": "Класични", "bacc": .654, "sens": .377, "spec": .940, "f1": .344, "auc": .682},
    {"name": "LR · 4 канали",    "family": "Класични", "bacc": .649, "sens": .498, "spec": .810, "f1": .340, "auc": .730},
    {"name": "RF · 4 канали",    "family": "Класични", "bacc": .655, "sens": .364, "spec": .954, "f1": .350, "auc": .714},
    {"name": "EEGNet · O1",      "family": "EEGNet",   "bacc": .655, "sens": .528, "spec": .797, "f1": .343, "auc": .758},
    {"name": "EEGNet · O1/O2",   "family": "EEGNet",   "bacc": .636, "sens": .502, "spec": .786, "f1": .328, "auc": .727},
    {"name": "EEGNet · 4 канали", "family": "EEGNet",  "bacc": .649, "sens": .472, "spec": .843, "f1": .305, "auc": .747},
    {"name": "LaBraM · O1",      "family": "LaBraM",   "bacc": .737, "sens": .561, "spec": .913, "f1": .498, "auc": .793},
]

COLORS = {"Класични": "#7A8594", "EEGNet": "#2563A9", "LaBraM": "#D66A2C"}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#28323C",
        "xtick.color": "#53606D",
        "ytick.color": "#28323C",
    })

    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="white")
    grid = fig.add_gridspec(1, 2, left=.075, right=.965, top=.80, bottom=.18,
                           width_ratios=[1.15, 1], wspace=.34)
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_trade = fig.add_subplot(grid[0, 1])

    ordered = sorted(RESULTS, key=lambda d: (d["family"] != "LaBraM", -d["bacc"]))
    y = np.arange(len(ordered))[::-1]

    for yi, item in zip(y, ordered):
        color = COLORS[item["family"]]
        ax_rank.plot([item["bacc"], item["auc"]], [yi, yi], color="#CDD3DA", lw=2, zorder=1)
        ax_rank.scatter(item["bacc"], yi, s=78, marker="o", color=color,
                        edgecolor="white", linewidth=1.2, zorder=3)
        ax_rank.scatter(item["auc"], yi, s=82, marker="D", facecolor="white",
                        edgecolor=color, linewidth=1.8, zorder=3)
        ax_rank.text(item["bacc"] - .004, yi + .20, f'{item["bacc"]:.3f}',
                     color=color, ha="right", va="bottom", fontsize=9.5,
                     fontweight="semibold" if item["family"] == "LaBraM" else "normal")
        ax_rank.text(item["auc"] + .004, yi - .20, f'{item["auc"]:.3f}',
                     color=color, ha="left", va="top", fontsize=9.5,
                     fontweight="semibold" if item["family"] == "LaBraM" else "normal")

    ax_rank.set_yticks(y, [d["name"] for d in ordered])
    for tick, item in zip(ax_rank.get_yticklabels(), ordered):
        if item["family"] == "LaBraM":
            tick.set_color(COLORS["LaBraM"])
            tick.set_fontweight("semibold")
        elif item["name"] == "EEGNet · O1":
            tick.set_fontweight("semibold")
    ax_rank.set_xlim(.61, .82)
    ax_rank.set_ylim(-.65, len(ordered) - .35)
    ax_rank.set_xlabel("Резултат (0–1)", labelpad=10)
    ax_rank.set_title("Општа успешност", loc="left", fontsize=16, pad=18)
    ax_rank.grid(axis="x", color="#E7EBEF", lw=.9)
    ax_rank.set_axisbelow(True)
    ax_rank.tick_params(axis="y", length=0, pad=9)

    metric_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4B5563",
               markeredgecolor="white", markersize=8, label="Balanced accuracy"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#4B5563", markeredgewidth=1.5, markersize=7, label="ROC AUC"),
    ]
    ax_rank.legend(handles=metric_legend, loc="lower right", frameon=False,
                   ncol=2, bbox_to_anchor=(1.0, 1.01), fontsize=10)

    label_offsets = {
        "LR · O1": (.006, -.010), "RF · O1": (.006, -.003),
        "LR · O1/O2": (.006, .009), "RF · O1/O2": (.006, .008),
        "LR · 4 канали": (-.006, -.018), "RF · 4 канали": (.006, .005),
        "EEGNet · O1": (.006, .008), "EEGNet · O1/O2": (.006, -.012),
        "EEGNet · 4 канали": (.006, .008), "LaBraM · O1": (-.006, .012),
    }
    for item in RESULTS:
        color = COLORS[item["family"]]
        size = 90 + 760 * (item["f1"] - .28)
        ax_trade.scatter(item["sens"], item["spec"], s=size, color=color, alpha=.92,
                         edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy = label_offsets[item["name"]]
        ha = "right" if dx < 0 else "left"
        weight = "semibold" if item["name"] in {"LaBraM · O1", "EEGNet · O1"} else "normal"
        ax_trade.text(item["sens"] + dx, item["spec"] + dy, item["name"],
                      color=color, fontsize=9.1, ha=ha, va="center", fontweight=weight)

    ax_trade.set_xlim(.34, .60)
    ax_trade.set_ylim(.765, .98)
    ax_trade.set_xlabel("Sensitivity — откриен замор", labelpad=10)
    ax_trade.set_ylabel("Specificity — препознаена будност", labelpad=10)
    ax_trade.set_title("Рамнотежа при класификација", loc="left", fontsize=16, pad=18)
    ax_trade.grid(color="#E7EBEF", lw=.9)
    ax_trade.set_axisbelow(True)

    for ax in (ax_rank, ax_trade):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C8CED5")
        ax.spines["bottom"].set_color("#C8CED5")

    family_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[f],
               markeredgecolor="white", markersize=9, label=f)
        for f in ("Класични", "EEGNet", "LaBraM")
    ]
    ax_trade.legend(handles=family_legend, loc="lower left", frameon=False,
                    ncol=1, bbox_to_anchor=(0.01, 0.01), fontsize=9.5,
                    handletextpad=.5, labelspacing=.45)

    fig.text(.075, .93, "Детекција на когнитивен замор со нискоканален EEG",
             fontsize=25, fontweight="semibold", color="#17212B")
    fig.text(.075, .875,
             "MPD-DF · 30 испитаници · споредба на класични, deep-learning и foundation модели",
             fontsize=13.5, color="#5B6773")

    fig.text(.075, .105,
             "Најдобар забележан резултат: LaBraM · O1 — balanced accuracy 0.737, "
             "sensitivity 0.561, specificity 0.913, F1 0.498, ROC AUC 0.793.",
             fontsize=11.2, fontweight="semibold", color=COLORS["LaBraM"])
    fig.text(.075, .064,
             "Евалуација: класични модели и EEGNet = 30-fold LOSO; LaBraM = фиксна subject-independent "
             "поделба (train 01–24, validation 25–27, test 28–30).",
             fontsize=9.8, color="#65717C")
    fig.text(.075, .035,
             "LaBraM има најдобри забележани метрики, но директната споредба се толкува внимателно "
             "бидејќи протоколите не се идентични. Големината на точките десно го претставува F1-score.",
             fontsize=9.8, color="#65717C")

    png = args.output_dir / "model_results_presentation.png"
    pdf = args.output_dir / "model_results_presentation.pdf"
    fig.savefig(png, dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
