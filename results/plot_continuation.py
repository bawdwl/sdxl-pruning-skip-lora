"""Plot the recorded continuation comparison without accessing model data."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with (HERE / "continuation_comparison.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))

ratios = (20, 30, 40)
methods = ("ordinary_continuation", "skip_aware_latent_only")
values = {
    (int(row["pruning_percent"]), row["method"]): float(row["lpips_to_full"])
    for row in rows
}
fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
width = 0.32
for offset, method, label, color in (
    (-width / 2, methods[0], "Ordinary continuation", "#3d6a91"),
    (width / 2, methods[1], "Skip-aware continuation", "#e0784f"),
):
    bars = ax.bar([i + offset for i in range(len(ratios))],
                  [values[(ratio, method)] for ratio in ratios], width,
                  label=label, color=color)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.set_xticks(range(len(ratios)), [f"{ratio}%" for ratio in ratios])
ax.set_xlabel("OBS pruning ratio")
ax.set_ylabel("LPIPS to Full SDXL (lower is better)")
ax.set_ylim(0, 0.49)
ax.set_title("Continuation comparison on 100 evaluation prompts")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(HERE / "continuation_lpips.png", dpi=180)
