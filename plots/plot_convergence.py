"""
plot_convergence.py
===================
Plot r_strat over Gauss-Seidel iterations for all sensitivity runs.
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENS_DIR   = os.path.join(SCRIPT_DIR, "..", "outputs", "sens")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Okabe-Ito colorblind-safe palette (standard in academic publications)
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# ---------------------------------------------------------------------------
# Load iters data from all sens files
# ---------------------------------------------------------------------------
sens_files = sorted(glob.glob(os.path.join(SENS_DIR, "**", "sens_*.xlsx"), recursive=True))

runs = {}
for fpath in sens_files:
    name = os.path.splitext(os.path.basename(fpath))[0]
    # strip "sens_" prefix for cleaner labels
    label = name.replace("sens_", "")
    # drop the run ending in ch_early
    if label.endswith("ch_early"):
        continue
    df = pd.read_excel(fpath, sheet_name="iters")
    runs[label] = df

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(runs))]

for (label, df), color in zip(runs.items(), colors):
    ls = "--" if label.endswith("_early") else "-"
    ax.plot(df["iter"], df["r_strat"], label=label, color=color,
            linestyle=ls, linewidth=1.6, marker="o", markersize=3)

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel(r"$|\Delta\theta_k|$  (relative strategy change)", fontsize=11)
ax.set_title(r"Gauss-Seidel convergence — $|\Delta\theta_k|$ across player orderings", fontsize=12)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
ax.set_xlim(1, max(df["iter"].max() for df in runs.values()))
ax.set_ylim(bottom=0)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "convergence_r_strat.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
