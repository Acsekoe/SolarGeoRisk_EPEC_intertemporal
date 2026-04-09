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
import matplotlib.cm as cm
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENS_DIR   = os.path.join(SCRIPT_DIR, "..", "outputs", "sens")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load iters data from all sens files
# ---------------------------------------------------------------------------
sens_files = sorted(glob.glob(os.path.join(SENS_DIR, "sens_*.xlsx")))

runs = {}
for fpath in sens_files:
    name = os.path.splitext(os.path.basename(fpath))[0]
    # strip "sens_" prefix for cleaner labels
    label = name.replace("sens_", "")
    df = pd.read_excel(fpath, sheet_name="iters")
    runs[label] = df

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

colors = cm.tab10(np.linspace(0, 1, len(runs)))

for (label, df), color in zip(runs.items(), colors):
    ls = "--" if label.endswith("_early") else "-"
    ax.plot(df["iter"], df["r_strat"], label=label, color=color,
            linestyle=ls, linewidth=1.6, marker="o", markersize=3)

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("r_strat  (relative strategy change)", fontsize=11)
ax.set_title("Gauss-Seidel convergence — r_strat across player orderings", fontsize=12)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
ax.set_xlim(1, max(df["iter"].max() for df in runs.values()))
ax.set_ylim(bottom=0)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "convergence_r_strat.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
