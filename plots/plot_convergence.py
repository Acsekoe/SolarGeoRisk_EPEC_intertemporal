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

# Match IEEEtran serif rendering (Times-like) so the figure typography blends
# with the surrounding body text and caption in the compiled PDF.
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENS_DIR   = os.path.join(SCRIPT_DIR, "..", "outputs", "sens")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "IEEE Paper", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# Dark2 colorblind-safe palette (darker variants for converged runs)
DARK_PALETTE = [
    "#1B9E77",  # dark teal
    "#D95F02",  # dark orange
    "#7570B3",  # dark purple
    "#E7298A",  # dark magenta
    "#66A61E",  # dark green
    "#A6761D",  # dark brown
]

# Light grey + dashed style for all non-converged runs
NONCONV_COLOR = "#B0B0B0"
NONCONV_STYLE = (0, (5, 2))

# ---------------------------------------------------------------------------
# Load iters data from all sens files, separated by convergence status
# ---------------------------------------------------------------------------
def load_runs(subdir):
    files = sorted(glob.glob(os.path.join(SENS_DIR, subdir, "sens_*.xlsx")))
    out = {}
    for fpath in files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        label = name.replace("sens_", "")
        if label.endswith("ch_early"):
            continue
        out[label] = pd.read_excel(fpath, sheet_name="iters")
    return out

converged_runs    = load_runs("converged")
nonconverged_runs = load_runs("not_converged")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# Source figure ~2x the paper render width (\includegraphics width=0.5\textwidth ~ 3.58in),
# with fonts roughly doubled so that after LaTeX scales the figure to ~0.5x, the rendered
# text lands near IEEE body/caption sizes (~7-9pt).
fig, ax = plt.subplots(figsize=(7.0, 3.8))

# Converged: solid lines, dark colors
conv_handles = []
for i, (label, df) in enumerate(converged_runs.items()):
    color = DARK_PALETTE[i % len(DARK_PALETTE)]
    line, = ax.plot(df["iter"], df["r_strat"], label=label, color=color,
                    linestyle="-", linewidth=1.8, marker="o", markersize=3.5, zorder=3)
    conv_handles.append(line)

# Non-converged: uniform light grey, dashed, plotted underneath
nonconv_handles = []
for label, df in nonconverged_runs.items():
    line, = ax.plot(df["iter"], df["r_strat"], label=label, color=NONCONV_COLOR,
                    linestyle=NONCONV_STYLE, linewidth=1.4, marker="o", markersize=2.5,
                    alpha=0.9, zorder=2)
    nonconv_handles.append(line)

ax.set_xlabel("iterations", fontsize=20)
ax.set_ylabel(r"$\Delta\theta$", fontsize=20)
ax.tick_params(axis="both", labelsize=15)

# Single legend with two column headers ("Converged" / "Non-converged")
from matplotlib.lines import Line2D
header_proxy = Line2D([0], [0], color="none")

# matplotlib fills ncol=2 in column-major order, so each column needs the same
# number of rows (pad converged column with one blank).
n_rows = max(len(conv_handles), len(nonconv_handles)) + 1  # +1 for header
conv_col    = [header_proxy] + conv_handles    + [header_proxy] * (n_rows - 1 - len(conv_handles))
conv_labels = ["Converged"]  + [h.get_label() for h in conv_handles] + [""] * (n_rows - 1 - len(conv_handles))
nc_col      = [header_proxy] + nonconv_handles + [header_proxy] * (n_rows - 1 - len(nonconv_handles))
nc_labels   = ["Non-converged"] + [h.get_label() for h in nonconv_handles] + [""] * (n_rows - 1 - len(nonconv_handles))

leg = ax.legend(conv_col + nc_col, conv_labels + nc_labels,
                ncol=2, loc="upper right", bbox_to_anchor=(0.98, 1.01),
                fontsize=13, framealpha=0.9, handletextpad=0.4,
                columnspacing=0.9, borderpad=0.5, labelspacing=0.4,
                markerscale=0.9)
# Bold the two header rows
texts = leg.get_texts()
texts[0].set_fontweight("bold")        # "Converged" header
texts[n_rows].set_fontweight("bold")   # "Non-converged" header
all_runs = {**converged_runs, **nonconverged_runs}
ax.set_xlim(1, max(df["iter"].max() for df in all_runs.values()))
ax.set_ylim(bottom=0)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "RESULT_convergence_pattern.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
