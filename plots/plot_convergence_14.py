"""Plot Stage-1 strategy movement and Stage-2 equilibrium-test gains for the 14 retained equilibria.

Panel (a) shows the penalized Stage-1 Gauss--Seidel movement for all seven update
orders, with the damping factor of the converged orders below. Panel (b) shows the
largest frozen-profile one-start unilateral gain after every unpenalized Stage-2
sweep of the 14 retained profiles (sweep 0 is the restart point).

The Stage-1 histories live outside the repository (``_MOVE/new_equilibria``). On the
first run their ``iters`` sheets are copied into ``stage1_convergence.csv`` in the
14-equilibria folder, which later runs read instead.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE1_ARCHIVE = ROOT.parent / "_MOVE/new_equilibria/penalized_profiles"
STAGE2 = ROOT / "outputs/clean_stage2_factorial_20260923_123037"
FOLDER = ROOT / "outputs/paper_plots/14_equilibria"
OVERLEAF_FIGURES = ROOT / "IEEE Paper/images/results"
STAGE1_CSV = FOLDER / "stage1_convergence.csv"
STAGE2_CSV = FOLDER / "stage2_equilibrium_test.csv"
STEM = "convergence_two_stage"
TOL = 1.0  # percent, both epsilon_conv and epsilon_eq
# Converged Stage-1 orders, which are also the orders of the 14 equilibria.
ORDERS = {
    "ch-af-apac-eu-row-us": ("CH-first", "#2a78d6", "-"),
    "af-eu-us-apac-row-ch": ("AF-first", "#eb6834", "--"),
    "eu-us-af-row-apac-ch": ("EU-first", "#1baf7a", ":"),
}
GREY = "#b5b3ac"
INK = "#343434"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def global_sweeps(workbook: Path) -> tuple[int, int]:
    first, last = map(int, re.search(r"_(\d{3})_(\d{3})\.xlsx$", workbook.name).groups())
    return first, last


def stage1() -> pd.DataFrame:
    if STAGE1_CSV.exists():
        return pd.read_csv(STAGE1_CSV)
    index = pd.read_csv(STAGE1_ARCHIVE / "profile_index.csv")
    frames = []
    for row in index.itertuples(index=False):
        for name in (row.initial_workbook, row.continuation_workbook):
            if not isinstance(name, str):
                continue
            workbook = STAGE1_ARCHIVE / name
            iters = pd.read_excel(workbook, sheet_name="iters")
            first, last = global_sweeps(workbook)
            if len(iters) != last - first + 1:
                raise ValueError(f"{workbook} has {len(iters)} sweeps, expected {first}-{last}")
            frames.append(iters.assign(sequence=row.sequence, sweep=range(first, last + 1),
                                       status=row.final_status))
    data = pd.concat(frames, ignore_index=True)
    data = data[["sequence", "status", "sweep", "r_strat", "omega", "c_pen_p", "c_pen_dk",
                 "stable_count", "omega_reason"]]
    data.to_csv(STAGE1_CSV, index=False)
    return data


def retained() -> list[str]:
    lines = (FOLDER / "retained_profiles.txt").read_text(encoding="utf-8").splitlines()
    ids = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError(f"Expected 14 unique retained profiles, got {len(ids)}")
    return ids


def stage2(ids: list[str]) -> pd.DataFrame:
    rows = []
    for candidate in ids:
        audits = STAGE2 / candidate / "audits"
        paths = [audits / "audit_initial_one_start.json",
                 *sorted(audits.glob("audit_sweep_*_one_start.json"))]
        for sweep, path in enumerate(paths):
            audit = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"candidate": candidate, "sequence": candidate.split("/")[0],
                         "sweep": sweep, "gain_percent": 100 * audit["max_relative_gain"],
                         "player": audit["max_gain_player"]})
    data = pd.DataFrame(rows)
    final = data.sort_values("sweep").groupby("candidate").last()
    if (final.gain_percent >= TOL).any():
        raise ValueError("A retained profile ends above the equilibrium tolerance")
    data.to_csv(STAGE2_CSV, index=False)
    return data


def highlights() -> dict[str, str]:
    drivers = pd.read_csv(FOLDER / "price_supply_drivers.csv")
    return dict(drivers.dropna(subset=["label"])[["candidate", "label"]].values)


def style(ax) -> None:
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot(s1: pd.DataFrame, s2: pd.DataFrame, marked: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.4, 4.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(3.2, 1, 1), width_ratios=(1.25, 1),
                            hspace=0.14, wspace=0.28)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_w = fig.add_subplot(grid[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(grid[2, 0], sharex=ax_a)
    ax_b = fig.add_subplot(grid[:, 1])

    # (a) Stage-1 movement: non-converged orders first so the converged ones sit on top.
    for sequence, path in s1.groupby("sequence"):
        if sequence in ORDERS:
            continue
        ax_a.plot(path.sweep, 100 * path.r_strat, color=GREY, linewidth=1.2, zorder=2)
    for sequence, (label, color, dash) in ORDERS.items():
        path = s1[s1.sequence == sequence]
        ax_a.plot(path.sweep, 100 * path.r_strat, color=color, linestyle=dash,
                  linewidth=1.8, zorder=3, label=label)
        ax_w.plot(path.sweep, path.omega, color=color, linestyle=dash, linewidth=1.6)
    ax_a.plot([], [], color=GREY, linewidth=1.2, label="Not converged")
    ax_a.axhline(TOL, color=INK, linestyle="--", linewidth=1.0, zorder=1)
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\Delta\theta$ [%]", fontsize=13)
    ax_a.set_title("(a) Penalized Stage 1", fontsize=13, pad=4)
    ax_w.set_ylabel(r"$\omega_k$", fontsize=13)
    ax_w.set_ylim(0.3, 0.9)
    ax_w.set_yticks([0.4, 0.8])
    # The penalty ramp is identical for every update order, so one path is drawn.
    ramp = s1[s1.sequence == next(iter(ORDERS))]
    ax_c.plot(ramp.sweep, ramp.c_pen_p, color=INK, linewidth=1.4)
    ax_c.plot(ramp.sweep, ramp.c_pen_dk, color=INK, linewidth=1.4, linestyle="--")
    ax_c.text(ramp.sweep.iloc[-1] + 0.8, ramp.c_pen_p.iloc[-1], r"$p^{\mathrm{offer}}$",
              fontsize=10, va="center", color=INK)
    ax_c.text(ramp.sweep.iloc[-1] + 0.8, ramp.c_pen_dk.iloc[-1], r"$\Delta K$",
              fontsize=10, va="center", color=INK)
    ax_c.set_ylabel(r"$c^{\mathrm{pen}}_{\theta,k}$", fontsize=13)
    ax_c.set_ylim(0, 3.6)
    ax_c.set_yticks([0, 3])
    ax_c.set_xlabel("Sweep", fontsize=13)
    for ax in (ax_a, ax_w):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (ax_a, ax_w, ax_c):
        # Continuation runs of the slower orders start after sweep 30.
        ax.axvline(30.5, color=GREY, linewidth=0.8, linestyle="-", zorder=0)

    # (b) Stage-2 equilibrium test; Eq 1 and Eq 2 are drawn last and labelled at the end.
    for candidate, path in sorted(s2.groupby("candidate"), key=lambda item: item[0] in marked):
        label, color, dash = ORDERS[candidate.split("/")[0]]
        emphasis = candidate in marked
        ax_b.plot(path.sweep, path.gain_percent, color=color, linestyle=dash,
                  linewidth=2.0 if emphasis else 1.1, alpha=1.0 if emphasis else 0.7,
                  marker="o" if emphasis else None, markersize=3.5, zorder=4 if emphasis else 2)
        if emphasis:
            end = path.iloc[-1]
            ax_b.annotate(marked[candidate], (end.sweep, end.gain_percent), xytext=(4, -9),
                          textcoords="offset points", fontsize=11, color=INK, zorder=5,
                          bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"})
    ax_b.axhline(TOL, color=INK, linestyle="--", linewidth=1.0, zorder=1)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("Max. unilateral gain [%]", fontsize=13)
    ax_b.set_xlabel("Sweep", fontsize=13)
    ax_b.set_title("(b) Unpenalized Stage 2", fontsize=13, pad=4)

    for ax in (ax_a, ax_w, ax_c, ax_b):
        style(ax)
    for ax in (ax_a, ax_b):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.13), handlelength=2.4)

    for extension in ("pdf", "png"):
        path = FOLDER / f"{STEM}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        shutil.copy2(path, OVERLEAF_FIGURES / path.name)
    plt.close(fig)


def main() -> None:
    s1 = stage1()
    s2 = stage2(retained())
    plot(s1, s2, highlights())
    first = s1[s1.r_strat < TOL / 100].groupby("sequence").sweep.min()
    print("Stage 1 final movement [%]:")
    print((100 * s1.sort_values("sweep").groupby("sequence").r_strat.last()).round(2).to_string())
    print("First sweep below tolerance:", first.to_dict())
    summary = s2.groupby("candidate").agg(sweeps=("sweep", "max"), start=("gain_percent", "first"),
                                          peak=("gain_percent", "max"), end=("gain_percent", "last"))
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
