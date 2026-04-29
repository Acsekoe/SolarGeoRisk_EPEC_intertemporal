"""
plot_welfare.py
===============
Paper figure for regional welfare under the global planner and the selected
strategic EPEC run.

The figure shows the cumulative present-value welfare change and the annual
period-by-period welfare change. It intentionally has no plot titles so the
caption can carry the narrative in the paper.
"""

import os
import sys
import contextlib
import io
import ast

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Match IEEEtran serif rendering used by plot_convergence.py.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
INPUT_PATH = os.path.join(ROOT_DIR, "inputs", "input_data_intertemporal.xlsx")
PLANNER_PATH = os.path.join(ROOT_DIR, "outputs", "llp_planner_results.xlsx")
EPEC_PATH = os.path.join(
    ROOT_DIR,
    "outputs",
    "sens",
    "converged",
    "sens_ch-row-apac-us-eu-af.xlsx",
)
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SELECTED_RUN = "sens_ch-row-apac-us-eu-af"
PERIODS = ["2025", "2030", "2035", "2040"]
REGIONS = ["ch", "eu", "us", "apac", "af", "row"]
REGION_NAMES = {
    "ch": "China",
    "eu": "Europe",
    "us": "United States",
    "apac": "Asia-Pacific",
    "af": "Africa",
    "row": "Rest of World",
}

COLOR_GAIN = "#2E7D32"
COLOR_LOSS = "#C62828"
COLOR_PLAN = "#BDBDBD"
COLOR_GRID = "#D0D0D0"

# The welfare workbook is in million USD: USD/kW times GW equals 1e6 USD.
MUSD_TO_TUSD = 1e6
MUSD_TO_BUSD = 1e3


class SplitLegendKey:
    def __init__(self, gain_color, loss_color):
        self.gain_color = gain_color
        self.loss_color = loss_color


class SplitLegendHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        half_width = width / 2
        gain_rect = Rectangle(
            (xdescent, ydescent),
            half_width,
            height,
            facecolor=orig_handle.gain_color,
            edgecolor="white",
            linewidth=0.8,
            transform=trans,
        )
        loss_rect = Rectangle(
            (xdescent + half_width, ydescent),
            half_width,
            height,
            facecolor=orig_handle.loss_color,
            edgecolor="white",
            linewidth=0.8,
            transform=trans,
        )
        border = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor="none",
            edgecolor="#BBBBBB",
            linewidth=0.6,
            transform=trans,
        )
        return [gain_rect, loss_rect, border]


def fmt_signed(value, digits=2):
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}"


def load_model_data():
    sys.path.insert(0, os.path.join(ROOT_DIR, "model"))
    from data_prep import load_data_from_excel

    # data_prep prints calibration summaries; keep plot generation quiet.
    with contextlib.redirect_stdout(io.StringIO()):
        return load_data_from_excel(INPUT_PATH, params_region_sheet="params_region_new")


def add_demand_parameters(df_reg, data):
    df_reg = df_reg.copy()
    df_reg["t"] = df_reg["t"].astype(str)
    a_dem = dict(data.a_dem_t) if data.a_dem_t else {
        (r, t): float(data.a_dem.get(r, 0.0))
        for r in data.regions
        for t in data.times
    }
    b_dem = dict(data.b_dem_t) if data.b_dem_t else {
        (r, t): float(data.b_dem.get(r, 1.0))
        for r in data.regions
        for t in data.times
    }
    if "a_dem_used" not in df_reg.columns:
        df_reg["a_dem_used"] = df_reg.apply(
            lambda row: a_dem.get((row["r"], row["t"]), 0.0), axis=1
        )
    if "b_dem_used" not in df_reg.columns:
        df_reg["b_dem_used"] = df_reg.apply(
            lambda row: b_dem.get((row["r"], row["t"]), 0.0), axis=1
        )
    return df_reg


def compute_welfare(df_reg, df_flows, data, run_label):
    df_reg = add_demand_parameters(df_reg, data)
    df_flows = df_flows.copy()
    df_flows["t"] = df_flows["t"].astype(str)

    beta_t = dict(data.beta_t) if data.beta_t else {t: 1.0 for t in PERIODS}
    ytn_t = dict(data.years_to_next) if data.years_to_next else {t: 5.0 for t in PERIODS}
    f_hold = dict(data.f_hold) if data.f_hold else {r: 0.0 for r in REGIONS}
    c_inv = dict(data.c_inv) if data.c_inv else {r: 0.0 for r in REGIONS}
    lam = {(row["r"], row["t"]): float(row["lam"]) for _, row in df_reg.iterrows()}
    cost_col = "c_man_t" if "c_man_t" in df_reg.columns else "c_man_var"
    c_man = {
        (row["r"], row["t"]): float(row[cost_col])
        for _, row in df_reg.iterrows()
    } if cost_col in df_reg.columns else None

    rows = []
    for _, reg in df_reg[df_reg["t"].isin(PERIODS)].iterrows():
        r = reg["r"]
        t = reg["t"]
        x_dem = float(reg["x_dem"])
        gross_cs = float(reg["a_dem_used"]) * x_dem - 0.5 * float(reg["b_dem_used"]) * x_dem**2
        net_cs = gross_cs - float(reg["lam"]) * x_dem

        flows_r = df_flows[(df_flows["exp"] == r) & (df_flows["t"] == t)]
        ps = 0.0
        for _, flow in flows_r.iterrows():
            imp = flow["imp"]
            cost = c_man[(r, t)] if c_man is not None else float(flow["c_man"])
            ps += (
                lam.get((imp, t), 0.0)
                - cost
                - float(flow["c_ship"])
            ) * float(flow["x"])

        cap_cost = f_hold.get(r, 0.0) * float(reg["Kcap"]) + c_inv.get(r, 0.0) * float(reg["Icap_report"])
        welfare = net_cs + ps - cap_cost
        rows.append(
            {
                "run": run_label,
                "r": r,
                "t": t,
                "gross_CS": gross_cs,
                "net_CS": net_cs,
                "PS": ps,
                "CapCost": cap_cost,
                "W": welfare,
                "W_npv": beta_t.get(t, 1.0) * ytn_t.get(t, 5.0) * welfare,
            }
        )
    return pd.DataFrame(rows)


def read_meta_dict(path):
    meta = pd.read_excel(path, sheet_name="meta")
    return dict(zip(meta["key"], meta["value"]))


def parse_mapping(value, fallback):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return fallback


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = load_model_data()
epec_meta = read_meta_dict(EPEC_PATH)
data.beta_t = parse_mapping(epec_meta.get("beta_t"), data.beta_t or {t: 1.0 for t in PERIODS})
data.years_to_next = parse_mapping(
    epec_meta.get("ytn"), data.years_to_next or {t: 5.0 for t in PERIODS}
)
df_plan_reg = pd.read_excel(PLANNER_PATH, sheet_name="regions")
df_plan_flows = pd.read_excel(PLANNER_PATH, sheet_name="flows")
df_epec_reg = pd.read_excel(EPEC_PATH, sheet_name="regions")
df_epec_flows = pd.read_excel(EPEC_PATH, sheet_name="flows")

df_plan = compute_welfare(df_plan_reg, df_plan_flows, data, "planner")
df_epec = compute_welfare(df_epec_reg, df_epec_flows, data, SELECTED_RUN)

summary_plan = df_plan.groupby("r")["W_npv"].sum().loc[REGIONS]
summary_epec = df_epec.groupby("r")["W_npv"].sum().loc[REGIONS]

df_total = pd.DataFrame(
    {
        "region": [REGION_NAMES[r] for r in REGIONS],
        "planner": summary_plan / MUSD_TO_TUSD,
        "epec": summary_epec / MUSD_TO_TUSD,
        "delta": (summary_epec - summary_plan) / MUSD_TO_TUSD,
        "delta_pct": 100 * (summary_epec - summary_plan) / summary_plan,
    },
    index=REGIONS,
)

# Keep China first for the narrative, then order losses by absolute PV effect.
loss_order = (
    df_total.drop(index="ch")
    .assign(abs_delta=lambda d: d["delta"].abs())
    .sort_values("abs_delta", ascending=False)
    .index.tolist()
)
plot_order = ["ch"] + loss_order
df_total = df_total.loc[plot_order]

df_plan_idx = df_plan.set_index(["r", "t"])
df_epec_idx = df_epec.set_index(["r", "t"])

delta_annual = pd.DataFrame(index=plot_order, columns=PERIODS, dtype=float)
for r in plot_order:
    for t in PERIODS:
        delta_annual.loc[r, t] = (
            df_epec_idx.loc[(r, t), "W"] - df_plan_idx.loc[(r, t), "W"]
        ) / MUSD_TO_BUSD

y = np.arange(len(df_total))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_total, ax_time) = plt.subplots(
    1,
    2,
    figsize=(11.8, 5.4),
    gridspec_kw={"width_ratios": [0.92, 1.08], "wspace": 0.16},
)

# Cumulative present-value welfare change against the planner benchmark.
bar_colors = [COLOR_GAIN if v > 0 else COLOR_LOSS for v in df_total["delta"]]
ax_total.barh(y, df_total["delta"], color=bar_colors, height=0.58, alpha=0.94)
ax_total.axvline(0, color="#222222", linewidth=1.0)

for yi, (_, row) in enumerate(df_total.iterrows()):
    if 0 <= row["delta"] < 0.05:
        x_text = -0.025
        ha = "right"
        label_text = f"{fmt_signed(row['delta'])} ({row['delta_pct']:+.1f}%)"
    elif row["delta"] >= 0:
        x_text = row["delta"] + 0.08
        ha = "left"
        label_text = f"{fmt_signed(row['delta'])} ({row['delta_pct']:+.1f}%)"
    else:
        x_text = row["delta"] - 0.025
        ha = "right"
        label_text = f"{fmt_signed(row['delta'])} ({row['delta_pct']:+.1f}%)"
    ax_total.text(
        x_text,
        yi,
        label_text,
        va="center",
        ha=ha,
        fontsize=11.5,
        color=COLOR_GAIN if row["delta"] > 0 else COLOR_LOSS,
        fontweight="bold" if row["delta"] > 0 else "normal",
    )
ax_total.set_yticks(y)
ax_total.set_yticklabels(df_total["region"], fontsize=10.5)
ax_total.invert_yaxis()
ax_total.set_xlabel("Cumulative welfare difference (EPEC - planner) [trillion USD]", fontsize=10.5)
ax_total.grid(True, axis="x", linestyle=":", color=COLOR_GRID)
ax_total.set_axisbelow(True)
ax_total.spines[["top", "right", "left"]].set_visible(False)
ax_total.tick_params(axis="y", length=0)
limit = max(abs(df_total["delta"].min()), abs(df_total["delta"].max())) * 2.05
ax_total.set_xlim(-limit, limit)

# Period-by-period annual welfare change.
heat_values = delta_annual.to_numpy(dtype=float)
heat_limit = np.nanmax(np.abs(heat_values))
norm = TwoSlopeNorm(vmin=-heat_limit, vcenter=0.0, vmax=heat_limit)
im = ax_time.imshow(heat_values, cmap="RdBu", norm=norm, aspect="auto")

ax_time.set_xticks(np.arange(len(PERIODS)))
ax_time.set_xticklabels(PERIODS, fontsize=10)
ax_time.set_yticks(y)
ax_time.set_yticklabels([])
ax_time.tick_params(axis="y", length=0)
ax_time.spines[["top", "right", "left", "bottom"]].set_visible(False)

for i in range(heat_values.shape[0]):
    for j in range(heat_values.shape[1]):
        value = heat_values[i, j]
        text_color = "white" if abs(value) > 55 else "#222222"
        ax_time.text(
            j,
            i,
            fmt_signed(value, digits=1),
            ha="center",
            va="center",
            fontsize=11.5,
            color=text_color,
            fontweight="bold" if value > 0 else "normal",
        )

cax = inset_axes(
    ax_time,
    width="100%",
    height="7%",
    loc="lower center",
    bbox_to_anchor=(0, -0.20, 1, 1),
    bbox_transform=ax_time.transAxes,
    borderpad=0,
)
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("Annual welfare difference (EPEC - planner) [billion USD/year]", fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.22, wspace=0.16)

out_png = os.path.join(OUT_DIR, "welfare_epec_vs_planner.png")
out_pdf = os.path.join(OUT_DIR, "welfare_epec_vs_planner.pdf")
fig.savefig(out_png, dpi=180, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved to {out_png}")
print(f"Saved to {out_pdf}")


# ---------------------------------------------------------------------------
# Alternative figure: period-stacked annual welfare differences
# ---------------------------------------------------------------------------
fig_stack, ax_stack = plt.subplots(figsize=(7.4, 4.8))

gain_colors = {
    "2025": "#DCEFD9",
    "2030": "#A8D5A2",
    "2035": "#5EA267",
    "2040": "#2E6F40",
}
loss_colors = {
    "2025": "#FADBD8",
    "2030": "#F1948A",
    "2035": "#D95F5F",
    "2040": "#A83232",
}
positive_base = np.zeros(len(plot_order))
negative_base = np.zeros(len(plot_order))

for j, period in enumerate(PERIODS):
    values = delta_annual[period].to_numpy(dtype=float)
    positive_values = np.where(values > 0, values, 0.0)
    negative_values = np.where(values < 0, values, 0.0)

    ax_stack.barh(
        y,
        positive_values,
        left=positive_base,
        height=0.62,
        color=gain_colors[period],
        edgecolor="white",
        linewidth=1.2,
    )
    ax_stack.barh(
        y,
        np.abs(negative_values),
        left=negative_base + negative_values,
        height=0.62,
        color=loss_colors[period],
        edgecolor="white",
        linewidth=1.2,
    )

    positive_base += positive_values
    negative_base += negative_values

ax_stack.axvline(0, color="#1A1A1A", linewidth=1.35, zorder=3)
ax_stack.set_yticks(y)
ax_stack.set_yticklabels(df_total["region"], fontsize=15)
ax_stack.invert_yaxis()
ax_stack.set_xlabel(
    "[billion USD/year]",
    fontsize=15,
)
ax_stack.grid(True, axis="x", linestyle=":", color=COLOR_GRID)
ax_stack.set_axisbelow(True)
ax_stack.spines[["top", "right", "left"]].set_visible(False)
ax_stack.tick_params(axis="x", labelsize=15)
ax_stack.tick_params(axis="y", length=0, labelsize=15)
legend_handles = [
    SplitLegendKey(gain_colors[period], loss_colors[period])
    for period in PERIODS
]
ax_stack.legend(
    handles=legend_handles,
    labels=PERIODS,
    handler_map={SplitLegendKey: SplitLegendHandler()},
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.34),
    frameon=True,
    fontsize=13,
    framealpha=0.9,
    handlelength=2.0,
    handletextpad=0.5,
    columnspacing=1.0,
    borderpad=0.45,
    labelspacing=0.35,
)

stack_limit = max(abs(negative_base.min()), abs(positive_base.max())) * 1.12
ax_stack.set_xlim(-stack_limit, stack_limit)
fig_stack.subplots_adjust(left=0.23, right=0.96, top=0.96, bottom=0.36)

out_stack_png = os.path.join(OUT_DIR, "welfare_epec_vs_planner_stacked_annual.png")
out_stack_pdf = os.path.join(OUT_DIR, "welfare_epec_vs_planner_stacked_annual.pdf")
fig_stack.savefig(out_stack_png, dpi=180, bbox_inches="tight")
fig_stack.savefig(out_stack_pdf, bbox_inches="tight")
print(f"Saved to {out_stack_png}")
print(f"Saved to {out_stack_pdf}")
