"""Sensitivity analysis runner for the intertemporal EPEC model.

Varies two orthogonal dimensions:
  1. Player order — the sequence in which players are solved in each GS sweep.
  2. dK_net initial conditions — the warm-start values for net capacity changes.

Usage
-----
    from model.sensitivity import SensitivitySpec, run_sensitivity
    from model.run_gs import RunConfig

    spec = SensitivitySpec(
        base_cfg=RunConfig(solver="ipopt"),
        all_permutations=True,
        dk_init_modes=["zero", "max_growth", "half_growth"],
    )
    summary_path = run_sensitivity(spec)

dK_net modes
------------
  "zero"          All net capacity changes initialised to 0 (no investment assumption).
  "max_growth"    dK_net = g_exp_ub * Kcap_init for every region and period.
  "half_growth"   dK_net = 0.5 * g_exp_ub * Kcap_init.
  "max_decline"   dK_net = -g_dec_ub * Kcap_init (max decommissioning).
  "random_<seed>" dK_net sampled uniformly from [0, g_exp_ub * Kcap_init] with the
                  given integer seed, independently per region and period.

Q_offer is always initialised to the implied Kcap path (as enforced in the model),
so it is not a free dimension of the sensitivity analysis.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from itertools import permutations
from typing import Dict, List, Tuple

import pandas as pd

try:
    from .data_prep import load_data_from_excel
    from . import model_main as _it
    from .run_gs import RunConfig, run, PROJECT_ROOT
except ImportError:
    from data_prep import load_data_from_excel
    import model_main as _it
    from run_gs import RunConfig, run, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass
class SensitivitySpec:
    """Configuration for a sensitivity analysis sweep."""

    base_cfg: RunConfig = field(default_factory=RunConfig)

    # --- Player order axis ---
    # Explicit list of orderings to try (each is a list of player strings).
    # None means only the default order is used (unless all_permutations or
    # n_random_orders are set).
    player_orders: List[List[str]] | None = None
    # Try every permutation of the player list (feasible for <= 5 players).
    all_permutations: bool = False
    # Additionally sample this many random permutations (on top of the above).
    n_random_orders: int = 0
    # Seed for random permutation sampling.
    random_order_seed: int = 42

    # --- dK_net initial condition axis ---
    # See module docstring for valid mode strings.
    dk_init_modes: List[str] = field(
        default_factory=lambda: ["zero", "max_growth", "half_growth"]
    )

    # --- Output ---
    out_dir: str = os.path.join(PROJECT_ROOT, "outputs", "sensitivity")
    run_label: str = "sensitivity"


# ---------------------------------------------------------------------------
# Player-order generation
# ---------------------------------------------------------------------------

def _generate_player_orders(
    players: List[str],
    *,
    all_perms: bool,
    n_random: int,
    explicit: List[List[str]] | None,
    seed: int,
) -> List[Tuple[str, List[str]]]:
    """Return a list of (label, order) tuples covering the requested variations.

    The default order is always included first.
    """
    default = list(players)
    seen: set[tuple[str, ...]] = set()
    orders: List[Tuple[str, List[str]]] = []

    def _add(label: str, order: List[str]) -> None:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            orders.append((label, list(order)))

    _add("default", default)

    if all_perms:
        for perm in permutations(players):
            _add("perm_" + "_".join(perm), list(perm))

    if n_random > 0:
        rng = random.Random(seed)
        attempts = 0
        added = 0
        while added < n_random and attempts < n_random * 50:
            perm = list(players)
            rng.shuffle(perm)
            label = f"rand{seed}_{attempts}"
            prev_len = len(orders)
            _add(label, perm)
            if len(orders) > prev_len:
                added += 1
            attempts += 1

    if explicit:
        for i, order in enumerate(explicit):
            _add(f"explicit_{i}", list(order))

    return orders


# ---------------------------------------------------------------------------
# dK_net initial state construction
# ---------------------------------------------------------------------------

def _build_dk_init_state(data: "_it.ModelData", mode: str) -> Dict[str, Dict]:
    """Build a full initial-state dict with dK_net set according to *mode*.

    Q_offer is set to the implied Kcap path (matching model behaviour where
    Q_offer is fixed to Kcap).  p_offer is initialised to the midpoint of the
    allowed bilateral price range.  a_bid is fixed to the true demand intercept.
    """
    times: List[str] = data.times or ["2025", "2030", "2035", "2040", "2045"]
    move_times = _it._move_times(times)
    kcap_init = dict(_it._initial_capacity_by_region(data))
    g_exp_map = data.g_exp_ub or {}
    g_dec_map = data.g_dec_ub or {}
    g_exp_is_abs = bool(getattr(data, "g_exp_ub_is_absolute", False))

    dK_net: Dict[Tuple[str, str], float] = {}

    if mode == "zero":
        for r in data.players:
            for tp in move_times:
                dK_net[(r, tp)] = 0.0

    elif mode == "max_growth":
        for r in data.players:
            k0 = float(kcap_init.get(r, 0.0))
            g = float(g_exp_map.get(r, 0.0))
            rate = g if g_exp_is_abs else g * k0
            for tp in move_times:
                dK_net[(r, tp)] = rate

    elif mode == "half_growth":
        for r in data.players:
            k0 = float(kcap_init.get(r, 0.0))
            g = float(g_exp_map.get(r, 0.0))
            rate = g if g_exp_is_abs else g * k0
            for tp in move_times:
                dK_net[(r, tp)] = 0.5 * rate

    elif mode == "max_decline":
        for r in data.players:
            k0 = float(kcap_init.get(r, 0.0))
            g = float(g_dec_map.get(r, 0.0))
            for tp in move_times:
                dK_net[(r, tp)] = -g * k0

    elif mode.startswith("random_"):
        try:
            seed = int(mode.split("_", 1)[1])
        except ValueError:
            raise ValueError(
                f"Invalid random mode '{mode}'. Expected format: 'random_<int_seed>'."
            )
        rng = random.Random(seed)
        for r in data.players:
            k0 = float(kcap_init.get(r, 0.0))
            g = float(g_exp_map.get(r, 0.0))
            max_rate = g if g_exp_is_abs else g * k0
            for tp in move_times:
                dK_net[(r, tp)] = rng.uniform(0.0, max_rate)

    else:
        raise ValueError(
            f"Unknown dk_init_mode '{mode}'. "
            "Valid options: 'zero', 'max_growth', 'half_growth', 'max_decline', "
            "'random_<seed>'."
        )

    # Q_offer fixed to implied Kcap path
    implied_kcap = _it._implied_capacity_path(data, times, dK_net)
    q_offer: Dict[Tuple[str, str], float] = {
        (r, tp): max(float(implied_kcap.get((r, tp), 0.0)), 0.0)
        for r in data.players
        for tp in times
    }

    # p_offer at midpoint of bilateral price upper-bound
    p_offer: Dict[Tuple[str, str, str], float] = {
        (ex, im, tp): 0.5 * float(data.p_offer_ub[(ex, im)])
        for ex in data.regions
        for im in data.regions
        for tp in times
    }

    # a_bid fixed to true demand (consistent with fix_a_bid_to_true_dem=True default)
    a_bid: Dict[Tuple[str, str], float] = {
        (r, tp): _it._true_demand_intercept(data, r, tp)
        for r in data.regions
        for tp in times
    }

    return {"Q_offer": q_offer, "dK_net": dK_net, "p_offer": p_offer, "a_bid": a_bid}


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------

def _read_run_summary(output_path: str, data: "_it.ModelData") -> Dict[str, object]:
    """Extract key convergence metrics and equilibrium values from a run's Excel output."""
    summary: Dict[str, object] = {}
    try:
        df_iters = pd.read_excel(output_path, sheet_name="iters")
        if not df_iters.empty:
            last = df_iters.iloc[-1]
            summary["n_iters"] = int(last.get("iter", -1))
            summary["r_strat_final"] = float(last.get("r_strat", float("nan")))
            summary["r_obj_final"] = float(last.get("r_obj", float("nan")))
            summary["stable_count_final"] = int(last.get("stable_count", 0))
    except Exception:
        summary["n_iters"] = -1
        summary["r_strat_final"] = float("nan")
        summary["r_obj_final"] = float("nan")
        summary["stable_count_final"] = 0

    try:
        df_reg = pd.read_excel(output_path, sheet_name="regions")
        times = data.times or ["2025", "2030", "2035", "2040", "2045"]
        for r in data.regions:
            for t in times:
                row = df_reg[(df_reg["r"] == r) & (df_reg["t"].astype(str) == str(t))]
                if not row.empty:
                    summary[f"Kcap_{r}_{t}"] = float(row["Kcap"].iloc[0])
                    summary[f"lam_{r}_{t}"] = float(row["lam"].iloc[0])
                    summary[f"dK_net_{r}_{t}"] = float(row["net_cap_change"].iloc[0])
    except Exception:
        pass  # equilibrium columns optional in summary

    return summary


# ---------------------------------------------------------------------------
# Main sensitivity runner
# ---------------------------------------------------------------------------

def run_sensitivity(spec: SensitivitySpec) -> str:
    """Run all (player_order, dk_init_mode) combinations defined by *spec*.

    Returns the path to the summary CSV file.
    """
    os.makedirs(spec.out_dir, exist_ok=True)

    # Load data once to get player names and parameters for initial-state construction
    excel_path = spec.base_cfg.excel_path
    data = load_data_from_excel(excel_path)

    # Generate player orderings
    player_orders = _generate_player_orders(
        list(data.players),
        all_perms=spec.all_permutations,
        n_random=spec.n_random_orders,
        explicit=spec.player_orders,
        seed=spec.random_order_seed,
    )

    combos = [
        (order_info, dk_mode)
        for order_info in player_orders
        for dk_mode in spec.dk_init_modes
    ]

    print(
        f"[SENSITIVITY] {len(combos)} runs: "
        f"{len(player_orders)} player orderings × {len(spec.dk_init_modes)} dk_init modes"
    )
    print(f"[SENSITIVITY] Player orderings: {[lbl for lbl, _ in player_orders]}")
    print(f"[SENSITIVITY] dk_init modes:    {spec.dk_init_modes}")
    print(f"[SENSITIVITY] Output dir:       {spec.out_dir}")

    summary_rows: List[Dict[str, object]] = []
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for run_idx, ((order_label, order), dk_mode) in enumerate(combos, 1):
        print(
            f"\n[SENSITIVITY] Run {run_idx}/{len(combos)} — "
            f"order={order_label} ({' > '.join(order)})  dk_init={dk_mode}"
        )
        t0 = time.perf_counter()

        # Build initial state for this dk_mode
        init_state = _build_dk_init_state(data, dk_mode)

        # Sub-directory per run so outputs don't collide
        run_out_dir = os.path.join(spec.out_dir, f"run_{run_idx:03d}_{order_label}_{dk_mode}")
        os.makedirs(run_out_dir, exist_ok=True)

        cfg = dataclasses.replace(
            spec.base_cfg,
            player_order=order,
            force_ch_last=False,   # respect the explicit ordering we're testing
            initial_state_override=init_state,
            out_dir=run_out_dir,
            plots_dir=os.path.join(run_out_dir, "plots"),
        )

        row: Dict[str, object] = {
            "run_idx": run_idx,
            "order_label": order_label,
            "player_order": " > ".join(order),
            "dk_init_mode": dk_mode,
        }

        try:
            output_path = run(cfg)
            elapsed = time.perf_counter() - t0
            row["status"] = "ok"
            row["elapsed_s"] = round(elapsed, 1)
            row["output_path"] = output_path
            row.update(_read_run_summary(output_path, data))
            print(
                f"[SENSITIVITY] Run {run_idx} done in {elapsed:.1f}s — "
                f"r_strat={row.get('r_strat_final', '?'):.4g}  "
                f"iters={row.get('n_iters', '?')}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row["status"] = f"failed: {exc}"
            row["elapsed_s"] = round(elapsed, 1)
            row["output_path"] = ""
            print(f"[SENSITIVITY] Run {run_idx} FAILED after {elapsed:.1f}s: {exc}")
            traceback.print_exc()

        summary_rows.append(row)

    # Write summary CSV and Excel
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(spec.out_dir, f"{spec.run_label}_{run_ts}_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    xlsx_path = os.path.join(spec.out_dir, f"{spec.run_label}_{run_ts}_summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]
        wb = writer.book
        hdr_fmt = wb.add_format({"bold": True, "border": 1})
        for col_idx, col_name in enumerate(df_summary.columns):
            ws.write(0, col_idx, col_name, hdr_fmt)
            col_width = max(len(str(col_name)), df_summary[col_name].astype(str).map(len).max())
            ws.set_column(col_idx, col_idx, min(max(col_width + 2, 10), 40))
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df_summary), len(df_summary.columns) - 1)

    print(f"\n[SENSITIVITY] Done. Summary written to:")
    print(f"  CSV:  {csv_path}")
    print(f"  XLSX: {xlsx_path}")
    return xlsx_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a default sensitivity analysis with all permutations and three dk_init modes."""
    spec = SensitivitySpec(
        base_cfg=RunConfig(),
        all_permutations=True,
        dk_init_modes=["zero", "half_growth", "max_growth"],
        run_label="sensitivity",
    )
    run_sensitivity(spec)


if __name__ == "__main__":
    main()
