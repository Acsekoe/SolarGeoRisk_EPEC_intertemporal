from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

try:
    from .data_prep import load_data_from_excel, load_initial_state
    from .gauss_seidel import solve_gs_intertemporal
    from . import model_main as _it
    from .plot_results import write_default_plots
    from .results_writer import write_results_excel
except ImportError:
    from data_prep import load_data_from_excel, load_initial_state
    from gauss_seidel import solve_gs_intertemporal
    import model_main as _it
    from plot_results import write_default_plots
    from results_writer import write_results_excel


# Define project root relative to this script
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
_VALID_CONVERGENCE_MODES = {"strategy", "objective", "combined"}

# ── Player sweep order ────────────────────────────────────────────────────────
# Edit this list to control the Gauss-Seidel sweep order.
# Every strategic player must appear exactly once (case-insensitive).
PLAYER_ORDER: List[str] = ["ch", "us", "apac", "af", "row", "eu"]
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    excel_path: str = os.path.join(PROJECT_ROOT, "inputs", "input_data_intertemporal.xlsx")
    out_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    plots_dir: str = os.path.join(PROJECT_ROOT, "plots")

    solver: str = "ipopt"
    feastol: float = 1e-4
    opttol: float = 1e-4
    
    method: str = "gauss_seidel"
    iters: int = 15
    omega: float = 0.7
    tol_strat: float = 1e-2
    tol_obj: float = 1e-2
    stable_iters: int = 3
    eps_x: float = 1e-3
    eps_comp: float = 1e-3
    workdir: str | None = None
    convergence_mode: str = "combined"  # "strategy", "objective", or "combined"

    keep_workdir: bool = False

    knitro_outlev: int | None = None
    knitro_maxit: int | None = None
    knitro_hessopt: int | None = None
    knitro_algorithm: int | None = None

    # Algorithmic proximal penalties: -0.5 * c_pen * (X - X_last)^2 added to ULP objective.
    # Set to 0.0 to disable. Larger values shrink step sizes and improve GS stability.
    c_pen_q: float = 0.1   # For Q_offer
    c_pen_p: float = 0.1   # For p_offer
    c_pen_a: float = 0.1   # For a_bid

    # Economic quadratic penalties: -0.5 * c_quad * X^2
    # Represents convex costs or disutility.
    c_quad_q: float = 1  # For Q_offer (production cost)
    c_quad_p: float = 0.1  # For p_offer (offer deviation)
    c_quad_a: float = 0.1  # For a_bid (demand withholding cost)

    # Capacity-policy incentives (objective terms).
    # Positive values encourage capacity retention/expansion and penalize decommissioning.
    cap_keep_reward: float = 0
    capex_subsidy: float = 0
    terminal_capacity_value: float = 0
    decommission_penalty: float = 0

    # Force Q_offer == Kcap for all regions and periods (no quantity withholding).
    # Eliminates mu_offer gaming; useful as a diagnostic run.
    fix_q_offer_to_kcap: bool = False

    # Minimum fraction of Kcap that must be offered (0.0 = no floor, 1.0 = same as fix_q_offer_to_kcap).
    # E.g. 0.99 limits withholding to 1% of capacity while allowing mu_offer to go to zero (interior solution).
    q_offer_lb_frac: float = 0.99

    # NPV discounting: override beta_t computed in data_prep.py.
    # 0.0 = undiscounted (block-length weighted); any positive value recomputes beta_t.
    discount_rate: float = 0.02
    base_year: int = 2025




def _resolve_excel_path(raw: str | None, default_path: str) -> str:
    if raw is None or not str(raw).strip():
        return default_path
    candidate = str(raw)
    if os.path.exists(candidate):
        return candidate
    in_inputs = os.path.join("inputs", candidate)
    if os.path.exists(in_inputs):
        return in_inputs
    return candidate


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _time_label_to_year(label: object) -> int | None:
    try:
        return int(str(label).strip())
    except Exception:
        return None



def _print_state_summary(*, data: _it.ModelData, regions: list[str], state: dict[str, dict], tag: str = "SUMMARY") -> None:
    kcap = state.get("Kcap", {}) or {}
    dk_map = state.get("dK_net", {}) or {}
    q_offer = state.get("Q_offer", {}) or {}
    lam = state.get("lam", {}) or {}
    x_map = state.get("x", {}) or {}

    if not regions:
        print(f"[{tag}] No regions configured; skipping Q_offer/lam print.")
        return

    print(f"[{tag}] Q_offer, endogenous Kcap, net capacity change, utilization, lam, declared demand, and max offer prices by region and time:")
    
    # Collect all unique time periods from Q_offer keys
    times = sorted(list(set(k[1] for k in q_offer.keys() if isinstance(k, tuple) and len(k) > 1)))
    if not times:
        times = ["2025", "2030", "2035", "2040", "2045"]

    # Keep console output focused on the policy horizon (<=2040) when
    # additional terminal-buffer periods are present.
    print_times = [t for t in times if (_time_label_to_year(t) is not None and _time_label_to_year(t) <= 2040)]
    if not print_times:
        print_times = list(times)
    for r in regions:
        for t in print_times:
            k_val = _safe_float(kcap.get((r, t), (data.Kcap_2025 or data.Qcap).get(r, 0.0)))
            dk_val = _safe_float(dk_map.get((r, t), 0.0))
            q_val = _safe_float(q_offer.get((r, t), 0.0))
            l_val = _safe_float(lam.get((r, t), 0.0))
            utilized = sum(_safe_float(x_map.get((r, dest, t), 0.0)) for dest in data.regions)
            util_rate = utilized / k_val if k_val > 0.0 else 0.0
            
            x_dem_val = _safe_float(state.get("x_dem", {}).get((r, t), 0.0))
            a_true = float(data.a_dem_t.get((r, t), 0.0)) if data.a_dem_t else float(data.a_dem.get(r, 0.0))
            a_bid_val = _safe_float(state.get("a_bid", {}).get((r, t), a_true))
            
            # Exogenous LBD cost: read directly from the pre-computed schedule
            c_man_val = float(
                (data.c_man_t or {}).get((r, t), data.c_man.get(r, 0.0))
            )
            
            # Find max p_offer for this region and time (from r to any destination)
            p_offer_map = state.get("p_offer", {})
            max_p = max([_safe_float(v) for k, v in p_offer_map.items() 
                          if isinstance(k, tuple) and len(k) == 3 and k[0] == r and k[2] == t], default=0.0)
            
            print(f"  {r:<4} {t:<4} D={x_dem_val:<6.1f} a_bid={a_bid_val:<6.1f} | Q={q_val:<7.1f} K={k_val:<7.1f} dK={dk_val:<7.2f} util={util_rate:<5.2f} lam={l_val:<6.1f} poffer={max_p:<6.1f} | c={c_man_val:<6.1f}")


def _gams_workdir(run_id: str, configured_workdir: str | None) -> str:
    # The current GS solver builds one shared GAMS container up front.
    # It still benefits from an explicit space-free workdir, but this is
    # per solve, not per player or per iteration.
    if configured_workdir and configured_workdir.strip():
        workdir = os.path.abspath(configured_workdir.strip())
    else:
        base = tempfile.gettempdir()
        workdir = os.path.join(base, f"solargeorisk_gams_{run_id}")

    if " " in workdir:
        workdir = os.path.join("C:\\temp", f"solargeorisk_gams_{run_id}")

    os.makedirs(workdir, exist_ok=True)
    return workdir


def _solver_options(
    *,
    solver: str,
    feastol: float,
    opttol: float,
    cfg: RunConfig,
) -> Dict[str, float]:
    name = solver.strip().lower()

    if name == "conopt":
        return {"Tol_Feas_Max": float(feastol), "Tol_Optimality": float(opttol)}

    if name == "knitro":
        opts: Dict[str, float] = {
            "feastol": float(feastol),
            "opttol": float(opttol),
        }
        if cfg.knitro_outlev is not None:
            opts["outlev"] = float(cfg.knitro_outlev)
        if cfg.knitro_maxit is not None:
            opts["maxit"] = float(cfg.knitro_maxit)
        if cfg.knitro_hessopt is not None:
            opts["hessopt"] = float(cfg.knitro_hessopt)
        if cfg.knitro_algorithm is not None:
            opts["algorithm"] = float(cfg.knitro_algorithm)
        return opts

    return {}


def _apply_data_overrides(data, cfg: RunConfig) -> None:
    data.eps_x = float(cfg.eps_x)
    data.eps_comp = float(cfg.eps_comp)

    if data.settings is None:
        data.settings = {}

    # fix_a_bid_to_true_dem=True clamps declared demand to true demand (no strategic withholding).
    # This override always takes effect; it cannot be disabled via Excel settings.
    data.settings["fix_a_bid_to_true_dem"] = True

    # Proximal penalty scalars — passed through to build_model via data.settings.
    data.settings["c_pen_q"]   = float(cfg.c_pen_q)
    data.settings["c_pen_p"]   = float(cfg.c_pen_p)
    data.settings["c_pen_a"]   = float(cfg.c_pen_a)
    
    # Economic quadratic scalars
    data.settings["c_quad_q"]  = float(cfg.c_quad_q)
    data.settings["c_quad_p"]  = float(cfg.c_quad_p)
    data.settings["c_quad_a"]  = float(cfg.c_quad_a)

    # Capacity-policy incentives
    data.settings["cap_keep_reward"] = float(cfg.cap_keep_reward)
    data.settings["capex_subsidy"] = float(cfg.capex_subsidy)
    data.settings["terminal_capacity_value"] = float(cfg.terminal_capacity_value)
    data.settings["decommission_penalty"] = float(cfg.decommission_penalty)

    # Force Q_offer == Kcap (no quantity withholding)
    data.settings["fix_q_offer_to_kcap"] = bool(cfg.fix_q_offer_to_kcap)
    data.settings["q_offer_lb_frac"] = float(cfg.q_offer_lb_frac)

    # Discount rate for NPV computation
    data.settings["discount_rate"] = float(cfg.discount_rate)

    # If RunConfig specifies a non-zero discount_rate, recompute beta_t to override
    # whatever was loaded from Excel (or the default of 1.0).
    # NOTE: This silently overwrites any discount_rate set in the Excel settings sheet.
    if cfg.discount_rate != 0.0 or cfg.base_year != 2025:
        excel_beta = data.beta_t or {}
        print(
            f"[CONFIG] RunConfig.discount_rate={cfg.discount_rate} base_year={cfg.base_year} "
            f"overrides Excel beta_t (was: {excel_beta})"
        )
        times = data.times or ["2025", "2030", "2035", "2040", "2045"]
        r = float(cfg.discount_rate)
        by = int(cfg.base_year)
        year_labels = {tp: _time_label_to_year(tp) for tp in times}
        bad_labels = [tp for tp, year in year_labels.items() if year is None]
        if bad_labels:
            raise ValueError(
                "discount_rate/base_year override requires integer-like time labels. "
                f"Invalid labels: {bad_labels}"
            )
        data.beta_t = {
            tp: (1.0 if r == 0.0 else 1.0 / ((1.0 + r) ** (year_labels[tp] - by)))
            for tp in times
        }


def _build_initial_state(data, cfg: RunConfig, excel_path: str) -> dict[str, dict]:
    times = data.times or ["2025", "2030", "2035", "2040", "2045"]
    move_times = _it._move_times(times)

    # Try reading warm-start from the initial_state sheet in the input Excel
    excel_ws = load_initial_state(excel_path, data)
    if excel_ws is not None:
        print(f"[CONFIG] Loaded initial state from Excel sheet 'initial_state'")
        for r in data.players:
            q25 = excel_ws["Q_offer"].get((r, times[0]), 0.0)
            q_end = excel_ws["Q_offer"].get((r, times[-1]), 0.0)
            dk25 = excel_ws["dK_net"].get((r, times[0]), 0.0)
            print(f"  Q_offer[{r}] ({times[0]}->{times[-1]}) = {q25:.1f} -> {q_end:.1f}  dK_net({times[0]})={dk25:.2f}")
        return excel_ws

    print("[CONFIG] Building deterministic EPEC warm-start (no initial_state sheet)...")

    kcap_current = dict(_it._initial_capacity_by_region(data))
    ytn_dict = data.years_to_next or {t: 5.0 for t in times}
    g_exp_dict = data.g_exp_ub or {r: 0.0 for r in data.regions}
    g_exp_is_abs = bool(getattr(data, "g_exp_ub_is_absolute", False))

    q_offer: dict[tuple[str, str], float] = {}
    dK_net: dict[tuple[str, str], float] = {}
    for tp in times:
        for r in data.players:
            q_offer[(r, tp)] = max(float(kcap_current.get(r, 0.0)), 0.0)
        if tp not in move_times:
            continue
        years = float(ytn_dict.get(tp, 5.0))
        for r in data.regions:
            g = float(g_exp_dict.get(r, 0.0))
            k_now = max(float(kcap_current.get(r, 0.0)), 0.0)
            rate = g if g_exp_is_abs else g * k_now
            dK_net[(r, tp)] = rate
            kcap_current[r] = k_now + years * rate

    p_offer = {
        (ex, im, tp): 0.5 * float(data.p_offer_ub[(ex, im)])
        for ex in data.regions
        for im in data.regions
        for tp in times
    }
    a_bid = {
        (r, tp): _it._true_demand_intercept(data, r, tp)
        for r in data.regions
        for tp in times
    }
    ws = {
        "Q_offer": q_offer,
        "dK_net": dK_net,
        "p_offer": p_offer,
        "a_bid": a_bid,
    }

    # Print summary
    for r in data.players:
        q25 = ws["Q_offer"].get((r, times[0]), 0.0)
        q_end = ws["Q_offer"].get((r, times[-1]), 0.0)
        dk25 = ws["dK_net"].get((r, times[0]), 0.0)
        print(f"  Q_offer[{r}] ({times[0]}->{times[-1]}) = {q25:.1f} -> {q_end:.1f}  dK_net({times[0]})={dk25:.2f}")
    print(f"[CONFIG] Deterministic warm-start ready (time-indexed across {len(times)} periods)")
    return ws


def _append_detailed_iter_rows(
    *,
    data,
    state: dict[str, dict],
    it: int,
    r_strat: float,
    stable_count: int,
    rows: list[dict[str, object]],
) -> None:
    q_map = state.get("Q_offer", {})
    kcap_map = state.get("Kcap", {})
    dk_map = state.get("dK_net", {})
    lam_map = state.get("lam", {})
    mu_map = state.get("mu_offer", {})
    beta_dem_map = state.get("beta_dem", {})
    psi_dem_map = state.get("psi_dem", {})
    obj_map = state.get("obj", {})
    x_map = state.get("x", {})
    gamma_map = state.get("gamma", {})
    p_offer_map = state.get("p_offer", {})

    times = sorted(list(set(k[1] for k in q_map.keys() if isinstance(k, tuple) and len(k) > 1)))
    if not times:
        times = ["2025", "2030", "2035", "2040", "2045"]
    
    a_bid_map = state.get("a_bid", {})

    for r in data.regions:
        for t in times:
            row: dict[str, object] = {
                "iter": it,
                "r": r,
                "t": t,
                "stable_count": stable_count,
                "r_strat": r_strat,
                "Kcap": _safe_float(kcap_map.get((r, t), (data.Kcap_2025 or data.Qcap).get(r, 0.0))),
                "net_cap_change": _safe_float(dk_map.get((r, t), 0.0)),
                "Q_offer": _safe_float(q_map.get((r, t))),
                "lam": _safe_float(lam_map.get((r, t))),
                "mu_offer": _safe_float(mu_map.get((r, t))),
                "beta_dem": _safe_float(beta_dem_map.get((r, t))),
                "psi_dem": _safe_float(psi_dem_map.get((r, t))),
                "a_bid": _safe_float(a_bid_map.get((r, t), data.a_dem_t.get((r, t)) if data.a_dem_t else data.a_dem.get(r))),
                # Exogenous LBD cost schedule (pre-computed via Swanson's Law)
                "c_man_var": float((data.c_man_t or {}).get((r, t), data.c_man.get(r, 0.0))),
                "obj": _safe_float(obj_map.get(r)) if r in data.players else 0.0,
            }
            
            # Post-solve computed diagnostics
            x_dem_val = sum(_safe_float(x_map.get((src, r, t))) for src in data.regions)
            utilized_capacity = sum(_safe_float(x_map.get((r, dest, t))) for dest in data.regions)
            a_true = float(data.a_dem_t.get((r, t), 0.0)) if data.a_dem_t else float(data.a_dem.get(r, 0.0))
            row["wedge"] = a_true - float(row["a_bid"] or a_true)
            row["utilized_capacity"] = utilized_capacity
            row["utilization_rate"] = utilized_capacity / float(row["Kcap"]) if float(row["Kcap"]) > 0.0 else 0.0
            row["Icap_report"] = max(float(row["net_cap_change"]), 0.0)
            row["Dcap_report"] = max(-float(row["net_cap_change"]), 0.0)
    
            for dest in data.regions:
                row[f"x_exp_to_{dest}"] = _safe_float(x_map.get((r, dest, t)))
                row[f"gamma_exp_to_{dest}"] = _safe_float(gamma_map.get((r, dest, t)))
                row[f"p_offer_to_{dest}"] = _safe_float(p_offer_map.get((r, dest, t)))
    
            for src in data.regions:
                row[f"x_imp_from_{src}"] = _safe_float(x_map.get((src, r, t)))
                row[f"gamma_imp_from_{src}"] = _safe_float(gamma_map.get((src, r, t)))
                row[f"p_offer_from_{src}"] = _safe_float(p_offer_map.get((src, r, t)))
    
            rows.append(row)


def run(cfg: RunConfig) -> str:
    method = cfg.method.lower().strip()
    if method != "gauss_seidel":
        raise ValueError(f"Unsupported method '{cfg.method}'. Supported: 'gauss_seidel'.")

    excel_path = _resolve_excel_path(cfg.excel_path, cfg.excel_path)
    out_dir = cfg.out_dir
    plots_dir = cfg.plots_dir

    solver = cfg.solver.strip()
    feastol = float(cfg.feastol)
    opttol = float(cfg.opttol)
    iters = int(cfg.iters)
    omega = float(cfg.omega)
    tol_rel = float(cfg.tol_strat)
    tol_obj = float(cfg.tol_obj)
    stable_iters = int(cfg.stable_iters)
    convergence_mode = cfg.convergence_mode.strip().lower()
    if convergence_mode not in _VALID_CONVERGENCE_MODES:
        raise ValueError(
            f"Unsupported convergence_mode '{cfg.convergence_mode}'. "
            f"Supported: {sorted(_VALID_CONVERGENCE_MODES)}."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"results_{run_id}.xlsx")
    workdir = _gams_workdir(run_id, cfg.workdir)

    data = load_data_from_excel(excel_path)
    _apply_data_overrides(data, cfg)

    print(f"[CONFIG] Model type: Offer Model EPEC")
    print(f"[CONFIG] Method: {method}")
    print(f"[CONFIG] Solver: {solver}  feastol={feastol:g}  opttol={opttol:g}")
    print(f"[CONFIG] iters={iters} omega={omega:g} tol_rel={tol_rel:g} stable_iters={stable_iters}")
    print(f"[CONFIG] eps_x={float(data.eps_x):g} eps_comp={float(data.eps_comp):g}")
    print(f"[CONFIG] convergence_mode={convergence_mode}")
    print(f"[CONFIG] player_order={PLAYER_ORDER}")
    print(f"[CONFIG] workdir={workdir}{' (keep)' if cfg.keep_workdir else ' (auto-cleanup)'}")

    sweep_times: list[float] = []
    timing_state = {"sweep_start": 0.0}
    detailed_iter_rows: list[dict[str, object]] = []

    def _iter_log(it: int, state: dict[str, dict], r_strat: float, stable_count: int) -> None:
        sweep_elapsed = time.perf_counter() - timing_state["sweep_start"]
        sweep_times.append(sweep_elapsed)
        print(f"[ITER {it}] r_strat={r_strat:.6g} stable_count={stable_count} sweep_time={sweep_elapsed:.2f}s")
        # Show shuffled player order if available
        if "_sweep_order" in state:
            print(f"[ITER {it}] player order: {state['_sweep_order']}")
        _print_state_summary(data=data, regions=list(data.regions), state=state, tag=f"ITER {it}")
        _append_detailed_iter_rows(
            data=data,
            state=state,
            it=it,
            r_strat=r_strat,
            stable_count=stable_count,
            rows=detailed_iter_rows,
        )
        timing_state["sweep_start"] = time.perf_counter()

    solver_opts = _solver_options(
        solver=solver,
        feastol=feastol,
        opttol=opttol,
        cfg=cfg,
    )
    print(f"[CONFIG] solver_options={solver_opts}")

    total_start = time.perf_counter()
    timing_state["sweep_start"] = total_start
    state: dict[str, dict] | None = None
    iter_rows: list[dict[str, object]] = []
    try:
        init_state = _build_initial_state(data, cfg, excel_path)
        print(f"[MAIN] Starting {iters} sweeps with {solver}")
        state, iter_rows = solve_gs_intertemporal(
            data,
            solver=solver,
            solver_options=solver_opts,
            iters=iters,
            omega=omega,
            tol_rel=cfg.tol_strat,
            tol_obj=cfg.tol_obj,
            stable_iters=stable_iters,
            working_directory=workdir,
            iter_callback=_iter_log,
            initial_state=init_state,
            convergence_mode=convergence_mode,
            player_order=PLAYER_ORDER,
        )
        _print_state_summary(data=data, regions=list(data.regions), state=state, tag="FINAL")

        write_results_excel(
            data=data,
            state=state,
            iter_rows=iter_rows,
            detailed_iter_rows=detailed_iter_rows,
            output_path=output_path,
            meta={
                # --- Identity ---
                "excel_path":            excel_path,
                "run_id":                run_id,
                # --- Algorithm ---
                "method":                method,
                "iters":                 iters,
                "omega":                 omega,
                "tol_rel":               tol_rel,
                "tol_obj":               float(cfg.tol_obj),
                "stable_iters":          stable_iters,
                "convergence_mode":      convergence_mode,
                # --- Solver ---
                "solver":                solver,
                "feastol":               feastol,
                "opttol":                opttol,
                "solver_options":        str(solver_opts),
                # --- Tolerances ---
                "eps_x":                 float(data.eps_x),
                "eps_comp":              float(data.eps_comp),
                # --- Discounting (effective values used in solve) ---
                "discount_rate":         float(cfg.discount_rate),
                "base_year":             int(cfg.base_year),
                "beta_t":                str(data.beta_t or {}),
                "ytn":                   str(data.years_to_next or {}),
                # --- Strategic demand bidding ---
                "fix_a_bid_to_true_dem": bool(data.settings.get("fix_a_bid_to_true_dem", False)),
                # --- Penalties and Scalers ---
                "c_pen_q":               float(cfg.c_pen_q),
                "c_pen_p":               float(cfg.c_pen_p),
                "c_pen_a":               float(cfg.c_pen_a),
                "c_quad_q":              float(cfg.c_quad_q),
                "c_quad_p":              float(cfg.c_quad_p),
                "c_quad_a":              float(cfg.c_quad_a),
                "cap_keep_reward":       float(cfg.cap_keep_reward),
                "capex_subsidy":         float(cfg.capex_subsidy),
                "terminal_capacity_value": float(cfg.terminal_capacity_value),
                "decommission_penalty":  float(cfg.decommission_penalty),
                # --- Paths ---
                "workdir":               workdir,
            },
        )

        try:
            write_default_plots(output_path=output_path, plots_dir=plots_dir)
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}")
        print(f"[OK] wrote: {output_path}")
        return output_path
    finally:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[TIMING] Total solve time: {total_elapsed:.2f}s")
        if sweep_times:
            print(f"[TIMING] Mean sweep time: {sum(sweep_times)/len(sweep_times):.2f}s  (n={len(sweep_times)})")
        if not cfg.keep_workdir:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
                print(f"[CLEANUP] Deleted workdir: {workdir}")
            except Exception as e:
                print(f"[WARN] Could not delete workdir {workdir}: {e}")
        else:
            print(f"[KEEP] Workdir retained: {workdir}")


def main() -> None:
    run(RunConfig())


if __name__ == "__main__":
    main()
