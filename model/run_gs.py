from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set PYTHONPATH so parallel workers (spawned subprocesses) can find the package
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if "PYTHONPATH" in os.environ:
    if src_path not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = src_path + os.pathsep + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = src_path

from data_prep import load_data_from_excel
from gauss_seidel import solve_gs_intertemporal
import model_llp_planner as llp
import model_main as _it
from plot_results import write_default_plots
from results_writer import write_results_excel


# Define project root relative to this script
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

@dataclass(frozen=True)
class RunConfig:
    excel_path: str = os.path.join(PROJECT_ROOT, "inputs", "input_data_intertemporal.xlsx")
    out_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    plots_dir: str = os.path.join(PROJECT_ROOT, "plots")

    solver: str = "ipopt"
    feastol: float = 1e-4
    opttol: float = 1e-4
    
    method: str = "gauss_seidel"
    iters: int = 10
    omega: float = 0.7
    tol_strat: float = 1e-2
    tol_obj: float = 1e-2
    stable_iters: int = 3
    eps_x: float = 1e-3
    eps_comp: float = 1
    workdir: str | None = None
    convergence_mode: str = "combined"  # "strategy", "objective", or "combined"
    workers: int = 1  # 1=sequential, >1=parallel
    worker_timeout: float = 120.0
    player_order: List[str] | None = field(default_factory=lambda: ["roa", "af", "row", "us", "eu", "apac", "ch"])
    shuffle_players: bool = False

    keep_workdir: bool = False
    debug_workers: bool = False

    knitro_outlev: int | None = None
    knitro_maxit: int | None = None
    knitro_hessopt: int | None = None
    knitro_algorithm: int | None = None

    # Scalers
    kappa_q: float | None = 0.1
    rho_prox: float | None = 0.00
    use_quad: bool = True

    # Proximal penalty scalars: -0.5 * c_pen * (X - X_last)^2 added to each player's ULP.
    # Set to 0.0 to disable. Larger values shrink step sizes and improve GS stability.
    c_pen_q:   float = 0.1   # Q_offer
    c_pen_cap: float = 0.0   # Icap_pos and Dcap_neg
    c_pen_p:   float = 0.1   # p_offer (additive on top of rho_prox)
    c_pen_a:   float = 0.1   # a_bid



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



def _print_state_summary(*, data: ModelData, regions: list[str], state: dict[str, dict], tag: str = "SUMMARY") -> None:
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
        times = ["2025", "2030", "2035", "2040"]
    
    for r in regions:
        for t in times:
            k_val = _safe_float(kcap.get((r, t), (data.Kcap_2025 or data.Qcap).get(r, 0.0)))
            dk_val = _safe_float(dk_map.get((r, t), 0.0))
            q_val = _safe_float(q_offer.get((r, t), 0.0))
            l_val = _safe_float(lam.get((r, t), 0.0))
            utilized = sum(_safe_float(x_map.get((r, dest, t), 0.0)) for dest in data.regions)
            util_rate = utilized / k_val if k_val > 0.0 else 0.0
            
            x_dem_val = _safe_float(state.get("x_dem", {}).get((r, t), 0.0))
            a_true = float(data.a_dem_t.get((r, t), 0.0)) if data.a_dem_t else float(data.a_dem.get(r, 0.0))
            a_bid_val = _safe_float(state.get("a_bid", {}).get((r, t), a_true))
            
            w_cum_val = _safe_float(state.get("W_cum", {}).get((r, t), 0.0))
            
            # Deterministically calculate c_man_val instead of taking the solver's unconstrained floating bounds for non-focal players
            base_c = data.c_man.get(r, 0.0)
            c_floor = max(50.0, base_c * 0.5)
            c_man_val = max(c_floor, base_c - 0.022 * w_cum_val)
            
            # Find max p_offer for this region and time (from r to any destination)
            p_offer_map = state.get("p_offer", {})
            max_p = max([_safe_float(v) for k, v in p_offer_map.items() 
                          if isinstance(k, tuple) and len(k) == 3 and k[0] == r and k[2] == t], default=0.0)
            
            print(f"  {r:<4} {t:<4} D={x_dem_val:<6.1f} a_bid={a_bid_val:<6.1f} | Q={q_val:<7.1f} K={k_val:<7.1f} dK={dk_val:<7.2f} util={util_rate:<5.2f} lam={l_val:<6.1f} poffer={max_p:<6.1f} | c={c_man_val:<6.1f}")


def _gams_workdir(run_id: str, configured_workdir: str | None) -> str:
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

    if cfg.kappa_q is not None and data.kappa_Q is not None:
        for r in data.regions:
            data.kappa_Q[r] = float(cfg.kappa_q)

    if data.settings is None:
        data.settings = {}
    if cfg.rho_prox is not None:
        data.settings["rho_prox"] = float(cfg.rho_prox)
    data.settings["use_quad"] = bool(cfg.use_quad)
    # fix_a_bid_to_true_dem=True clamps declared demand to true demand (no strategic withholding).
    # This override always takes effect; it cannot be disabled via Excel settings.
    data.settings["fix_a_bid_to_true_dem"] = True

    # Proximal penalty scalars — passed through to build_model via data.settings.
    data.settings["c_pen_q"]   = float(cfg.c_pen_q)
    data.settings["c_pen_cap"] = float(cfg.c_pen_cap)
    data.settings["c_pen_p"]   = float(cfg.c_pen_p)
    data.settings["c_pen_a"]   = float(cfg.c_pen_a)

    # If RunConfig specifies a non-zero discount_rate, recompute beta_t to override
    # whatever was loaded from Excel (or the default of 1.0).
    if cfg.discount_rate != 0.0 or cfg.base_year != 2025:
        times = data.times or ["2025", "2030", "2035", "2040"]
        r = float(cfg.discount_rate)
        by = int(cfg.base_year)
        data.beta_t = {
            tp: (1.0 if r == 0.0 else 1.0 / ((1.0 + r) ** (int(tp) - by)))
            for tp in times
        }


def _build_initial_state(data, cfg: RunConfig) -> dict[str, dict]:
    times = data.times or ["2025", "2030", "2035", "2040"]
    dK_zero = {(r, t): 0.0 for r in data.regions for t in _it._move_times(times)}

    print("[CONFIG] Solving LLP planner benchmark to compute dynamic warm-start...")
    llp_ctx = llp.build_llp_planner_model(data)
    llp.solve_llp_planner(llp_ctx, solver="conopt")
    llp_state = llp.extract_llp_state(llp_ctx, data)
    ws = llp.build_epec_warmstart(llp_state, data)
    
    # Print summary
    for r in data.players:
        q25 = ws["Q_offer"].get((r, times[0]), 0.0)
        q_end = ws["Q_offer"].get((r, times[-1]), 0.0)
        print(f"  Q_offer[{r}] ({times[0]}->{times[-1]}) = {q25:.1f} -> {q_end:.1f}")
    ws["dK_net"] = dict(dK_zero)
    print(f"[CONFIG] LLP warm-start ready (time-indexed across {len(times)} periods)")
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
        times = ["2025", "2030", "2035", "2040"]
    
    w_cum_map = state.get("W_cum", {})
    a_bid_map = state.get("a_bid", {})
    c_man_var_map = state.get("c_man_var", {})
    
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
                "W_cum": _safe_float(w_cum_map.get((r, t))),
                "a_bid": _safe_float(a_bid_map.get((r, t), data.a_dem_t.get((r, t)) if data.a_dem_t else data.a_dem.get(r))),
                # Use the actual solved c_man_var level.  Fall back to the LBD formula
                # only when the key is absent (e.g. non-player before first solve).
                "c_man_var": _safe_float(
                    c_man_var_map.get((r, t)),
                    default=max(
                        max(50.0, data.c_man.get(r, 0.0) * 0.5),
                        data.c_man.get(r, 0.0) - 0.022 * _safe_float(w_cum_map.get((r, t))),
                    ),
                ),
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
    convergence_mode = cfg.convergence_mode

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
    print(f"[CONFIG] workdir={workdir}{' (keep)' if cfg.keep_workdir else ' (auto-cleanup)'}")
    if cfg.workers != 1:
        print(f"[WARN] workers={cfg.workers} requested, but run_gs currently executes sequential GS only.")

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

    if cfg.debug_workers and cfg.knitro_outlev is None and solver.strip().lower() == "knitro":
        # Keep debug mode behavior from previous script.
        cfg = dataclass_replace(cfg, knitro_outlev=1)

    solver_opts = _solver_options(
        solver=solver,
        feastol=feastol,
        opttol=opttol,
        cfg=cfg,
    )
    print(f"[CONFIG] solver_options={solver_opts}")

    total_start = time.perf_counter()
    timing_state["sweep_start"] = total_start
    init_state = _build_initial_state(data, cfg)
    print(f"[MAIN] Starting {iters} sweeps with {solver}")

    try:
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
            shuffle_players=cfg.shuffle_players,
        )
    finally:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[TIMING] Total solve time: {total_elapsed:.2f}s")
        if sweep_times:
            print(f"[TIMING] Mean sweep time: {sum(sweep_times)/len(sweep_times):.2f}s  (n={len(sweep_times)})")

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
            # --- Capacity scalers ---
            "kappa_q":               float(cfg.kappa_q) if cfg.kappa_q is not None else 0.0,
            "rho_prox":              float(cfg.rho_prox) if cfg.rho_prox is not None else 0.0,
            "use_quad":              bool(cfg.use_quad),
            # --- Paths ---
            "workdir":               workdir,
        },
    )

    try:
        write_default_plots(output_path=output_path, plots_dir=plots_dir)
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
    print(f"[OK] wrote: {output_path}")

    if not cfg.keep_workdir:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"[CLEANUP] Deleted workdir: {workdir}")
        except Exception as e:
            print(f"[WARN] Could not delete workdir {workdir}: {e}")
    else:
        print(f"[KEEP] Workdir retained: {workdir}")

    return output_path


def dataclass_replace(cfg: RunConfig, **kwargs) -> RunConfig:
    values = dict(cfg.__dict__)
    values.update(kwargs)
    return RunConfig(**values)


def main() -> None:
    run(RunConfig())


if __name__ == "__main__":
    main()
