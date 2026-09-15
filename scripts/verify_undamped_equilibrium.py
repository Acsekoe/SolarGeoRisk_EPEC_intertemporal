from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import run_gs
from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.continue_selected_equilibrium import (
    PLAYER_ORDER,
    _initial_model_data,
    replay_accepted_state,
)
from scripts.nested_market_audit import nested_economic_objective, solve_nested_market
from scripts.search_nested_equilibrium import _deserialize_state, _load


OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verification"
    / "ch-row-apac-us-eu-af"
    / "original_mpec_undamped_zero_prox"
)
MANIFEST_PATH = ROOT / "workflow" / "final_undamped_verification_manifest.json"
CERTIFIED_CHECKPOINT = (
    ROOT
    / "outputs"
    / "equilibrium_search"
    / "ch-row-apac-us-eu-af"
    / "nested_zero_prox"
    / "checkpoint_20260914_184749.json"
)
CERTIFIED_AUDIT = CERTIFIED_CHECKPOINT.with_name("audit_20260914_184948.json")


def _objectives(
    data, state
) -> tuple[dict[str, float], dict[str, float], dict[str, dict]]:
    market, diagnostics = solve_nested_market(data, state)
    objectives = {
        player: float(nested_economic_objective(data, state, market, player))
        for player in PLAYER_ORDER
    }
    return objectives, diagnostics, market


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one original MPEC Gauss-Seidel verification sweep."
    )
    parser.add_argument("--solver", choices=["conopt", "ipopt"], default="conopt")
    args = parser.parse_args()
    data, template, _ = _load()
    checkpoint_payload = json.loads(CERTIFIED_CHECKPOINT.read_text(encoding="utf-8"))
    audit_payload = json.loads(CERTIFIED_AUDIT.read_text(encoding="utf-8"))
    if not audit_payload.get("equilibrium_verified"):
        raise RuntimeError("Pinned candidate audit is not equilibrium-verified")
    candidate = _deserialize_state(checkpoint_payload["state"], data, template)
    candidate_checkpoint = str(CERTIFIED_CHECKPOINT.relative_to(ROOT))
    input_objectives, input_market, input_market_state = _objectives(data, candidate)
    warm_candidate = {name: dict(values) for name, values in candidate.items()}
    warm_candidate.update(
        {name: dict(values) for name, values in input_market_state.items()}
    )

    _, base_cfg, _ = _initial_model_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = run_gs.RunConfig(
        excel_path=base_cfg.excel_path,
        out_dir=str(OUTPUT_DIR),
        plots_dir=str(OUTPUT_DIR / "plots"),
        params_region_sheet="params_region_new",
        solver=args.solver,
        feastol=1e-4,
        opttol=1e-4,
        iters=1,
        omega=1.0,
        adaptive_omega=False,
        omega_min=1.0,
        omega_aggressive_sweeps=0,
        omega_ramp_iters=1,
        tol_strat=1e-12,
        tol_raw_br=1e-12,
        stable_iters=2,
        eps_x=1e-3,
        eps_comp=1e-3,
        exclude_terminal_from_convergence=False,
        keep_workdir=False,
        c_pen_q=0.0,
        c_pen_p=0.0,
        c_pen_a=0.0,
        c_pen_dk=0.0,
        c_pen_q_mid=None,
        c_pen_p_mid=None,
        c_pen_a_mid=None,
        c_pen_dk_mid=None,
        c_pen_q_final=None,
        c_pen_p_final=None,
        c_pen_a_final=None,
        c_pen_dk_final=None,
        c_pen_ramp_iters=1,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        cap_keep_reward=0.0,
        capex_subsidy=0.0,
        terminal_capacity_value=0.0,
        decommission_penalty=0.0,
        fix_q_offer_to_kcap=True,
        force_mu_offer_zero=False,
        fix_a_bid_to_true_dem=True,
        discount_rate=0.02,
        base_year=2025,
        initial_state_override=warm_candidate,
        player_order=PLAYER_ORDER,
    )

    run_gs.write_default_plots = None
    started = datetime.now().astimezone()
    print(
        f"[FINAL VERIFY] original GAMS/{args.solver} sweep: iters=1 omega=1 "
        "c_pen=(0,0,0,0) c_quad=(0.1,0.1,0.1)",
        flush=True,
    )
    output_path = Path(run_gs.run(cfg))
    finished = datetime.now().astimezone()
    output_state, replay_error = replay_accepted_state(output_path, candidate, data)
    output_objectives, output_market, _ = _objectives(data, output_state)
    strategy_changes = {}
    for player in PLAYER_ORDER:
        move, coordinate, absolute = _strategy_distance(
            data, candidate, output_state, player
        )
        strategy_changes[player] = {
            "normalized_move": move,
            "largest_coordinate": coordinate,
            "largest_absolute_move": absolute,
        }
    objective_changes = {
        player: {
            "before": input_objectives[player],
            "after": output_objectives[player],
            "relative_change": (
                output_objectives[player] - input_objectives[player]
            )
            / max(abs(input_objectives[player]), 1.0),
        }
        for player in PLAYER_ORDER
    }
    iter_frame = pd.read_excel(output_path, sheet_name="iters")
    detailed = pd.read_excel(output_path, sheet_name="detailed_iters")
    last = iter_frame.iloc[-1]
    turn_objectives = {
        str(player): float(group.iloc[-1]["obj"])
        for player, group in detailed.groupby("r", sort=False)
    }
    payload = {
        "created": finished.isoformat(timespec="seconds"),
        "started": started.isoformat(timespec="seconds"),
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_audit": str(CERTIFIED_AUDIT.relative_to(ROOT)),
        "workbook": str(output_path.relative_to(ROOT)),
        "solver": args.solver,
        "sweeps": 1,
        "omega": 1.0,
        "algorithmic_proximal_penalties": {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0},
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "terminal_in_strategy_residual": True,
        "reported_r_strat": float(last["r_strat"]),
        "replay_error": float(replay_error),
        "input_market_diagnostics": input_market,
        "output_market_diagnostics": output_market,
        "common_profile_objective_changes": objective_changes,
        "strategy_changes": strategy_changes,
        "sequential_turn_objectives_from_workbook": turn_objectives,
    }
    if MANIFEST_PATH.exists():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        prior_runs = list(previous.get("runs", [])) if "runs" in previous else [previous]
    else:
        prior_runs = []
    prior_runs.append(payload)
    MANIFEST_PATH.write_text(
        json.dumps({"latest": payload, "runs": prior_runs}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[FINAL VERIFY] wrote {output_path}", flush=True)
    print(f"[FINAL VERIFY] manifest {MANIFEST_PATH}", flush=True)
    print(
        f"[FINAL VERIFY] reported_r_strat={payload['reported_r_strat']:.6g} "
        f"replay_error={replay_error:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
