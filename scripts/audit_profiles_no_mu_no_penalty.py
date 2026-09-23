from __future__ import annotations

"""Audit representative accepted profiles under a counterfactual ULP objective.

The counterfactual removes both:

1. the subtraction of the offer-capacity dual ``mu_offer`` from producer
   revenue; and
2. all economic quadratic and algorithmic proximal penalties.

The lower-level market formulation and each common strategy profile are held
fixed while one player at a time computes an unrestricted best response.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import data_prep
from model import model_main as mm
from scripts.audit_selected_equilibrium import _strategy_distance, _zero_prox_data
from scripts.continue_selected_equilibrium import _initial_model_data
from scripts import nested_market_audit as nma
from scripts.search_nested_equilibrium import _deserialize_state


CANDIDATE_ROOT = ROOT / "outputs" / "new_equilibria" / "candidates"
DEFAULT_PROFILES = (
    "ch-af-apac-eu-row-us/pf080_k100_a030",
    "af-eu-us-apac-row-ch/pf100_k050_a040",
    "eu-us-af-row-apac-ch/pf100_k050_a030",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def counterfactual_objective(
    data: mm.ModelData,
    candidate: dict[str, dict],
    market_state: dict[str, dict],
    player: str,
) -> float:
    """ULP payoff without ``-mu_offer*x`` or any quadratic penalty."""

    times = list(data.times or [])
    operating_times = mm._operating_times(data)
    move_times = set(mm._move_times(times))
    kcap = mm._implied_capacity_path(data, times, candidate["dK_net"])
    value = 0.0

    for tp in operating_times:
        weight = float((data.beta_t or {}).get(tp, 1.0)) * float(
            (data.years_to_next or {}).get(tp, 1.0)
        )
        demand = float(market_state["x_dem"][(player, tp)])
        own_price = float(market_state["lam"][(player, tp)])
        a_dem = float((data.a_dem_t or {})[(player, tp)])
        b_dem = float((data.b_dem_t or {})[(player, tp)])
        c_man = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))

        period = a_dem * demand - 0.5 * b_dem * demand * demand - own_price * demand
        for importer in data.regions:
            flow = float(market_state["x"][(player, importer, tp)])
            period += (
                float(market_state["lam"][(importer, tp)])
                - c_man
                - float(data.c_ship[(player, importer)])
            ) * flow

        period -= float((data.f_hold or {})[player]) * float(kcap[(player, tp)])
        if tp in move_times:
            investment = max(float(candidate["dK_net"][(player, tp)]), 0.0)
            period -= float((data.c_inv or {})[player]) * investment
        value += weight * period

    value += mm._terminal_salvage_credit(
        data,
        player,
        kcap[(player, times[-1])],
    )
    return value


def audit_profile(
    profile_id: str,
    *,
    base_cfg: Any,
    template: dict[str, dict],
    starts: int,
    maxiter: int,
    output_dir: Path,
) -> dict[str, Any]:
    branch_root = CANDIDATE_ROOT / Path(profile_id)
    profile_path = branch_root / "profile.json"
    original_audit_path = branch_root / "audit_one_start.json"
    profile_document = load_json(profile_path)
    original_audit = load_json(original_audit_path)

    salvage_fraction = float(profile_document.get("terminal_salvage_fraction", 0.0))
    data = _zero_prox_data(
        base_cfg,
        terminal_salvage_fraction=salvage_fraction,
    )
    data.settings.update(
        {
            "c_pen_q": 0.0,
            "c_pen_p": 0.0,
            "c_pen_a": 0.0,
            "c_pen_dk": 0.0,
            "c_quad_q": 0.0,
            "c_quad_p": 0.0,
            "c_quad_a": 0.0,
            "subtract_mu_offer_from_producer_margin": False,
            "terminal_capacity_state_only": True,
        }
    )
    state = _deserialize_state(
        profile_document["ending_profile"]["strategy"],
        data,
        template,
    )
    market, market_diagnostics = nma.solve_nested_market(data, state)

    rows: list[dict[str, Any]] = []
    max_gain = 0.0
    all_successful = True
    for player in data.players:
        reference = float(counterfactual_objective(data, state, market, player))
        best, best_state, diagnostics = nma.nested_best_response(
            data,
            state,
            market,
            player,
            maxiter=maxiter,
            starts=starts,
        )
        relative_gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
        move, coordinate, absolute_move = _strategy_distance(
            data,
            state,
            best_state,
            player,
        )
        attempts_successful = all(
            bool(attempt["success"]) for attempt in diagnostics["attempts"]
        )
        all_successful = all_successful and attempts_successful
        max_gain = max(max_gain, relative_gain)
        rows.append(
            {
                "player": player,
                "reference_objective": reference,
                "best_response_objective": float(best),
                "relative_gain": relative_gain,
                "relative_gain_percent": 100.0 * relative_gain,
                "strategy_move": float(move),
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": float(absolute_move),
                "optimizer_success": bool(diagnostics["success"]),
                "all_attempts_successful": attempts_successful,
                "chosen_start_index": int(diagnostics["chosen_start_index"]),
                "attempts": diagnostics["attempts"],
                "capacity_feasibility_min": float(
                    diagnostics["capacity_feasibility_min"]
                ),
                "best_response_market_diagnostics": diagnostics["market"],
            }
        )
        print(
            f"[NO-MU/NO-PENALTY {profile_id}] {player}: "
            f"gain={relative_gain:.3%} success={diagnostics['success']}",
            flush=True,
        )

    limiting = max(rows, key=lambda row: float(row["relative_gain"]))
    successful_rows = [
        row for row in rows if bool(row["all_attempts_successful"])
    ]
    successful_limiting = (
        max(successful_rows, key=lambda row: float(row["relative_gain"]))
        if successful_rows
        else None
    )
    payload = {
        "created": now(),
        "profile_id": profile_id,
        "profile_path": str(profile_path.relative_to(ROOT)),
        "audit_design": {
            "common_profile_frozen": True,
            "starts_per_best_response": starts,
            "maxiter": maxiter,
            "relative_gain_tolerance": 0.01,
            "algorithmic_proximal_penalties": 0.0,
            "economic_quadratic_penalties": 0.0,
            "subtract_mu_offer_from_producer_margin": False,
            "lower_level_market_unchanged": True,
            "terminal_capacity_state_only": True,
            "terminal_salvage_fraction": salvage_fraction,
        },
        "original_one_start_audit": {
            "max_relative_gain": float(original_audit["max_relative_gain"]),
            "max_gain_player": str(original_audit["max_gain_player"]),
            "equilibrium_verified": bool(original_audit["equilibrium_verified"]),
        },
        "reference_market_diagnostics": market_diagnostics,
        "players": rows,
        "max_relative_gain": max_gain,
        "max_gain_player": str(limiting["player"]),
        "max_successful_relative_gain": (
            None
            if successful_limiting is None
            else float(successful_limiting["relative_gain"])
        ),
        "max_successful_gain_player": (
            None if successful_limiting is None else str(successful_limiting["player"])
        ),
        "all_attempts_successful": all_successful,
        "equilibrium_verified": bool(all_successful and max_gain <= 0.01),
    }
    destination = output_dir / profile_id / "audit_no_mu_no_penalty.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[NO-MU/NO-PENALTY {profile_id}] max={max_gain:.3%} "
        f"verified={payload['equilibrium_verified']}",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit accepted profiles after removing mu_offer from the ULP producer "
            "margin and setting all quadratic/proximal penalties to zero."
        )
    )
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.starts < 1 or args.starts > 3:
        raise ValueError("--starts must be between 1 and 3")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        ROOT
        / "outputs"
        / "new_equilibria"
        / f"counterfactual_no_mu_no_penalty_{stamp}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # These profiles were generated with the explicit a_dem/b_dem columns now
    # stored in params_region_new.  The workbook's p_full/elasticity metadata
    # has since drifted and fails the newer replay guard, so relax only that
    # guard while retaining the explicit coefficients used by the profiles.
    original_demand_tolerance = data_prep._DEMAND_CALIBRATION_ABS_TOL
    data_prep._DEMAND_CALIBRATION_ABS_TOL = float("inf")
    _, base_cfg, template = _initial_model_data()
    original_objective = nma.nested_economic_objective
    nma.nested_economic_objective = counterfactual_objective
    try:
        results = [
            audit_profile(
                profile_id,
                base_cfg=base_cfg,
                template=template,
                starts=args.starts,
                maxiter=args.maxiter,
                output_dir=output_dir,
            )
            for profile_id in args.profiles
        ]
    finally:
        nma.nested_economic_objective = original_objective
        data_prep._DEMAND_CALIBRATION_ABS_TOL = original_demand_tolerance

    summary_rows = [
        {
            "profile_id": result["profile_id"],
            "original_max_gain_percent": 100.0
            * float(result["original_one_start_audit"]["max_relative_gain"]),
            "counterfactual_max_gain_percent": 100.0
            * float(result["max_relative_gain"]),
            "counterfactual_limiting_player": result["max_gain_player"],
            "max_successful_gain_percent": (
                None
                if result["max_successful_relative_gain"] is None
                else 100.0 * float(result["max_successful_relative_gain"])
            ),
            "max_successful_gain_player": result["max_successful_gain_player"],
            "all_attempts_successful": result["all_attempts_successful"],
            "counterfactual_equilibrium_verified": result["equilibrium_verified"],
        }
        for result in results
    ]
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "created": now(),
        "experiment": "counterfactual frozen-profile equilibrium audit",
        "profiles": list(args.profiles),
        "starts_per_best_response": args.starts,
        "maxiter": args.maxiter,
        "objective_changes": {
            "algorithmic_proximal_penalties": 0.0,
            "economic_quadratic_penalties": 0.0,
            "subtract_mu_offer_from_producer_margin": False,
        },
        "terminal_capacity_state_only": True,
        "data_replay": {
            "explicit_demand_coefficients_preserved": True,
            "p_full_elasticity_consistency_guard_relaxed": True,
        },
        "summary": summary_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[NO-MU/NO-PENALTY] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
