from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.continue_selected_equilibrium import PLAYER_ORDER
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.search_nested_equilibrium import (
    _deserialize_state,
    _load,
    _serialize_state,
    _sync_quantity,
    _usable_runs,
)


OUTPUT_DIR = ROOT / "outputs" / "verification" / "ch-row-apac-us-eu-af" / "nested_undamped"
CERTIFIED_CHECKPOINT = (
    ROOT
    / "outputs"
    / "equilibrium_search"
    / "ch-row-apac-us-eu-af"
    / "nested_zero_prox"
    / "checkpoint_20260914_184749.json"
)
CERTIFIED_AUDIT = CERTIFIED_CHECKPOINT.with_name("audit_20260914_184948.json")


def _copy_state(state):
    return {name: dict(values) for name, values in state.items()}


def _audit_profile(data, state, *, starts: int, maxiter: int):
    market, market_diagnostics = solve_nested_market(data, state)
    records = []
    max_gain = 0.0
    all_successful = True
    for player in PLAYER_ORDER:
        reference = nested_economic_objective(data, state, market, player)
        best, best_state, diagnostics = nested_best_response(
            data, state, market, player, maxiter=maxiter, starts=starts
        )
        relative_gain = max(best - reference, 0.0) / max(abs(reference), 1.0)
        move, coordinate, absolute = _strategy_distance(data, state, best_state, player)
        attempt_success = all(bool(item["success"]) for item in diagnostics["attempts"])
        all_successful = all_successful and attempt_success
        max_gain = max(max_gain, relative_gain)
        records.append(
            {
                "player": player,
                "reference_objective": reference,
                "best_response_objective": best,
                "relative_gain": relative_gain,
                "strategy_move": move,
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": absolute,
                "chosen_start_index": diagnostics["chosen_start_index"],
                "all_attempts_successful": attempt_success,
                "market_kkt_residual": diagnostics["market"][
                    "max_positive_flow_stationarity"
                ],
            }
        )
        print(
            f"[UNDAMPED FINAL AUDIT] {player}: gain={relative_gain:.3%} "
            f"move={move:.3g} success={attempt_success}",
            flush=True,
        )
    return {
        "max_relative_gain": max_gain,
        "all_attempts_successful": all_successful,
        "reference_market": market_diagnostics,
        "players": records,
    }


def main() -> None:
    data, template, _ = _load()
    checkpoint_payload = json.loads(CERTIFIED_CHECKPOINT.read_text(encoding="utf-8"))
    audit_payload = json.loads(CERTIFIED_AUDIT.read_text(encoding="utf-8"))
    if not audit_payload.get("equilibrium_verified"):
        raise RuntimeError("Pinned candidate audit is not equilibrium-verified")
    candidate = _deserialize_state(checkpoint_payload["state"], data, template)
    candidate_checkpoint = str(CERTIFIED_CHECKPOINT.relative_to(ROOT))
    state = _copy_state(candidate)
    sweep_records = []
    for player in PLAYER_ORDER:
        market, market_diagnostics = solve_nested_market(data, state)
        reference = nested_economic_objective(data, state, market, player)
        best, best_state, diagnostics = nested_best_response(
            data, state, market, player, maxiter=600, starts=3
        )
        attempt_success = all(bool(item["success"]) for item in diagnostics["attempts"])
        if not attempt_success:
            raise RuntimeError(f"Multistart best response failed for {player}")
        relative_gain = max(best - reference, 0.0) / max(abs(reference), 1.0)
        before = _copy_state(state)
        for tp in mm._move_times(list(data.times or [])):
            state["dK_net"][(player, tp)] = float(best_state["dK_net"][(player, tp)])
        for importer in data.regions:
            if importer == player:
                continue
            for tp in list(data.times or []):
                state["p_offer"][(player, importer, tp)] = float(
                    best_state["p_offer"][(player, importer, tp)]
                )
        _sync_quantity(data, state)
        move, coordinate, absolute = _strategy_distance(data, before, state, player)
        sweep_records.append(
            {
                "player": player,
                "reference_objective": reference,
                "best_response_objective": best,
                "relative_gain": relative_gain,
                "undamped_strategy_move": move,
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": absolute,
                "chosen_start_index": diagnostics["chosen_start_index"],
                "all_attempts_successful": attempt_success,
                "reference_market_kkt_residual": market_diagnostics[
                    "max_positive_flow_stationarity"
                ],
            }
        )
        print(
            f"[UNDAMPED GS] {player}: gain={relative_gain:.3%} "
            f"move={move:.3g} success={attempt_success}",
            flush=True,
        )

    final_audit = _audit_profile(data, state, starts=3, maxiter=600)
    finished = datetime.now().astimezone()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"verification_{finished.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "created": finished.isoformat(timespec="seconds"),
        "candidate_checkpoint": candidate_checkpoint,
        "omega": 1.0,
        "sweeps": 1,
        "algorithmic_proximal_penalties": 0.0,
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "multistart_count": 3,
        "sweep": sweep_records,
        "ending_state": _serialize_state(state),
        "final_common_profile_audit": final_audit,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[UNDAMPED VERIFY] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
