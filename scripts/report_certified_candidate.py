from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _candidate_state, _zero_prox_data
from scripts.nested_market_audit import nested_economic_objective, solve_nested_market
from scripts.search_nested_equilibrium import _deserialize_state


CHECKPOINT = (
    ROOT
    / "outputs"
    / "equilibrium_search"
    / "ch-row-apac-us-eu-af"
    / "nested_zero_prox"
    / "checkpoint_20260914_184749.json"
)
AUDIT = CHECKPOINT.with_name("audit_20260914_184948.json")
OUTPUT = (
    ROOT
    / "outputs"
    / "verification"
    / "ch-row-apac-us-eu-af"
    / "certified_candidate_comparison.json"
)


def main() -> None:
    _, base_cfg, source, _, source_label = _candidate_state("source")
    data = _zero_prox_data(base_cfg)
    candidate_payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    candidate = _deserialize_state(candidate_payload["state"], data, source)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    source_market, source_market_diag = solve_nested_market(data, source)
    candidate_market, candidate_market_diag = solve_nested_market(data, candidate)
    source_kcap = mm._implied_capacity_path(data, list(data.times or []), source["dK_net"])
    candidate_kcap = mm._implied_capacity_path(
        data, list(data.times or []), candidate["dK_net"]
    )
    audit_by_player = {row["player"]: row for row in audit["players"]}
    players = {}
    for player in data.players:
        source_obj = nested_economic_objective(data, source, source_market, player)
        candidate_obj = nested_economic_objective(
            data, candidate, candidate_market, player
        )
        price_keys = [
            (player, importer, tp)
            for importer in data.regions
            if importer != player
            for tp in list(data.times or [])
        ]
        source_prices = [float(source["p_offer"][key]) for key in price_keys]
        candidate_prices = [float(candidate["p_offer"][key]) for key in price_keys]
        changes = [after - before for before, after in zip(source_prices, candidate_prices)]
        largest_index = max(range(len(changes)), key=lambda index: abs(changes[index]))
        largest_key = price_keys[largest_index]
        source_markups = []
        candidate_markups = []
        for key, before, after in zip(price_keys, source_prices, candidate_prices):
            _, _, tp = key
            cost = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))
            source_markups.append(before - cost)
            candidate_markups.append(after - cost)
        players[player] = {
            "objective": {
                "source": source_obj,
                "candidate": candidate_obj,
                "relative_change": (candidate_obj - source_obj) / max(abs(source_obj), 1.0),
                "certified_best_response_relative_gain": audit_by_player[player][
                    "relative_gain"
                ],
            },
            "capacity": {
                tp: {
                    "source": float(source_kcap[(player, tp)]),
                    "candidate": float(candidate_kcap[(player, tp)]),
                    "change": float(candidate_kcap[(player, tp)])
                    - float(source_kcap[(player, tp)]),
                }
                for tp in list(data.times or [])
            },
            "export_offers": {
                "source_mean": sum(source_prices) / len(source_prices),
                "candidate_mean": sum(candidate_prices) / len(candidate_prices),
                "mean_change": sum(changes) / len(changes),
                "source_mean_markup": sum(source_markups) / len(source_markups),
                "candidate_mean_markup": sum(candidate_markups) / len(candidate_markups),
                "largest_absolute_change": {
                    "coordinate": f"p_offer[{largest_key[0]},{largest_key[1]},{largest_key[2]}]",
                    "source": source_prices[largest_index],
                    "candidate": candidate_prices[largest_index],
                    "change": changes[largest_index],
                },
            },
        }
    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source_label,
        "candidate_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "candidate_audit": str(AUDIT.relative_to(ROOT)),
        "algorithmic_proximal_penalties": 0.0,
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "source_market_diagnostics": source_market_diag,
        "candidate_market_diagnostics": candidate_market_diag,
        "players": players,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    for player, row in players.items():
        print(
            f"{player}: objective_change={row['objective']['relative_change']:.3%} "
            f"audit_gain={row['objective']['certified_best_response_relative_gain']:.3%} "
            f"mean_offer_change={row['export_offers']['mean_change']:.3f}"
        )


if __name__ == "__main__":
    main()
