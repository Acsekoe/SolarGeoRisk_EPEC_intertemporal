from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from model.results_writer import write_results_excel
from scripts.audit_selected_equilibrium import _candidate_state, _zero_prox_data
from scripts.nested_market_audit import nested_economic_objective, solve_nested_market
from scripts.search_nested_equilibrium import _deserialize_state


DEFAULT_PROFILE = (
    ROOT
    / "outputs"
    / "equilibria"
    / "ch-row-apac-us-eu-af"
    / "cost_price_basin_20260914_184749"
    / "profile.json"
)

ACCEPTED_STATUSES = {
    "verified_1pct_epsilon_equilibrium",
    "accepted_one_start_local_1pct_equilibrium",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct and export the certified equilibrium market outcome."
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profile = args.profile.resolve()
    output = (
        args.output.resolve()
        if args.output
        else profile.with_name("certified_equilibrium_results.xlsx")
    )
    certification_path = profile.with_name("certification.json")
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    if certification.get("status") not in ACCEPTED_STATUSES:
        raise RuntimeError(
            "Refusing to export a profile without an accepted certification status: "
            f"{certification_path}"
        )

    _, base_cfg, source_template, _, _ = _candidate_state("source")
    data = _zero_prox_data(base_cfg)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    state = _deserialize_state(payload["state"], data, source_template)
    market, diagnostics = solve_nested_market(data, state)
    state.update(market)
    state["Kcap"] = mm._implied_capacity_path(
        data, list(data.times or []), state["dK_net"]
    )
    state["obj"] = {
        player: nested_economic_objective(data, state, market, player)
        for player in data.players
    }

    try:
        profile_label = str(profile.relative_to(ROOT))
        certification_label = str(certification_path.relative_to(ROOT))
    except ValueError:
        profile_label = str(profile)
        certification_label = str(certification_path)

    write_results_excel(
        data=data,
        state=state,
        iter_rows=[],
        output_path=str(output),
        meta={
            "result_type": "certified equilibrium market reconstruction",
            "profile": profile_label,
            "certification": certification_label,
            "certification_status": certification["status"],
            "maximum_frozen_profile_relative_gain": certification[
                "maximum_frozen_profile_relative_gain"
            ],
            "relative_gain_tolerance": certification["relative_gain_tolerance"],
            "market_diagnostics": json.dumps(diagnostics, sort_keys=True),
            "discount_rate": base_cfg.discount_rate,
            "base_year": base_cfg.base_year,
            "beta_t": str(data.beta_t or {}),
            "ytn": str(data.years_to_next or {}),
            "economic_quadratic_penalties": json.dumps(
                certification["economic_quadratic_penalties"], sort_keys=True
            ),
        },
    )
    print(f"Wrote {output}")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
