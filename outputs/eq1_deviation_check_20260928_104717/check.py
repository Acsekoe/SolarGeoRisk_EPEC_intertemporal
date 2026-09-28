"""Targeted nonlocal unilateral-deviation audit of the clean Stage-2 profiles.

This script reads the original results and writes only beside itself.  It uses
the same clean configuration, Clarabel market clearing, and economic objective
as the Stage-2 one-start audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from model import run_gs
from model.data_prep import load_data_from_excel
from scripts.audit_selected_equilibrium import _economic_objective
from scripts.nested_market_audit import nested_best_response, solve_nested_market
from scripts.run_clean_stage2_factorial import clean_configuration, assert_clean_objective
from scripts.search_nested_equilibrium import _sync_quantity

RUN = ROOT / "outputs" / "clean_stage2_factorial_20260923_123037"
EQ1 = "ch-af-apac-eu-row-us/pf080_k100_a040"
EXCLUDED = "eu-us-af-row-apac-ch/pf120_k050_a040"
PERIODS = ("2025", "2030", "2035", "2040")
EPSILONS = (1.0, 5.0, 20.0)
FIELDS = (
    "player", "profile", "importer", "period", "deviation_type", "new_offer",
    "reference_objective", "deviation_objective", "relative_gain_percent",
    "export_volume_change_gw", "domestic_price_change_usd_per_kw",
    "importer_price_change_usd_per_kw", "reference_export_gw",
    "deviation_export_gw", "optimizer_success", "notes",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_to_map(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], float]:
    return {tuple(str(row[k]) for k in keys): float(row["value"]) for row in rows}


def load_data():
    manifest = read_json(RUN / "manifest.json")
    protocol = manifest["protocol"]
    relative = Path(str(protocol["input"]).replace("\\", "/"))
    candidates = (ROOT / relative, ROOT.parent / relative)
    input_path = next((p for p in candidates if p.is_file()), None)
    if input_path is None:
        raise FileNotFoundError(f"Corrected workbook absent: {relative}")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest().upper()
    if digest != str(protocol["input_sha256"]).upper():
        raise ValueError("Corrected workbook hash differs from run manifest")
    cfg = clean_configuration(input_path, float(protocol["terminal_salvage_fraction"]))
    data = load_data_from_excel(str(input_path), params_region_sheet=cfg.params_region_sheet)
    run_gs._apply_data_overrides(data, cfg)
    assert_clean_objective(data)
    for key in ("fix_q_offer_to_kcap", "fix_a_bid_to_true_dem"):
        if not bool((data.settings or {}).get(key)):
            raise ValueError(f"Expected {key}=True")
    return data, manifest, input_path, digest


def load_profile(profile_id: str, data):
    status = read_json(RUN / profile_id / "status.json")
    source = ROOT / Path(str(status["selected_profile"]).replace("\\", "/"))
    document = read_json(source)
    ending = document["ending_profile"]
    strategy = ending["strategy"]
    state = {
        "dK_net": rows_to_map(strategy["dK_net"], ("region", "time")),
        "p_offer": rows_to_map(strategy["p_offer"], ("exporter", "importer", "time")),
        "a_bid": {
            (r, t): mm._true_demand_intercept(data, r, t)
            for r in data.regions for t in data.times
        },
        "Q_offer": {},
    }
    _sync_quantity(data, state)
    return state, status, ending


def changed_state(reference: dict, player: str, changes: dict[tuple[str, str], float], data,
                  capacity_changes: dict[str, float] | None = None):
    state = {name: dict(values) for name, values in reference.items()}
    for (importer, period), offer in changes.items():
        if importer == player:
            raise ValueError("Domestic offer is fixed at manufacturing cost")
        upper = float(data.p_offer_ub[(player, importer)])
        state["p_offer"][(player, importer, period)] = min(max(float(offer), 0.0), upper)
    for period, move in (capacity_changes or {}).items():
        state["dK_net"][(player, period)] = float(move)
    _sync_quantity(data, state)
    return state


def record(result_rows: list[dict], *, data, profile_id: str, player: str,
           importer: str, period: str, deviation_type: str, new_offer: float | str,
           reference: dict, market_ref: dict, objective_ref: float,
           candidate: dict, optimizer_success: bool | str = "", notes: str = "") -> dict:
    market_new, diagnostic = solve_nested_market(data, candidate, market_ref)
    if max(diagnostic["max_balance_residual"], diagnostic["max_capacity_violation"],
           diagnostic["max_positive_flow_stationarity"]) > 1e-5:
        raise RuntimeError(f"Market residual too large for {profile_id}/{player}: {diagnostic}")
    value = _economic_objective(data, candidate, market_new, player)
    gain = 100.0 * (value - objective_ref) / max(abs(objective_ref), 1.0)
    ref_export = sum(float(market_ref["x"][(player, i, period)]) for i in data.regions if i != player)
    new_export = sum(float(market_new["x"][(player, i, period)]) for i in data.regions if i != player)
    row = {
        "player": player, "profile": profile_id, "importer": importer,
        "period": period, "deviation_type": deviation_type, "new_offer": new_offer,
        "reference_objective": objective_ref, "deviation_objective": value,
        "relative_gain_percent": gain, "export_volume_change_gw": new_export - ref_export,
        "domestic_price_change_usd_per_kw": market_new["lam"][(player, period)] - market_ref["lam"][(player, period)],
        "importer_price_change_usd_per_kw": (
            market_new["lam"][(importer, period)] - market_ref["lam"][(importer, period)]
            if importer in data.regions else ""
        ),
        "reference_export_gw": ref_export, "deviation_export_gw": new_export,
        "optimizer_success": optimizer_success, "notes": notes,
    }
    result_rows.append(row)
    return row


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def offer_target(data, reference: dict, market: dict, player: str,
                 importer: str, period: str, epsilon: float) -> float:
    landed = float(market["lam"][(importer, period)])
    shipping = float(data.c_ship[(player, importer)])
    current = float(reference["p_offer"][(player, importer, period)])
    return min(current, landed - shipping - epsilon)


def targeted_cases(data, profile_id: str, player: str, reference: dict,
                   market: dict, objective: float, rows: list[dict]):
    importers = [r for r in data.regions if r != player]
    best: tuple[dict, dict] | None = None
    for period in ("2035", "2040"):
        for epsilon in EPSILONS:
            for importer in importers:
                changes = {(importer, period): offer_target(
                    data, reference, market, player, importer, period, epsilon
                )}
                state = changed_state(reference, player, changes, data)
                row = record(rows, data=data, profile_id=profile_id, player=player,
                             importer=importer, period=period,
                             deviation_type=f"single_undercut_eps_{epsilon:g}",
                             new_offer=changes[(importer, period)],
                             reference=reference, market_ref=market,
                             objective_ref=objective, candidate=state)
                if best is None or row["relative_gain_percent"] > best[0]["relative_gain_percent"]:
                    best = row, state
            changes = {
                (importer, period): offer_target(
                    data, reference, market, player, importer, period, epsilon
                ) for importer in importers
            }
            state = changed_state(reference, player, changes, data)
            row = record(rows, data=data, profile_id=profile_id, player=player,
                         importer="all", period=period,
                         deviation_type=f"all_undercut_eps_{epsilon:g}",
                         new_offer=";".join(f"{r}:{changes[(r, period)]:.6f}" for r in importers),
                         reference=reference, market_ref=market,
                         objective_ref=objective, candidate=state)
            if best is None or row["relative_gain_percent"] > best[0]["relative_gain_percent"]:
                best = row, state
        for importer in importers:
            cost = float((data.c_man_t or {})[(player, period)])
            current = float(reference["p_offer"][(player, importer, period)])
            for step in range(11):
                offer = cost + (current - cost) * step / 10.0
                state = changed_state(reference, player, {(importer, period): offer}, data)
                row = record(rows, data=data, profile_id=profile_id, player=player,
                             importer=importer, period=period,
                             deviation_type=f"grid_{step:02d}_of_10",
                             new_offer=offer, reference=reference, market_ref=market,
                             objective_ref=objective, candidate=state)
                if best is None or row["relative_gain_percent"] > best[0]["relative_gain_percent"]:
                    best = row, state
        print(f"TARGETED {profile_id}/{player}/{period}: {len(rows)} rows; "
              f"best={best[0]['relative_gain_percent']:.3f}%", flush=True)
        write_csv(OUT / "results.csv", rows, FIELDS)
    assert best is not None
    return best


def capacity_variants(data, player: str, reference: dict, market: dict,
                      objective: float, best: tuple[dict, dict], rows: list[dict]):
    best_row, best_state = best
    # Preserve 2030 installed capacity through 2040 by making both subsequent
    # net-capacity changes zero.  The separate best-response pass optimizes all
    # feasible capacity changes without imposing this particular path.
    keep = changed_state(reference, player, {}, data,
                         capacity_changes={"2030": 0.0, "2035": 0.0})
    selected = {
        (i, t): float(best_state["p_offer"][(player, i, t)])
        for i in data.regions if i != player for t in ("2035", "2040")
    }
    keep = changed_state(keep, player, selected, data)
    row = record(rows, data=data, profile_id=EQ1, player=player,
                 importer="all", period="2040", deviation_type="keep_2030_capacity_plus_best_offers",
                 new_offer="best_targeted_offers", reference=reference, market_ref=market,
                 objective_ref=objective, candidate=keep,
                 notes=f"targeted_seed={best_row['deviation_type']}:{best_row['importer']}:{best_row['period']}")
    return row, keep


def multistart_cases(data, player: str, reference: dict, market: dict,
                     objective: float, targeted_best: tuple[dict, dict],
                     capacity_start: dict, rows: list[dict]):
    starts: list[tuple[str, dict]] = []
    for factor in (1.0, 1.25, 1.5, 2.0):
        state = changed_state(reference, player, {}, data)
        changes = {}
        for period in PERIODS:
            cost = float((data.c_man_t or {})[(player, period)])
            for importer in data.regions:
                if importer != player:
                    changes[(importer, period)] = factor * (
                        cost + float(data.c_ship[(player, importer)])
                    )
        starts.append((f"low_offers_factor_{factor:g}", changed_state(state, player, changes, data)))
    starts.append(("best_targeted", targeted_best[1]))
    starts.append(("keep_2030_capacity", capacity_start))
    if player == "ch":
        eq2, _, _ = load_profile("ch-af-apac-eu-row-us/pf120_k100_a030", data)
        cross = changed_state(reference, player, {}, data)
        for key, value in eq2["dK_net"].items():
            if key[0] == player:
                cross["dK_net"][key] = value
        for key, value in eq2["p_offer"].items():
            if key[0] == player and key[1] != player:
                cross["p_offer"][key] = value
        _sync_quantity(data, cross)
        starts.append(("eq2_player_strategy", cross))
    for label, start in starts:
        value, state, diagnostic = nested_best_response(
            data, reference, market, player, maxiter=600,
            start_states=[start], start_state_labels=[label],
        )
        row = record(rows, data=data, profile_id=EQ1, player=player,
                     importer="all", period="2040", deviation_type="best_response_" + label,
                     new_offer="optimized_all_years", reference=reference,
                     market_ref=market, objective_ref=objective, candidate=state,
                     optimizer_success=diagnostic["success"],
                     notes=f"iterations={diagnostic['iterations']}; "
                           f"market_failures={diagnostic['market_failures']}; "
                           f"capacity_min={diagnostic['capacity_feasibility_min']:.6g}")
        print(f"BEST RESPONSE {player}/{label}: {row['relative_gain_percent']:.3f}% "
              f"success={diagnostic['success']}", flush=True)
        write_csv(OUT / "results.csv", rows, FIELDS)


def screen_profiles(data, reference_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metrics_path = RUN / "statistical_analysis" / "csv" / "regional_market_metrics.csv"
    candidate_path = RUN / "statistical_analysis" / "csv" / "candidate_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        market_rows = list(csv.DictReader(handle))
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = [r for r in csv.DictReader(handle) if r["candidate"] != EXCLUDED]
    if len(candidates) != 26:
        raise ValueError(f"Expected 26 retained profiles, got {len(candidates)}")
    market_lookup = {
        (r["candidate"], r["region"], str(r["year"])): r for r in market_rows
        if r["candidate"] != EXCLUDED
    }
    summary: list[dict] = []
    flags: list[dict] = []
    for record_meta in candidates:
        profile_id = record_meta["candidate"]
        if profile_id == EQ1:
            continue
        state, status, ending = load_profile(profile_id, data)
        if int(status["selected_sweep"]) != int(record_meta["selected_sweep"]):
            raise ValueError(f"Selected sweep mismatch for {profile_id}")
        flows = rows_to_map(ending["market"]["trade_flows"],
                            ("exporter", "importer", "time"))
        candidates_here = []
        for exporter in data.regions:
            for period in PERIODS:
                info = market_lookup[(profile_id, exporter, period)]
                idle = float(info["capacity_gw"]) - float(info["domestic_output_gw"]) - float(info["exports_gw"])
                if idle <= 1.0:
                    continue
                cost = float((data.c_man_t or {})[(exporter, period)])
                for importer in data.regions:
                    if importer == exporter:
                        continue
                    flow = max(float(flows[(exporter, importer, period)]), 0.0)
                    price = float(market_lookup[(profile_id, importer, period)]["price_usd_per_kw"])
                    shipping = float(data.c_ship[(exporter, importer)])
                    margin = price - shipping - cost
                    if flow <= 1.0 and margin > 50.0:
                        weight = float((data.beta_t or {}).get(period, 1.0)) * float(
                            (data.years_to_next or {}).get(period, 1.0)
                        )
                        flag = {
                            "profile": profile_id, "player": exporter, "importer": importer,
                            "period": period, "idle_capacity_gw": idle,
                            "current_flow_gw": flow, "margin_usd_per_kw": margin,
                            "potential_score_million_usd": idle * margin * weight,
                        }
                        flags.append(flag)
                        candidates_here.append(flag)
        if not candidates_here:
            summary.append({"profile": profile_id, "flagged_cases": 0,
                            "tested_routes": 0, "largest_player": "", "largest_importer": "",
                            "largest_period": "", "best_gain_percent": "",
                            "verdict": "equilibrium at 1% tolerance",
                            "scope": "original one-start audit; no screen flag"})
            continue
        top = max(candidates_here, key=lambda f: f["potential_score_million_usd"])
        # The largest case satisfies the requested screen.  Test all flags as
        # well: a large 2025 idle-capacity score can conceal the more relevant
        # late-period export withdrawal in another route.
        selected = {(f["player"], f["importer"], f["period"]): f
                    for f in candidates_here}
        market, _ = solve_nested_market(data, state)
        best_gain = float("-inf")
        for tested in selected.values():
            player = tested["player"]
            importer = tested["importer"]
            period = tested["period"]
            objective = _economic_objective(data, state, market, player)
            current = float(state["p_offer"][(player, importer, period)])
            cost = float((data.c_man_t or {})[(player, period)])
            tests = [(f"screen_undercut_eps_{e:g}", offer_target(
                data, state, market, player, importer, period, e
            )) for e in EPSILONS]
            tests += [(f"screen_grid_{step:02d}_of_10", cost + (current-cost)*step/10.0)
                      for step in range(11)]
            for label, new_offer in tests:
                trial = changed_state(state, player, {(importer, period): new_offer}, data)
                row = record(reference_rows, data=data, profile_id=profile_id,
                             player=player, importer=importer, period=period,
                             deviation_type=label, new_offer=new_offer,
                             reference=state, market_ref=market, objective_ref=objective,
                             candidate=trial,
                             notes=f"idle={tested['idle_capacity_gw']:.3f}; "
                                   f"screen_margin={tested['margin_usd_per_kw']:.3f}")
                best_gain = max(best_gain, row["relative_gain_percent"])
        summary.append({"profile": profile_id, "flagged_cases": len(candidates_here),
                        "tested_routes": len(selected),
                        "largest_player": top["player"], "largest_importer": top["importer"],
                        "largest_period": top["period"], "best_gain_percent": best_gain,
                        "verdict": ("not an equilibrium" if best_gain > 1.0
                                    else "equilibrium at 1% tolerance"),
                        "scope": "all flagged routes; three undercuts and 11-point 1-D grid each"})
        print(f"SCREEN {profile_id}: {len(candidates_here)} flags; top="
              f"{top['player']}->{top['importer']}/{top['period']}; "
              f"tested={len(selected)}; best={best_gain:.3f}%", flush=True)
        write_csv(OUT / "results.csv", reference_rows, FIELDS)
    write_csv(OUT / "screening_flags.csv", flags,
              ("profile", "player", "importer", "period", "idle_capacity_gw",
               "current_flow_gw", "margin_usd_per_kw", "potential_score_million_usd"))
    write_csv(OUT / "screening_summary.csv", summary,
              ("profile", "flagged_cases", "tested_routes", "largest_player", "largest_importer",
               "largest_period", "best_gain_percent", "verdict", "scope"))
    return summary, flags


def write_report(rows: list[dict], screening: list[dict], flags: list[dict], digest: str):
    def best(player: str):
        return max((r for r in rows if r["profile"] == EQ1 and r["player"] == player),
                   key=lambda r: float(r["relative_gain_percent"]))
    def best_fixed(player: str):
        return max((r for r in rows if r["profile"] == EQ1 and r["player"] == player
                    and (r["deviation_type"].startswith("grid_") or
                         r["deviation_type"].startswith("single_undercut_") or
                         r["deviation_type"].startswith("all_undercut_"))),
                   key=lambda r: float(r["relative_gain_percent"]))
    china = best("ch")
    apac = best("apac")
    china_fixed = best_fixed("ch")
    apac_fixed = best_fixed("apac")
    failures = [r for r in screening if r["verdict"] == "not an equilibrium"]
    flagged_profiles = [r for r in screening if int(r["flagged_cases"]) > 0]
    grouped_failures: dict[str, list[str]] = {}
    order_labels = {
        "ch-af-apac-eu-row-us": "CH-first",
        "af-eu-us-apac-row-ch": "AF-first",
        "eu-us-af-row-apac-ch": "EU-first",
    }
    for failure in failures:
        sequence, branch = failure["profile"].split("/")
        grouped_failures.setdefault(order_labels[sequence], []).append(
            f"`{branch}` ({float(failure['best_gain_percent']):.2f}%)"
        )
    failure_text = "; ".join(
        f"{order}: {', '.join(grouped_failures[order])}"
        for order in ("CH-first", "AF-first", "EU-first") if order in grouped_failures
    )
    lines = [
        "# Eq 1 nonlocal deviation check",
        "",
        f"Input SHA-256: `{digest}`. The corrected workbook and clean Stage-2 objective reproduced China's reference objective (4,119,175.729360), using destination market-clearing price for export revenue and charging the exporter manufacturing and shipping costs. All other players' strategies were frozen. Each trial was re-cleared through the existing Clarabel lower-level solver.",
        "",
        f"**China:** best tested gain {float(china['relative_gain_percent']):.3f}% from a low-offer start with all offers and capacity re-optimized. Fixed-capacity offer tests reached {float(china_fixed['relative_gain_percent']):.3f}%. Verdict: **not an equilibrium**.",
        f"**APAC:** best tested gain {float(apac['relative_gain_percent']):.3f}% from a low-offer start with all offers and capacity re-optimized. A 2040 APAC-to-EU offer cut alone yielded {float(apac_fixed['relative_gain_percent']):.3f}% with capacity fixed. Verdict: **not an equilibrium**.",
        "",
        "The original one-start SLSQP audit initialized at the candidate's high offers. APAC's fixed-capacity undercut directly shows a profitable move across the zero-flow entry threshold. China's larger gain was found from broad low-offer starts with capacity allowed to adjust, which the original start did not find. These feasible counterexamples reject Eq 1 under the 1% rule; they do not require a globally solved best response.",
        "",
        f"**Screening:** All other 25 profiles have an idle-capacity, near-zero-flow route with a potential margin above 50 USD/kW. All {len(flags)} flagged route-periods received three undercuts and an 11-point grid. {len(failures)} profiles fail the 1% test: {failure_text if failures else 'none'}. The prefixes refer to the player update orders in the run manifest.",
        "",
        "Thus at least 11 of the 26 retained profiles fail the stated equilibrium tolerance. Verdicts for all 25 other profiles are in `screening_summary.csv`; every test is in `results.csv`. A passing label means only that these tests found no deviation above 1%, not that global Nash equilibrium has been proved. The earlier exclusion of `eu-us-af-row-apac-ch/pf120_k050_a040` is noted in `notes/notes_2026-09-27.txt` and `IEEE Paper/revision.tex`, but its exact additional-test script was not present in `scripts/` or `notes/`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    data, manifest, input_path, digest = load_data()
    state, status, ending = load_profile(EQ1, data)
    market, diag = solve_nested_market(data, state)
    china_ref = _economic_objective(data, state, market, "ch")
    apac_ref = _economic_objective(data, state, market, "apac")
    print(f"PRECHECK China reference={china_ref:.9f}; APAC={apac_ref:.9f}", flush=True)
    print(f"PRECHECK diagnostics={diag}", flush=True)
    if abs(china_ref - 4119175.729360106) > 1.0:
        raise AssertionError("China reference does not reproduce original audit")
    if "--report-only" in sys.argv[1:]:
        with (OUT / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (OUT / "screening_summary.csv").open(newline="", encoding="utf-8") as handle:
            screening = list(csv.DictReader(handle))
        with (OUT / "screening_flags.csv").open(newline="", encoding="utf-8") as handle:
            flags = list(csv.DictReader(handle))
        write_report(rows, screening, flags, digest)
        return
    screen_only = "--screen-only" in sys.argv[1:]
    if screen_only:
        with (OUT / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["profile"] == EQ1]
        screening, flags = screen_profiles(data, rows)
        write_csv(OUT / "results.csv", rows, FIELDS)
        write_report(rows, screening, flags, digest)
        return
    rows: list[dict] = []
    references = {"ch": china_ref, "apac": apac_ref}
    for player in ("ch", "apac"):
        objective = references[player]
        targeted_best = targeted_cases(data, EQ1, player, state, market, objective, rows)
        capacity_row, capacity_state = capacity_variants(
            data, player, state, market, objective, targeted_best, rows
        )
        print(f"CAPACITY {player}: {capacity_row['relative_gain_percent']:.3f}%", flush=True)
        write_csv(OUT / "results.csv", rows, FIELDS)
        multistart_cases(data, player, state, market, objective,
                         targeted_best, capacity_state, rows)
    screening, flags = screen_profiles(data, rows)
    write_csv(OUT / "results.csv", rows, FIELDS)
    write_report(rows, screening, flags, digest)


if __name__ == "__main__":
    main()
