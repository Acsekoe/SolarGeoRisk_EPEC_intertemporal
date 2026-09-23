"""Describe outcome ranges across the curated one-start equilibrium candidates.

The analysis is deliberately descriptive.  Candidate profiles are deterministic
computational outcomes selected because they pass a one-start audit; they are not
an independent random sample.  Correlations and matched contrasts therefore
describe associations within the accepted set, not economic causal effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "outputs" / "new_equilibria" / "candidates"
INDEX_PATH = CANDIDATE_ROOT / "candidate_index.csv"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "new_equilibria" / "statistical_analysis_20260921"
)
PENALIZED_PROFILE_ROOT = ROOT / "outputs" / "new_equilibria" / "penalized_profiles"

REGIONS = ("ch", "af", "eu", "us", "apac", "row")
REGION_LABELS = {
    "ch": "China",
    "af": "Africa",
    "eu": "EU",
    "us": "US",
    "apac": "APAC",
    "row": "ROW",
}
MARKET_YEARS = (2025, 2030, 2035, 2040)
CAPACITY_YEARS = MARKET_YEARS
HORIZON_WEIGHTS = {2025: 1.0 / 6.0, 2030: 1.0 / 3.0, 2035: 1.0 / 3.0, 2040: 1.0 / 6.0}
ORDER_LABELS = {
    "ch-af-apac-eu-row-us": "CH-first",
    "af-eu-us-apac-row-ch": "AF-first",
    "eu-us-af-row-apac-ch": "EU-first",
}
MOVEMENT_ORDER_LABELS = {
    "ch-af-apac-eu-row-us": "CH–AF–APAC–EU–ROW–US",
    "af-eu-us-apac-row-ch": "AF–EU–US–APAC–ROW–CH",
    "eu-us-af-row-apac-ch": "EU–US–AF–ROW–APAC–CH",
    "ch-af-eu-us-row-apac": "CH–AF–EU–US–ROW–APAC",
    "ch-row-apac-us-eu-af": "CH–ROW–APAC–US–EU–AF",
    "us-apac-af-row-eu-ch": "US–APAC–AF–ROW–EU–CH",
    "us-row-eu-apac-af-ch": "US–ROW–EU–APAC–AF–CH",
}
ORDER_COLORS = {
    "CH-first": "#1F4E79",
    "AF-first": "#C55A11",
    "EU-first": "#548235",
}
REGION_COLORS = {
    "ch": "#CA6180",
    "eu": "#FEFD99",
    "us": "#FCB7C7",
    "apac": "#B7A6D8",
    "af": "#B8D99E",
    "row": "#9ED3DC",
}
PAPER_REGION_ORDER = ("ch", "eu", "us", "apac", "af", "row")
PAPER_REGION_NAMES = {
    "ch": "China",
    "eu": "Europe",
    "us": "United States",
    "apac": "Asia-Pacific",
    "af": "Africa",
    "row": "Rest of World",
}
PRICE_MARKERS = {0.8: "v", 1.0: "o", 1.2: "s"}
BRANCH_RE = re.compile(r"pf(?P<pf>\d{3})_k(?P<k>\d{3})_a(?P<a>\d{3})$")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rows_to_map(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], float]:
    return {
        tuple(str(row[key]) for key in keys): float(row["value"])
        for row in rows
    }


def parse_branch(branch: str) -> tuple[float, float, float]:
    match = BRANCH_RE.fullmatch(branch)
    if not match:
        raise ValueError(f"Unrecognized branch code: {branch}")
    return tuple(float(match.group(name)) / 100.0 for name in ("pf", "k", "a"))


def short_candidate_code(order: str, branch: str) -> str:
    prefix = {"CH-first": "CH", "AF-first": "AF", "EU-first": "EU"}[order]
    return f"{prefix}-{branch}"


def collect_candidates() -> list[dict]:
    with INDEX_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        index_rows = list(csv.DictReader(handle))

    candidates: list[dict] = []
    for index_row in index_rows:
        sequence = index_row["sequence"]
        branch = index_row["branch"]
        order = ORDER_LABELS[sequence]
        price_factor, capacity_weight, damping = parse_branch(branch)
        branch_root = CANDIDATE_ROOT / sequence / branch
        profile_document = load_json(branch_root / "profile.json")
        audit = load_json(branch_root / "audit_one_start.json")
        profile = profile_document["ending_profile"]
        market = profile["market"]

        capacities = rows_to_map(profile["capacities"], ("player", "time"))
        prices = rows_to_map(market["clearing_prices"], ("region", "time"))
        demand = rows_to_map(market["demand"], ("region", "time"))
        flows = rows_to_map(market["trade_flows"], ("exporter", "importer", "time"))

        candidates.append(
            {
                "candidate": f"{sequence}/{branch}",
                "candidate_code": short_candidate_code(order, branch),
                "sequence": sequence,
                "order": order,
                "branch": branch,
                "origin": index_row["origin"],
                "price_factor": price_factor,
                "capacity_weight": capacity_weight,
                "damping": damping,
                "selected_sweep": int(index_row["selected_sweep"]),
                "max_gain_percent": 100.0 * float(audit["max_relative_gain"]),
                "limiting_player": str(audit["max_gain_player"]),
                "multistart_status": index_row["multistart_status"],
                "capacities": capacities,
                "prices": prices,
                "demand": demand,
                "flows": flows,
            }
        )

    if len(candidates) != 16:
        raise RuntimeError(f"Expected 16 curated candidates, found {len(candidates)}")
    return candidates


def candidate_and_observation_tables(
    candidates: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict] = []
    capacity_rows: list[dict] = []
    price_rows: list[dict] = []
    regional_rows: list[dict] = []
    system_rows: list[dict] = []

    for candidate in candidates:
        metadata = {
            key: candidate[key]
            for key in (
                "candidate",
                "candidate_code",
                "sequence",
                "order",
                "branch",
                "origin",
                "price_factor",
                "capacity_weight",
                "damping",
                "selected_sweep",
                "max_gain_percent",
                "limiting_player",
                "multistart_status",
            )
        }

        for year in CAPACITY_YEARS:
            for region in REGIONS:
                capacity_rows.append(
                    {
                        **metadata,
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        "year": year,
                        "capacity_gw": candidate["capacities"][(region, str(year))],
                    }
                )

        system_by_year: dict[int, dict] = {}
        for year in MARKET_YEARS:
            prices = np.array(
                [candidate["prices"][(region, str(year))] for region in REGIONS],
                dtype=float,
            )
            demands = np.array(
                [candidate["demand"][(region, str(year))] for region in REGIONS],
                dtype=float,
            )
            capacities = np.array(
                [candidate["capacities"][(region, str(year))] for region in REGIONS],
                dtype=float,
            )
            total_demand = float(demands.sum())
            cross_border_trade = sum(
                candidate["flows"].get((exporter, importer, str(year)), 0.0)
                for exporter in REGIONS
                for importer in REGIONS
                if exporter != importer
            )
            system = {
                **metadata,
                "year": year,
                "total_capacity_gw": float(capacities.sum()),
                "total_demand_gw": total_demand,
                "capacity_demand_ratio": float(capacities.sum() / total_demand),
                "mean_price_usd_per_kw": float(prices.mean()),
                "demand_weighted_price_usd_per_kw": float(
                    np.average(prices, weights=demands)
                ),
                "price_dispersion_usd_per_kw": float(prices.std(ddof=0)),
                "cross_border_trade_gw": float(cross_border_trade),
                "cross_border_share": float(cross_border_trade / total_demand),
                "china_capacity_share": float(
                    candidate["capacities"][("ch", str(year))] / capacities.sum()
                ),
                "apac_capacity_share": float(
                    candidate["capacities"][("apac", str(year))] / capacities.sum()
                ),
            }
            system_rows.append(system)
            system_by_year[year] = system

            for region in REGIONS:
                imports = sum(
                    candidate["flows"].get((exporter, region, str(year)), 0.0)
                    for exporter in REGIONS
                    if exporter != region
                )
                exports = sum(
                    candidate["flows"].get((region, importer, str(year)), 0.0)
                    for importer in REGIONS
                    if importer != region
                )
                domestic = candidate["flows"].get((region, region, str(year)), 0.0)
                regional = {
                    **metadata,
                    "region": region,
                    "region_label": REGION_LABELS[region],
                    "year": year,
                    "capacity_gw": candidate["capacities"][(region, str(year))],
                    "price_usd_per_kw": candidate["prices"][(region, str(year))],
                    "demand_gw": candidate["demand"][(region, str(year))],
                    "domestic_output_gw": domestic,
                    "imports_gw": imports,
                    "exports_gw": exports,
                    "import_share": imports / candidate["demand"][(region, str(year))],
                }
                regional_rows.append(regional)
                price_rows.append(
                    {
                        **metadata,
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        "year": year,
                        "price_usd_per_kw": candidate["prices"][(region, str(year))],
                    }
                )

        metrics = {**metadata}
        for year in CAPACITY_YEARS:
            metrics[f"total_capacity_{year}_gw"] = sum(
                candidate["capacities"][(region, str(year))] for region in REGIONS
            )
        for year in MARKET_YEARS:
            system = system_by_year[year]
            metrics[f"mean_price_{year}_usd_per_kw"] = system[
                "mean_price_usd_per_kw"
            ]
            metrics[f"demand_weighted_price_{year}_usd_per_kw"] = system[
                "demand_weighted_price_usd_per_kw"
            ]
        end = system_by_year[2040]
        metrics.update(
            {
                "capacity_demand_ratio_2040": end["capacity_demand_ratio"],
                "price_dispersion_2040_usd_per_kw": end[
                    "price_dispersion_usd_per_kw"
                ],
                "cross_border_trade_2040_gw": end["cross_border_trade_gw"],
                "cross_border_share_2040": end["cross_border_share"],
                "china_capacity_share_2040": end["china_capacity_share"],
                "apac_capacity_share_2040": end["apac_capacity_share"],
            }
        )
        candidate_rows.append(metrics)

    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(capacity_rows),
        pd.DataFrame(price_rows),
        pd.DataFrame(regional_rows),
        pd.DataFrame(system_rows),
    )


def describe_groups(
    frame: pd.DataFrame, group_columns: list[str], value_column: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, sort=False):
        values = group[value_column].dropna().astype(float).to_numpy()
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "metric": value_column,
                "n": len(values),
                "mean": mean_value,
                "std": std_value,
                "min": float(np.min(values)),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "max": float(np.max(values)),
                "iqr": float(q3 - q1),
                "range": float(np.max(values) - np.min(values)),
                "cv_percent": 100.0 * std_value / abs(mean_value)
                if abs(mean_value) > 1e-12
                else math.nan,
                "range_percent_of_median": 100.0
                * float(np.max(values) - np.min(values))
                / abs(float(median))
                if abs(float(median)) > 1e-12
                else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_row(
    scope: str,
    x_name: str,
    y_name: str,
    x: Iterable[float],
    y: Iterable[float],
    *,
    year: int = 2040,
    region: str = "system",
) -> dict:
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if len(x_values) < 3 or np.std(x_values) < 1e-12 or np.std(y_values) < 1e-12:
        pearson = spearman = math.nan
    else:
        pearson = float(pearsonr(x_values, y_values).statistic)
        spearman = float(spearmanr(x_values, y_values).statistic)
    strength = "not estimable"
    if math.isfinite(spearman):
        absolute = abs(spearman)
        strength = (
            "strong"
            if absolute >= 0.70
            else "moderate"
            if absolute >= 0.40
            else "weak"
        )
    return {
        "scope": scope,
        "year": year,
        "region": region,
        "x": x_name,
        "y": y_name,
        "n": len(x_values),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "strength_by_abs_spearman": strength,
    }


def build_associations(
    candidate_metrics: pd.DataFrame, regional_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_pairs = [
        (
            "total_capacity_2040_gw",
            "demand_weighted_price_2040_usd_per_kw",
        ),
        ("capacity_demand_ratio_2040", "demand_weighted_price_2040_usd_per_kw"),
        ("total_capacity_2040_gw", "cross_border_trade_2040_gw"),
        ("china_capacity_share_2040", "demand_weighted_price_2040_usd_per_kw"),
        ("apac_capacity_share_2040", "demand_weighted_price_2040_usd_per_kw"),
        ("cross_border_share_2040", "price_dispersion_2040_usd_per_kw"),
        ("max_gain_percent", "demand_weighted_price_2040_usd_per_kw"),
    ]
    rows = [
        correlation_row(
            "candidate-level system outcome",
            x_name,
            y_name,
            candidate_metrics[x_name],
            candidate_metrics[y_name],
        )
        for x_name, y_name in selected_pairs
    ]

    for year in MARKET_YEARS:
        for region in REGIONS:
            subset = regional_metrics[
                (regional_metrics["year"] == year)
                & (regional_metrics["region"] == region)
            ]
            rows.append(
                correlation_row(
                    "candidate-level regional outcome",
                    "capacity_gw",
                    "price_usd_per_kw",
                    subset["capacity_gw"],
                    subset["price_usd_per_kw"],
                    year=year,
                    region=REGION_LABELS[region],
                )
            )
            rows.append(
                correlation_row(
                    "candidate-level regional outcome",
                    "import_share",
                    "price_usd_per_kw",
                    subset["import_share"],
                    subset["price_usd_per_kw"],
                    year=year,
                    region=REGION_LABELS[region],
                )
            )

    matrix_columns = [
        "total_capacity_2040_gw",
        "demand_weighted_price_2040_usd_per_kw",
        "capacity_demand_ratio_2040",
        "price_dispersion_2040_usd_per_kw",
        "cross_border_trade_2040_gw",
        "cross_border_share_2040",
        "china_capacity_share_2040",
        "apac_capacity_share_2040",
        "max_gain_percent",
    ]
    correlation_matrix = candidate_metrics[matrix_columns].corr(method="spearman")
    correlation_matrix.index.name = "metric"
    return pd.DataFrame(rows), correlation_matrix.reset_index()


def pair_candidates(
    candidates: pd.DataFrame,
    fixed_columns: list[str],
    baseline_column: str,
    baseline_value: object,
    alternative_value: object,
    contrast_label: str,
) -> list[tuple[pd.Series, pd.Series, str]]:
    pairs: list[tuple[pd.Series, pd.Series, str]] = []
    for keys, group in candidates.groupby(fixed_columns, dropna=False):
        baseline = group[group[baseline_column] == baseline_value]
        alternative = group[group[baseline_column] == alternative_value]
        if len(baseline) == 1 and len(alternative) == 1:
            pairs.append((baseline.iloc[0], alternative.iloc[0], contrast_label))
    return pairs


def matched_contrasts(candidate_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs: list[tuple[pd.Series, pd.Series, str]] = []
    pairs.extend(
        pair_candidates(
            candidate_metrics,
            ["order", "capacity_weight", "damping"],
            "price_factor",
            1.0,
            1.2,
            "PF 1.20 minus PF 1.00",
        )
    )
    pairs.extend(
        pair_candidates(
            candidate_metrics,
            ["order", "capacity_weight", "damping"],
            "price_factor",
            1.0,
            0.8,
            "PF 0.80 minus PF 1.00",
        )
    )
    pairs.extend(
        pair_candidates(
            candidate_metrics,
            ["order", "price_factor", "damping"],
            "capacity_weight",
            0.5,
            1.0,
            "Capacity weight 1.00 minus 0.50",
        )
    )
    pairs.extend(
        pair_candidates(
            candidate_metrics,
            ["order", "price_factor", "capacity_weight"],
            "damping",
            0.3,
            0.4,
            "Damping 0.40 minus 0.30",
        )
    )
    pairs.extend(
        pair_candidates(
            candidate_metrics,
            ["branch"],
            "order",
            "CH-first",
            "AF-first",
            "AF-first minus CH-first",
        )
    )
    pairs.extend(
        pair_candidates(
            candidate_metrics,
            ["branch"],
            "order",
            "CH-first",
            "EU-first",
            "EU-first minus CH-first",
        )
    )

    metrics = [
        "total_capacity_2040_gw",
        "demand_weighted_price_2040_usd_per_kw",
        "cross_border_trade_2040_gw",
    ]
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for baseline, alternative, label in pairs:
        pair_id = (label, baseline["candidate"], alternative["candidate"])
        if pair_id in seen:
            continue
        seen.add(pair_id)
        for metric in metrics:
            base_value = float(baseline[metric])
            alt_value = float(alternative[metric])
            rows.append(
                {
                    "contrast": label,
                    "baseline_candidate": baseline["candidate_code"],
                    "alternative_candidate": alternative["candidate_code"],
                    "metric": metric,
                    "baseline_value": base_value,
                    "alternative_value": alt_value,
                    "absolute_difference": alt_value - base_value,
                    "percent_difference": 100.0 * (alt_value / base_value - 1.0)
                    if abs(base_value) > 1e-12
                    else math.nan,
                }
            )
    raw = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for (contrast, metric), group in raw.groupby(["contrast", "metric"], sort=False):
        values = group["percent_difference"].astype(float).to_numpy()
        summary_rows.append(
            {
                "contrast": contrast,
                "metric": metric,
                "n_pairs": len(values),
                "mean_percent_difference": float(np.mean(values)),
                "median_percent_difference": float(np.median(values)),
                "min_percent_difference": float(np.min(values)),
                "max_percent_difference": float(np.max(values)),
            }
        )
    return raw, pd.DataFrame(summary_rows)


def read_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def collect_branch_results() -> pd.DataFrame:
    rows: list[dict] = []

    ch_manifest = load_json(
        ROOT
        / "outputs"
        / "new_equilibria"
        / "existing_equilibria"
        / "ch-af-apac-eu-row-us"
        / "source_metadata"
        / "factorial"
        / "manifest.json"
    )
    for result in ch_manifest["results"]:
        if result["sequence"] != "ch-af-apac-eu-row-us":
            continue
        rows.append(
            {
                "source_experiment": "PF100/PF120 clean three-anchor design",
                "sequence": result["sequence"],
                "branch": result["branch"],
                "status": result["status"],
                "selected_sweep": int(result["selected_sweep"]),
                "max_gain_percent": 100.0
                * float(result["one_start_max_relative_gain"]),
                "limiting_player": result["one_start_max_gain_player"],
                "all_six_solves_successful": bool(
                    result["all_six_solves_successful"]
                ),
                "accepted": bool(result["local_one_percent_equilibrium"]),
            }
        )

    new_summary = (
        ROOT
        / "outputs"
        / "new_equilibria"
        / "new_profile_workflow_20260920_143353"
        / "results_summary.csv"
    )
    with new_summary.open("r", newline="", encoding="utf-8-sig") as handle:
        for result in csv.DictReader(handle):
            if result["record_type"] != "basin_grid":
                continue
            rows.append(
                {
                    "source_experiment": "PF100/PF120 clean three-anchor design",
                    "sequence": result["sequence"],
                    "branch": result["branch"],
                    "status": result["status"],
                    "selected_sweep": int(result["selected_sweep"]),
                    "max_gain_percent": 100.0 * float(result["max_relative_gain"]),
                    "limiting_player": result["max_gain_player"],
                    "all_six_solves_successful": read_bool(
                        result["all_six_solves_successful"]
                    ),
                    "accepted": read_bool(result["local_one_percent_equilibrium"]),
                }
            )

    pf080_summary = (
        ROOT
        / "outputs"
        / "pf080_low_start_20260920_171507"
        / "results_summary.csv"
    )
    with pf080_summary.open("r", newline="", encoding="utf-8-sig") as handle:
        for result in csv.DictReader(handle):
            rows.append(
                {
                    "source_experiment": "PF080 supplementary design",
                    "sequence": result["sequence"],
                    "branch": result["branch"],
                    "status": result["status"],
                    "selected_sweep": int(result["selected_sweep"]),
                    "max_gain_percent": float(result["max_gain_percent"]),
                    "limiting_player": result["max_gain_player"],
                    "all_six_solves_successful": read_bool(
                        result["all_six_solves_successful"]
                    ),
                    "accepted": read_bool(result["local_one_percent_equilibrium"]),
                }
            )

    for row in rows:
        pf, weight, damping = parse_branch(row["branch"])
        row.update(
            {
                "order": ORDER_LABELS[row["sequence"]],
                "price_factor": pf,
                "capacity_weight": weight,
                "damping": damping,
            }
        )

    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(["sequence", "branch"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate clean-design branch results:\n"
            + frame.loc[duplicates, ["sequence", "branch"]].to_string(index=False)
        )
    if len(frame) != 36:
        raise RuntimeError(f"Expected 36 branch results, found {len(frame)}")
    if not bool(frame["all_six_solves_successful"].all()):
        raise RuntimeError("At least one selected branch audit was not solver-clean")
    return frame


def collect_penalized_movement_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_index = pd.read_csv(PENALIZED_PROFILE_ROOT / "profile_index.csv")
    histories: list[pd.DataFrame] = []
    for _, profile in profile_index.iterrows():
        workbooks = [profile["initial_workbook"]]
        if pd.notna(profile["continuation_workbook"]):
            workbooks.append(profile["continuation_workbook"])
        for workbook_name in workbooks:
            range_match = re.search(
                r"penalized_sweeps_(\d{3})_(\d{3})\.xlsx$",
                str(workbook_name),
            )
            if range_match is None:
                raise RuntimeError(
                    f"Cannot infer absolute sweep range from {workbook_name}"
                )
            sweep_offset = int(range_match.group(1)) - 1
            workbook_path = PENALIZED_PROFILE_ROOT / str(workbook_name)
            iteration_rows = pd.read_excel(workbook_path, sheet_name="iters")
            history = pd.DataFrame(
                {
                    "sequence": profile["sequence"],
                    "order_label": MOVEMENT_ORDER_LABELS[profile["sequence"]],
                    "sweep": iteration_rows["iter"].astype(int) + sweep_offset,
                    "movement_percent": 100.0 * iteration_rows["r_strat"].astype(float),
                    "raw_best_response_move_percent": 100.0
                    * iteration_rows["r_raw_br"].astype(float),
                    "all_solves_acceptable": iteration_rows[
                        "all_solves_acceptable"
                    ].map(
                        lambda value: bool(value)
                        if isinstance(value, (bool, np.bool_))
                        else read_bool(value)
                    ),
                    "stable_count": iteration_rows["stable_count"].astype(int),
                    "final_status": profile["final_status"],
                    "solve_quality": profile["solve_quality"],
                }
            )
            if pd.notna(profile["first_clean_movement_convergence_sweep"]):
                stopping_sweep = int(
                    profile["first_clean_movement_convergence_sweep"]
                )
                history = history[history["sweep"] <= stopping_sweep]
            histories.append(history)
    movement_history = pd.concat(histories, ignore_index=True)
    movement_history = movement_history.sort_values(["sequence", "sweep"])
    return movement_history, profile_index


def collect_direct_anchor_audits(profile_index: pd.DataFrame) -> pd.DataFrame:
    audit_root = (
        ROOT
        / "outputs"
        / "new_equilibria"
        / "new_profile_workflow_20260920_143353"
        / "direct_audits"
    )
    documented = {
        "ch-af-apac-eu-row-us": {
            "max_gain_percent": 16.037,
            "limiting_player": "apac",
            "audit_depth": "three-start maximum",
            "movement_percent": 0.497676,
            "anchor_sweep": 26,
            "source": "workflow/summary_20260920_153447.md",
        }
    }
    for sequence in ("af-eu-us-apac-row-ch", "eu-us-af-row-apac-ch"):
        audit_path = audit_root / sequence / "audit_one_start.json"
        audit = load_json(audit_path)
        documented[sequence] = {
            "max_gain_percent": 100.0 * float(audit["max_relative_gain"]),
            "limiting_player": audit["max_gain_player"],
            "audit_depth": "one-start maximum",
            "anchor_sweep": 40,
            "source": str(audit_path.relative_to(ROOT)),
        }

    rows: list[dict] = []
    for sequence, audit in documented.items():
        profile = profile_index[profile_index["sequence"] == sequence].iloc[0]
        rows.append(
            {
                "sequence": sequence,
                "order": ORDER_LABELS[sequence],
                "movement_percent": float(
                    audit.get("movement_percent", profile["final_movement_percent"])
                ),
                "first_clean_movement_convergence_sweep": int(
                    profile["first_clean_movement_convergence_sweep"]
                ),
                **audit,
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - margin, center + margin


def pass_rate_summary(branch_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    dimensions = [
        ("price_factor", ["price_factor"]),
        ("update_order", ["order"]),
        ("capacity_weight", ["capacity_weight"]),
        ("damping", ["damping"]),
        ("order_x_price_factor", ["order", "price_factor"]),
    ]
    for dimension, columns in dimensions:
        for keys, group in branch_results.groupby(columns, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            successes = int(group["accepted"].sum())
            total = len(group)
            low, high = wilson_interval(successes, total)
            rows.append(
                {
                    "dimension": dimension,
                    "level": " / ".join(str(value) for value in keys),
                    "n_branches": total,
                    "n_accepted": successes,
                    "pass_rate": successes / total,
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                }
            )
    return pd.DataFrame(rows)


def silhouette_score_from_distance(distance: np.ndarray, labels: np.ndarray) -> float:
    values: list[float] = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        a = float(distance[index, same].mean()) if same.any() else 0.0
        b_values = [
            float(distance[index, labels == other].mean())
            for other in np.unique(labels)
            if other != label
        ]
        b = min(b_values)
        values.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(values))


def pca_and_clusters(
    candidates: list[dict], candidate_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    feature_names: list[str] = []
    rows: list[list[float]] = []
    for candidate_index, candidate in enumerate(candidates):
        values: list[float] = []
        if candidate_index == 0:
            for year in CAPACITY_YEARS:
                for region in REGIONS:
                    feature_names.append(f"capacity_{REGION_LABELS[region]}_{year}")
            for year in MARKET_YEARS:
                for region in REGIONS:
                    feature_names.append(f"price_{REGION_LABELS[region]}_{year}")
        for year in CAPACITY_YEARS:
            for region in REGIONS:
                values.append(candidate["capacities"][(region, str(year))])
        for year in MARKET_YEARS:
            for region in REGIONS:
                values.append(candidate["prices"][(region, str(year))])
        rows.append(values)

    matrix = np.asarray(rows, dtype=float)
    std = matrix.std(axis=0, ddof=0)
    keep = std > 1e-10
    matrix = matrix[:, keep]
    retained_names = [name for name, retained in zip(feature_names, keep) if retained]
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
    u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    scores = u * singular_values
    explained = singular_values**2 / np.sum(singular_values**2)

    hierarchy = linkage(standardized, method="ward")
    distance = squareform(pdist(standardized))
    silhouette_rows: list[dict] = []
    label_by_k: dict[int, np.ndarray] = {}
    for k in range(2, 6):
        labels = fcluster(hierarchy, k, criterion="maxclust")
        label_by_k[k] = labels
        silhouette_rows.append(
            {"n_clusters": k, "silhouette_score": silhouette_score_from_distance(distance, labels)}
        )
    silhouette_frame = pd.DataFrame(silhouette_rows)
    best_k = int(
        silhouette_frame.sort_values(
            ["silhouette_score", "n_clusters"], ascending=[False, True]
        ).iloc[0]["n_clusters"]
    )
    raw_labels = label_by_k[best_k]

    total_capacity = candidate_metrics.set_index("candidate")["total_capacity_2040_gw"]
    cluster_order = sorted(
        np.unique(raw_labels),
        key=lambda label: float(
            np.mean(
                [
                    total_capacity[candidate["candidate"]]
                    for candidate, value in zip(candidates, raw_labels)
                    if value == label
                ]
            )
        ),
    )
    label_map = {raw: number for number, raw in enumerate(cluster_order, start=1)}
    labels = np.array([label_map[value] for value in raw_labels])

    assignment_rows = []
    for index, candidate in enumerate(candidates):
        assignment_rows.append(
            {
                "candidate": candidate["candidate"],
                "candidate_code": candidate["candidate_code"],
                "figure_code": f"C{index + 1:02d}",
                "order": candidate["order"],
                "branch": candidate["branch"],
                "family": f"Family {labels[index]}",
                "pc1_score": float(scores[index, 0]),
                "pc2_score": float(scores[index, 1]),
            }
        )

    loading_rows = []
    for component_index in range(2):
        for feature, loading in zip(retained_names, vt[component_index]):
            loading_rows.append(
                {
                    "component": f"PC{component_index + 1}",
                    "feature": feature,
                    "loading": float(loading),
                    "absolute_loading": abs(float(loading)),
                }
            )
    pca_info = {
        "best_k": best_k,
        "pc1_explained_percent": 100.0 * float(explained[0]),
        "pc2_explained_percent": 100.0 * float(explained[1]),
        "pc1_pc2_explained_percent": 100.0 * float(explained[0] + explained[1]),
        "best_silhouette": float(
            silhouette_frame.loc[
                silhouette_frame["n_clusters"] == best_k, "silhouette_score"
            ].iloc[0]
        ),
    }
    return (
        pd.DataFrame(assignment_rows),
        pd.DataFrame(loading_rows),
        silhouette_frame,
        pca_info,
    )


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def draw_boxplots(
    frame: pd.DataFrame,
    value_column: str,
    years: tuple[int, ...],
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.3), sharex=True)
    rng = np.random.default_rng(20260921)
    for axis, year in zip(axes.flat, years):
        data = [
            frame[(frame["year"] == year) & (frame["region"] == region)][
                value_column
            ].to_numpy()
            for region in REGIONS
        ]
        box = axis.boxplot(
            data,
            tick_labels=[REGION_LABELS[region] for region in REGIONS],
            patch_artist=True,
            widths=0.62,
            whis=(0, 100),
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.5},
            whiskerprops={"color": "#666666"},
            capprops={"color": "#666666"},
        )
        for patch, region in zip(box["boxes"], REGIONS):
            patch.set_facecolor(REGION_COLORS[region])
            patch.set_alpha(0.65)
            patch.set_edgecolor("#505050")
        for position, values, region in zip(range(1, 7), data, REGIONS):
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=10,
                color="#2F2F2F",
                alpha=0.52,
                linewidths=0,
                zorder=3,
            )
        axis.set_title(str(year))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel(ylabel)
    figure.suptitle(title)
    figure.text(
        0.5,
        0.01,
        "Whiskers show the observed minimum and maximum; charcoal dots are individual candidates and horizontal jitter is only for visibility.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(figure, output_dir, stem)


def plot_capacity_price_scatter(
    candidate_metrics: pd.DataFrame, associations: pd.DataFrame, output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 6.1))
    for _, row in candidate_metrics.iterrows():
        axis.scatter(
            row["total_capacity_2040_gw"],
            row["demand_weighted_price_2040_usd_per_kw"],
            s=58,
            color=ORDER_COLORS[row["order"]],
            marker=PRICE_MARKERS[row["price_factor"]],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )

    x = candidate_metrics["total_capacity_2040_gw"].to_numpy()
    y = candidate_metrics["demand_weighted_price_2040_usd_per_kw"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    axis.plot(grid, intercept + slope * grid, color="#404040", linestyle="--", linewidth=1.1)

    corr = associations[
        (associations["x"] == "total_capacity_2040_gw")
        & (associations["y"] == "demand_weighted_price_2040_usd_per_kw")
    ].iloc[0]
    axis.text(
        0.43,
        0.98,
        f"Spearman ρ = {corr['spearman_rho']:.2f}\nPearson r = {corr['pearson_r']:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BFBFBF"},
    )

    for index in {
        int(candidate_metrics["total_capacity_2040_gw"].idxmin()),
        int(candidate_metrics["total_capacity_2040_gw"].idxmax()),
        int(candidate_metrics["demand_weighted_price_2040_usd_per_kw"].idxmin()),
        int(candidate_metrics["demand_weighted_price_2040_usd_per_kw"].idxmax()),
    }:
        row = candidate_metrics.loc[index]
        axis.annotate(
            row["candidate_code"],
            (row["total_capacity_2040_gw"], row["demand_weighted_price_2040_usd_per_kw"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    order_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            label=order,
            markersize=7,
        )
        for order, color in ORDER_COLORS.items()
    ]
    factor_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#505050",
            markerfacecolor="white",
            label=f"PF {factor:.2f}",
            markersize=7,
            linestyle="none",
        )
        for factor, marker in PRICE_MARKERS.items()
    ]
    first_legend = axis.legend(handles=order_handles, title="Update order", loc="upper right")
    axis.add_artist(first_legend)
    axis.legend(handles=factor_handles, title="Price initialization", loc="lower left")
    axis.set_xlabel("Total manufacturing capacity in 2040 (GW)")
    axis.set_ylabel("Demand-weighted clearing price in 2040 (USD/kW)")
    axis.set_title("Capacity and price across accepted candidates")
    axis.grid(color="#E6E6E6", linewidth=0.6)
    figure.tight_layout()
    save_figure(figure, output_dir, "capacity_price_association_2040")


def add_horizon_capacity_price_indicators(
    candidate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    result = candidate_metrics.copy()
    result["average_total_capacity_2025_2040_gw"] = sum(
        HORIZON_WEIGHTS[year] * result[f"total_capacity_{year}_gw"]
        for year in CAPACITY_YEARS
    )
    result["average_demand_weighted_price_2025_2040_usd_per_kw"] = sum(
        HORIZON_WEIGHTS[year]
        * result[f"demand_weighted_price_{year}_usd_per_kw"]
        for year in MARKET_YEARS
    )
    return result


def plot_horizon_capacity_price_equilibria(
    candidate_metrics: pd.DataFrame, output_dir: Path
) -> None:
    x_column = "average_total_capacity_2025_2040_gw"
    y_column = "average_demand_weighted_price_2025_2040_usd_per_kw"

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        axis.scatter(
            candidate_metrics[x_column],
            candidate_metrics[y_column],
            s=62,
            marker="x",
            color="#A83232",
            linewidth=1.8,
            label="Converged equilibria",
            zorder=3,
        )

        axis.set_xlabel("Average total manufacturing capacity [GW]", fontsize=15)
        axis.set_ylabel("Demand-weighted price [$/kW]", fontsize=15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=15)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.34),
            frameon=True,
            framealpha=0.9,
            fontsize=11.5,
            handletextpad=0.45,
            borderpad=0.45,
        )
        figure.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.31)
        save_figure(figure, output_dir, "capacity_price_pathway_equilibria")


def plot_pass_heatmap(branch_results: pd.DataFrame, output_dir: Path) -> None:
    orders = ("CH-first", "AF-first", "EU-first")
    price_factors = (0.8, 1.0, 1.2)
    values = np.zeros((len(orders), len(price_factors)))
    labels: list[list[str]] = []
    for order_index, order in enumerate(orders):
        label_row: list[str] = []
        for factor_index, factor in enumerate(price_factors):
            subset = branch_results[
                (branch_results["order"] == order)
                & (branch_results["price_factor"] == factor)
            ]
            accepted = int(subset["accepted"].sum())
            total = len(subset)
            values[order_index, factor_index] = accepted / total
            label_row.append(f"{accepted}/{total}\n{100 * accepted / total:.0f}%")
        labels.append(label_row)

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row_index in range(len(orders)):
        for column_index in range(len(price_factors)):
            color = "white" if values[row_index, column_index] >= 0.55 else "#202020"
            axis.text(
                column_index,
                row_index,
                labels[row_index][column_index],
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    axis.set_xticks(range(len(price_factors)), [f"PF {value:.2f}" for value in price_factors])
    axis.set_yticks(range(len(orders)), orders)
    axis.set_xlabel("Bilateral-offer initialization factor")
    axis.set_ylabel("Update-order anchor")
    axis.set_title("Share of branches passing the one-start 1% audit")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Pass rate")
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    figure.tight_layout()
    save_figure(figure, output_dir, "search_pass_rate_heatmap")


def plot_algorithm_convergence(
    movement_history: pd.DataFrame,
    direct_audits: pd.DataFrame,
    branch_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(16.0, 8.8))
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=(1.2, 1.0),
        width_ratios=(1.25, 1.0, 1.0, 1.0),
        hspace=0.43,
        wspace=0.34,
    )
    movement_axis = figure.add_subplot(grid[0, :])
    audit_axis = figure.add_subplot(grid[1, 0])
    branch_axes = [figure.add_subplot(grid[1, index]) for index in range(1, 4)]

    movement_colors = {
        "ch-af-apac-eu-row-us": "#1F4E79",
        "af-eu-us-apac-row-ch": "#C55A11",
        "eu-us-af-row-apac-ch": "#548235",
        "ch-af-eu-us-row-apac": "#8064A2",
        "ch-row-apac-us-eu-af": "#C00000",
        "us-apac-af-row-eu-ch": "#8C6D5A",
        "us-row-eu-apac-af-ch": "#595959",
    }
    endpoint_offsets = {
        "ch-af-apac-eu-row-us": 8,
        "af-eu-us-apac-row-ch": -5,
        "eu-us-af-row-apac-ch": 16,
        "ch-af-eu-us-row-apac": 5,
        "ch-row-apac-us-eu-af": 5,
        "us-apac-af-row-eu-ch": 9,
        "us-row-eu-apac-af-ch": -11,
    }
    for sequence in MOVEMENT_ORDER_LABELS:
        path = movement_history[movement_history["sequence"] == sequence]
        converged = str(path["final_status"].iloc[0]).startswith("converged")
        color = movement_colors[sequence]
        movement_axis.plot(
            path["sweep"],
            path["movement_percent"],
            color=color,
            linewidth=2.0 if converged else 1.35,
            linestyle="-" if converged else "--",
            alpha=0.95 if converged else 0.82,
            zorder=2,
        )
        convergence_rows = path[path["stable_count"] == 3]
        if not convergence_rows.empty:
            first_pass = convergence_rows.iloc[0]
            movement_axis.scatter(
                first_pass["sweep"],
                first_pass["movement_percent"],
                s=48,
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=5,
            )
        flagged = path[~path["all_solves_acceptable"]]
        if not flagged.empty:
            movement_axis.scatter(
                flagged["sweep"],
                flagged["movement_percent"],
                marker="x",
                s=25,
                color="#C00000",
                linewidth=1.0,
                zorder=6,
            )
        endpoint = path.iloc[-1]
        movement_axis.annotate(
            f"{MOVEMENT_ORDER_LABELS[sequence]}  {endpoint['movement_percent']:.2f}%",
            (endpoint["sweep"], endpoint["movement_percent"]),
            xytext=(7, endpoint_offsets[sequence]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=color,
        )

    movement_axis.axhspan(0.1, 1.0, color="#E2F0D9", alpha=0.55, zorder=0)
    movement_axis.axhline(1.0, color="#548235", linestyle=":", linewidth=1.3)
    movement_axis.axvline(30, color="#A6A6A6", linestyle=":", linewidth=1.0)
    movement_axis.text(30.3, 120, "six-order restart after 30", fontsize=8, color="#666666")
    movement_axis.text(1.2, 0.78, "movement threshold", fontsize=8, color="#3F6B2A")
    movement_axis.set_yscale("log")
    movement_axis.set_xlim(1, 50)
    movement_axis.set_ylim(0.1, 300)
    movement_axis.set_xlabel("Penalized Gauss–Seidel sweep")
    movement_axis.set_ylabel("Maximum damped strategy movement (%)")
    movement_axis.set_title(
        "A  Penalized and damped basin discovery: three of seven reach the 1% stopping rule",
        loc="left",
        fontweight="bold",
    )
    movement_axis.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.7)
    movement_axis.legend(
        handles=[
            Line2D([0], [0], color="#404040", linewidth=2.0, label="Movement-converged"),
            Line2D([0], [0], color="#707070", linewidth=1.4, linestyle="--", label="Did not converge"),
            Line2D([0], [0], marker="x", color="#C00000", linestyle="none", label="Solver-quality flag"),
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#404040", linestyle="none", label="First clean three-sweep pass"),
        ],
        loc="upper left",
        ncol=2,
        fontsize=8,
        frameon=True,
    )

    direct_colors = {
        "CH-first": "#1F4E79",
        "AF-first": "#C55A11",
        "EU-first": "#548235",
    }
    audit_offsets = {"CH-first": (6, 2), "AF-first": (6, 8), "EU-first": (6, -10)}
    for _, row in direct_audits.iterrows():
        marker = "D" if row["order"] == "CH-first" else "o"
        audit_axis.scatter(
            row["movement_percent"],
            row["max_gain_percent"],
            s=64,
            marker=marker,
            color=direct_colors[row["order"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        label = f"{row['order']}{'†' if row['order'] == 'CH-first' else ''}\n{row['max_gain_percent']:.1f}% gain"
        audit_axis.annotate(
            label,
            (row["movement_percent"], row["max_gain_percent"]),
            xytext=audit_offsets[row["order"]],
            textcoords="offset points",
            fontsize=8,
            color=direct_colors[row["order"]],
            va="center",
        )
    audit_axis.axvspan(0.14, 1.0, color="#E2F0D9", alpha=0.45, zorder=0)
    audit_axis.axhspan(1.0, 30.0, color="#FCE4D6", alpha=0.45, zorder=0)
    audit_axis.axvline(1.0, color="#548235", linestyle=":", linewidth=1.2)
    audit_axis.axhline(1.0, color="#C00000", linestyle="--", linewidth=1.2)
    audit_axis.set_xscale("log")
    audit_axis.set_yscale("log")
    audit_axis.set_xlim(0.14, 1.25)
    audit_axis.set_ylim(0.7, 25)
    audit_axis.set_xticks([0.2, 0.5, 1.0], ["0.2", "0.5", "1"])
    audit_axis.set_yticks([1, 3, 10, 20], ["1", "3", "10", "20"])
    audit_axis.minorticks_off()
    audit_axis.set_xlabel("Anchor movement residual (%)")
    audit_axis.set_ylabel("Independent maximum gain (%)")
    audit_axis.set_title(
        "B  Stable movement did not imply equilibrium",
        loc="left",
        fontweight="bold",
    )
    audit_axis.grid(color="#E0E0E0", linewidth=0.6)

    price_colors = {0.8: "#7030A0", 1.0: "#4472C4", 1.2: "#ED7D31"}
    branch_orders = ("CH-first", "AF-first", "EU-first")
    panel_letters = ("C", "D", "E")
    for axis, order, panel_letter in zip(branch_axes, branch_orders, panel_letters):
        subset = branch_results[branch_results["order"] == order].copy()
        for _, row in subset.iterrows():
            x_offset = (-0.10 if row["capacity_weight"] == 0.5 else 0.10) + (
                -0.035 if row["damping"] == 0.3 else 0.035
            )
            color = price_colors[float(row["price_factor"])]
            axis.scatter(
                row["selected_sweep"] + x_offset,
                row["max_gain_percent"],
                s=48,
                facecolor=color if row["accepted"] else "white",
                edgecolor=color,
                linewidth=1.35,
                zorder=4,
            )
        accepted = int(subset["accepted"].sum())
        axis.axhspan(0.45, 1.0, color="#E2F0D9", alpha=0.50, zorder=0)
        axis.axhline(1.0, color="#C00000", linestyle="--", linewidth=1.15)
        axis.set_yscale("log")
        axis.set_xlim(0.5, 15.5)
        axis.set_ylim(0.45, 6.2)
        axis.set_xticks([1, 5, 10, 15])
        axis.set_yticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
        axis.minorticks_off()
        axis.set_xlabel("Selected sweep")
        axis.set_title(
            f"{panel_letter}  {order}: {accepted}/12 pass",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    branch_axes[0].set_ylabel("Best audited maximum gain (%)")
    branch_axes[-1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, label=f"PF {factor:.2f}")
            for factor, color in price_colors.items()
        ]
        + [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#595959", markeredgecolor="#595959", label="Accepted"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#595959", label="No pass"),
        ],
        loc="upper right",
        fontsize=7.5,
        frameon=True,
    )

    figure.suptitle(
        "Algorithm convergence and equilibrium verification",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.947,
        "Penalized movement locates stable basins; frozen-profile zero-proximal regret determines acceptance.",
        ha="center",
        fontsize=10,
        color="#505050",
    )
    figure.text(
        0.5,
        0.025,
        "Movement pass: residual ≤1% for three consecutive solver-clean sweeps. Audit pass: all six best responses succeed and maximum gain ≤1%. "
        "† CH-first direct point is the documented three-start maximum; AF/EU direct points are one-start audits. Branch panels show the first passing sweep, or the best audited sweep if none passed.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        wrap=True,
    )
    figure.subplots_adjust(left=0.065, right=0.97, top=0.90, bottom=0.13)
    save_figure(figure, output_dir, "algorithm_convergence_diagnostics")


def plot_matched_contrasts(raw: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("total_capacity_2040_gw", "Total capacity 2040"),
        ("demand_weighted_price_2040_usd_per_kw", "Demand-weighted price 2040"),
    ]
    contrast_order = list(dict.fromkeys(raw["contrast"].tolist()))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True)
    rng = np.random.default_rng(20260921)
    for axis, (metric, title) in zip(axes, metrics):
        subset = raw[raw["metric"] == metric]
        for position, contrast in enumerate(contrast_order, start=1):
            values = subset[subset["contrast"] == contrast]["percent_difference"].to_numpy()
            if len(values) == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=30,
                color="#4472C4",
                alpha=0.70,
                edgecolor="white",
                linewidth=0.4,
            )
            axis.plot(
                [position - 0.18, position + 0.18],
                [np.median(values), np.median(values)],
                color="#C00000",
                linewidth=2.0,
            )
        axis.axhline(0, color="#404040", linewidth=0.8)
        axis.set_xticks(range(1, len(contrast_order) + 1), contrast_order, rotation=35, ha="right")
        axis.set_ylabel("Difference from matched baseline (%)")
        axis.set_title(title)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.suptitle("Matched contrasts within the accepted candidate set")
    figure.text(
        0.5,
        0.01,
        "Red line: median. Contrasts condition on both compared configurations being accepted.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    save_figure(figure, output_dir, "matched_candidate_contrasts")


def plot_regional_boxplots_paper_style(
    frame: pd.DataFrame,
    value_column: str,
    years: tuple[int, ...],
    ylabel: str,
    output_dir: Path,
    stem: str,
) -> None:
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        figure, axes = plt.subplots(3, 2, figsize=(7.0, 8.4))
        rng = np.random.default_rng(20260921)
        for index, (axis, region) in enumerate(zip(axes.flat, PAPER_REGION_ORDER)):
            data = [
                frame[(frame["region"] == region) & (frame["year"] == year)][
                    value_column
                ].to_numpy()
                for year in years
            ]
            box = axis.boxplot(
                data,
                tick_labels=[str(year) for year in years],
                patch_artist=True,
                widths=0.58,
                whis=(0, 100),
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 1.6},
                whiskerprops={"color": "#6E6E6E", "linewidth": 1.0},
                capprops={"color": "#6E6E6E", "linewidth": 1.0},
                boxprops={"edgecolor": "#6E6E6E", "linewidth": 1.0},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(REGION_COLORS[region])
                patch.set_alpha(0.78)
            for position, values in enumerate(data, start=1):
                jitter = rng.uniform(-0.07, 0.07, size=len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=10,
                    color="#222222",
                    alpha=0.45,
                    linewidths=0,
                    zorder=3,
                )
            axis.set_title(
                PAPER_REGION_NAMES[region], fontsize=18, fontweight="normal"
            )
            if index % 2 == 0:
                axis.set_ylabel(ylabel, fontsize=18)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.tick_params(axis="both", labelsize=15)

        figure.subplots_adjust(
            left=0.13, right=0.98, top=0.95, bottom=0.07, wspace=0.34, hspace=0.50
        )
        save_figure(figure, output_dir, stem)


def plot_capacity_by_region(capacity_rows: pd.DataFrame, output_dir: Path) -> None:
    plot_regional_boxplots_paper_style(
        capacity_rows,
        "capacity_gw",
        CAPACITY_YEARS,
        "Capacity [GW]",
        output_dir,
        "boxplots_capacity_by_region",
    )


def plot_prices_by_region(price_rows: pd.DataFrame, output_dir: Path) -> None:
    plot_regional_boxplots_paper_style(
        price_rows,
        "price_usd_per_kw",
        MARKET_YEARS,
        "Price [$/kW]",
        output_dir,
        "boxplots_prices_by_region",
    )


def plot_regional_equilibrium_bands_paper_style(
    frame: pd.DataFrame,
    value_column: str,
    years: tuple[int, ...],
    ylabel: str,
    band_color: str,
    output_dir: Path,
    stem: str,
) -> None:
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        figure, axes = plt.subplots(3, 2, figsize=(7.0, 8.4))
        rng = np.random.default_rng(20260921)
        x = np.arange(len(years), dtype=float)
        for index, (axis, region) in enumerate(zip(axes.flat, PAPER_REGION_ORDER)):
            data = [
                frame[(frame["region"] == region) & (frame["year"] == year)][
                    value_column
                ].to_numpy()
                for year in years
            ]
            matrix = np.column_stack(data)
            observed_minimum = matrix.min(axis=0)
            observed_maximum = matrix.max(axis=0)
            q10, q25, median, q75, q90 = np.quantile(
                matrix, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0
            )
            axis.fill_between(
                x,
                observed_minimum,
                observed_maximum,
                color=band_color,
                alpha=0.20,
                linewidth=0,
                zorder=0,
            )
            axis.fill_between(
                x,
                q10,
                q90,
                color=band_color,
                alpha=0.38,
                linewidth=0,
                zorder=1,
            )
            axis.fill_between(
                x,
                q25,
                q75,
                color=band_color,
                alpha=0.90,
                linewidth=0,
                zorder=2,
            )
            for boundary in (
                observed_minimum,
                observed_maximum,
                q10,
                q90,
                q25,
                q75,
            ):
                axis.plot(
                    x,
                    boundary,
                    color="#C7C7C7",
                    linewidth=0.65,
                    zorder=3,
                )
            axis.plot(
                x,
                median,
                color="#222222",
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                zorder=4,
            )
            for position, values in enumerate(data):
                jitter = rng.uniform(-0.055, 0.055, size=len(values))
                axis.scatter(
                    np.full(len(values), x[position]) + jitter,
                    values,
                    s=10,
                    color="#222222",
                    alpha=0.45,
                    linewidths=0,
                    zorder=3,
                )

            axis.set_title(
                PAPER_REGION_NAMES[region], fontsize=18, fontweight="normal"
            )
            axis.set_xticks(x, [str(year) for year in years])
            if index % 2 == 0:
                axis.set_ylabel(ylabel, fontsize=18)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.tick_params(axis="both", labelsize=15)

        legend_handles = [
            Patch(
                facecolor=band_color,
                alpha=0.20,
                edgecolor="#C7C7C7",
                linewidth=0.65,
                label="Total range",
            ),
            Patch(
                facecolor=band_color,
                alpha=0.38,
                edgecolor="#C7C7C7",
                linewidth=0.65,
                label="10th–90th percentiles",
            ),
            Patch(
                facecolor=band_color,
                alpha=0.90,
                edgecolor="#C7C7C7",
                linewidth=0.65,
                label="25th–75th percentiles",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                label="Median",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle="none",
                marker="o",
                markersize=4.0,
                alpha=0.45,
                label="Individual equilibria",
            ),
        ]
        legend_handles = [legend_handles[index] for index in (0, 3, 1, 4, 2)]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=10.2,
            frameon=True,
            framealpha=0.9,
            handlelength=1.4,
            handletextpad=0.45,
            columnspacing=0.9,
            borderpad=0.45,
            labelspacing=0.35,
            bbox_to_anchor=(0.5, 0.075),
        )
        figure.subplots_adjust(
            left=0.13, right=0.98, top=0.95, bottom=0.21, wspace=0.34, hspace=0.50
        )
        save_figure(figure, output_dir, stem)


def plot_capacity_bands_by_region(
    capacity_rows: pd.DataFrame, output_dir: Path
) -> None:
    plot_regional_equilibrium_bands_paper_style(
        capacity_rows,
        "capacity_gw",
        CAPACITY_YEARS,
        "Capacity [GW]",
        "#7570B3",
        output_dir,
        "equilibrium_bands_capacity_by_region",
    )


def plot_price_bands_by_region(price_rows: pd.DataFrame, output_dir: Path) -> None:
    plot_regional_equilibrium_bands_paper_style(
        price_rows,
        "price_usd_per_kw",
        MARKET_YEARS,
        "Price [$/kW]",
        "#A83232",
        output_dir,
        "equilibrium_bands_prices_by_region",
    )


def plot_pca(assignments: pd.DataFrame, pca_info: dict, output_dir: Path) -> None:
    family_colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#7030A0"]
    figure, axis = plt.subplots(figsize=(8.4, 6.2))
    families = sorted(assignments["family"].unique())
    for family_index, family in enumerate(families):
        subset = assignments[assignments["family"] == family]
        axis.scatter(
            subset["pc1_score"],
            subset["pc2_score"],
            s=62,
            color=family_colors[family_index],
            label=family,
            edgecolor="white",
            linewidth=0.7,
        )
        label_offsets = [
            (6, 6),
            (6, -12),
            (-22, 6),
            (-22, -12),
            (12, 14),
            (12, -16),
            (-30, 14),
            (-30, -16),
            (18, 4),
            (18, -8),
            (-30, 4),
            (-30, -8),
        ]
        for _, row in subset.iterrows():
            code_number = int(str(row["figure_code"])[1:])
            axis.annotate(
                row["figure_code"],
                (row["pc1_score"], row["pc2_score"]),
                xytext=label_offsets[(code_number - 1) % len(label_offsets)],
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(0, color="#BFBFBF", linewidth=0.7)
    axis.axvline(0, color="#BFBFBF", linewidth=0.7)
    axis.set_xlabel(f"PC1 ({pca_info['pc1_explained_percent']:.1f}% explained)")
    axis.set_ylabel(f"PC2 ({pca_info['pc2_explained_percent']:.1f}% explained)")
    axis.set_title("Outcome families from capacity and price pathways")
    axis.legend(title=f"Ward families (k={pca_info['best_k']})", loc="best")
    axis.grid(color="#ECECEC", linewidth=0.5)
    figure.tight_layout()
    save_figure(figure, output_dir, "pca_equilibrium_families")


def metric_summary(system_metrics: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for metric in (
        "total_capacity_gw",
        "mean_price_usd_per_kw",
        "demand_weighted_price_usd_per_kw",
        "price_dispersion_usd_per_kw",
        "capacity_demand_ratio",
        "cross_border_trade_gw",
        "cross_border_share",
    ):
        pieces.append(describe_groups(system_metrics, ["year"], metric))
    return pd.concat(pieces, ignore_index=True)


def readable_metric(metric: str) -> str:
    return {
        "total_capacity_2040_gw": "total capacity in 2040",
        "demand_weighted_price_2040_usd_per_kw": "demand-weighted price in 2040",
        "cross_border_trade_2040_gw": "cross-border trade in 2040",
    }.get(metric, metric)


def build_report(
    candidate_metrics: pd.DataFrame,
    price_summary: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    associations: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    pass_rates: pd.DataFrame,
    pca_info: dict,
) -> str:
    cap_2040 = candidate_metrics["total_capacity_2040_gw"]
    price_2040 = candidate_metrics["demand_weighted_price_2040_usd_per_kw"]
    corr = associations[
        (associations["x"] == "total_capacity_2040_gw")
        & (associations["y"] == "demand_weighted_price_2040_usd_per_kw")
    ].iloc[0]
    widest_price = price_summary[price_summary["year"] == 2040].sort_values(
        "iqr", ascending=False
    ).iloc[0]
    widest_capacity = capacity_summary[capacity_summary["year"] == 2040].sort_values(
        "iqr", ascending=False
    ).iloc[0]

    factor_pass = pass_rates[pass_rates["dimension"] == "price_factor"].copy()
    order_pass = pass_rates[pass_rates["dimension"] == "update_order"].copy()
    lines = [
        "# Statistical ranges across curated equilibrium candidates",
        "",
        "## Scope",
        "",
        "The analysis covers all 16 profiles in the curated candidate index. Each profile passed the common one-start, frozen-profile, zero-proximal 1% audit. The profiles are deterministic, selection-conditioned computational outcomes rather than independent statistical observations. Correlations, clusters, and matched contrasts are descriptive associations and must not be interpreted as economic causal effects.",
        "",
        "## Main ranges",
        "",
        f"- Total installed capacity in 2040 ranges from **{cap_2040.min():,.1f} to {cap_2040.max():,.1f} GW**.",
        f"- The demand-weighted 2040 clearing price ranges from **{price_2040.min():,.1f} to {price_2040.max():,.1f} USD/kW**.",
        f"- Across candidates, 2040 total capacity and the demand-weighted price have **Spearman ρ = {corr['spearman_rho']:.2f}** and **Pearson r = {corr['pearson_r']:.2f}**. This is an equilibrium-set association, not an estimated demand or supply effect.",
        f"- The widest regional 2040 price IQR occurs in **{widest_price['region_label']}** ({widest_price['iqr']:.1f} USD/kW). The widest 2040 capacity IQR occurs in **{widest_capacity['region_label']}** ({widest_capacity['iqr']:.1f} GW).",
        "- All period-based tables and figures use the common market years 2025, 2030, 2035, and 2040. The 2025 installed capacities are fixed by initialization and therefore have no cross-candidate dispersion.",
        "- All boxplots use minimum–maximum whiskers. Charcoal dots show individual candidate values and use small horizontal jitter only to reveal overlaps; the boxes still show the interquartile range and median.",
        "",
        "## Search success across the complete 36-branch design",
        "",
        "These rates describe the algorithm's ability to locate a one-start candidate within 15 sweeps. They do not rank economic equilibria.",
        "",
        "| Initialization factor | Accepted | Branches | Pass rate |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in factor_pass.sort_values("level").iterrows():
        lines.append(
            f"| {float(row['level']):.2f} | {int(row['n_accepted'])} | {int(row['n_branches'])} | {100 * row['pass_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "| Update-order anchor | Accepted | Branches | Pass rate |",
            "|---|---:|---:|---:|",
        ]
    )
    order_rank = {"CH-first": 0, "AF-first": 1, "EU-first": 2}
    order_pass = order_pass.assign(
        _order_rank=order_pass["level"].map(order_rank)
    ).sort_values("_order_rank")
    for _, row in order_pass.iterrows():
        lines.append(
            f"| {row['level']} | {int(row['n_accepted'])} | {int(row['n_branches'])} | {100 * row['pass_rate']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Matched candidate contrasts",
            "",
            "Each comparison holds the other encoded search settings fixed and includes only pairs for which both profiles were accepted. This conditioning makes the contrasts useful diagnostics but prevents causal interpretation.",
            "",
            "| Contrast | Outcome | Pairs | Median difference | Range |",
            "|---|---|---:|---:|---:|",
        ]
    )
    display_metrics = {
        "total_capacity_2040_gw",
        "demand_weighted_price_2040_usd_per_kw",
    }
    for _, row in contrast_summary[
        contrast_summary["metric"].isin(display_metrics)
    ].iterrows():
        lines.append(
            f"| {row['contrast']} | {readable_metric(row['metric'])} | {int(row['n_pairs'])} | "
            f"{row['median_percent_difference']:+.2f}% | {row['min_percent_difference']:+.2f}% to {row['max_percent_difference']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Outcome families",
            "",
            f"A PCA of regional capacity and price paths over 2025–2040 explains **{pca_info['pc1_pc2_explained_percent']:.1f}%** of standardized variation in its first two components. Constant features, including initialized 2025 capacities, are omitted automatically. Ward clustering selected **{pca_info['best_k']} descriptive families** by the highest silhouette score among two to five clusters ({pca_info['best_silhouette']:.3f}). Cluster membership is exploratory and is provided to support later economic interpretation.",
            "",
            "## Files",
            "",
            "- `equilibrium_statistical_analysis.xlsx`: compact workbook with summary and analysis tables.",
            "- `candidate_metrics.csv`, `price_summary.csv`, `capacity_summary.csv`: principal analysis tables.",
            "- `associations.csv`, `matched_contrasts.csv`, `cluster_assignments.csv`: exploratory relationship and family diagnostics.",
            "- `algorithm_convergence_diagnostics`: PNG/PDF figure separating penalized movement convergence, direct frozen-profile audits, and the 36-branch reinitialization outcomes.",
            "- `penalized_movement_history.csv` and `direct_anchor_audits.csv`: convergence data underlying that diagnostic figure.",
            "- PNG and vector PDF figures provide combined and region-specific price/capacity boxplots, the 2040 capacity-price relationship, matched contrasts, search pass rates, and PCA families.",
            "",
            "## Interpretation limits",
            "",
            "1. The accepted profiles are not a random sample and several share the same basin anchor.",
            "2. Damping is an algorithmic setting, not an economic primitive.",
            "3. The accepted design is unbalanced because failed branches are absent from economic-outcome comparisons.",
            "4. Only the six older CH-first PF100/PF120 profiles have three-start audit results, and all six fail that stricter 1% test. The remaining ten candidates, including both PF080 profiles, have not yet received it.",
            "5. Convergence paths stop at the first clean three-sweep movement pass: sweep 26 for CH-first and sweep 33 for AF-first and EU-first. A separate forced CH-first restart through sweep 36 is retained as a diagnostic artifact but excluded from the official stopping path.",
            "6. Statistical associations therefore support statements about observed computational equilibrium ranges and search sensitivity, not causal economic claims or global-equilibrium probabilities.",
            "",
        ]
    )
    return "\n".join(lines)


def table_records(frame: pd.DataFrame) -> dict:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return {"headers": list(clean.columns), "rows": clean.values.tolist()}


def build_workbook_bundle(
    output_dir: Path,
    workbook_json: Path,
    report: str,
    candidate_metrics: pd.DataFrame,
    price_summary: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    associations: pd.DataFrame,
    matched_summary: pd.DataFrame,
    pass_rates: pd.DataFrame,
    clusters: pd.DataFrame,
    price_rows: pd.DataFrame,
    capacity_rows: pd.DataFrame,
    branch_results: pd.DataFrame,
    system_summary: pd.DataFrame,
    pca_info: dict,
) -> None:
    workbook_json.parent.mkdir(parents=True, exist_ok=True)
    headline = {
        "candidate_count": len(candidate_metrics),
        "branch_count": len(branch_results),
        "accepted_branch_count": int(branch_results["accepted"].sum()),
        "capacity_2040_min": float(candidate_metrics["total_capacity_2040_gw"].min()),
        "capacity_2040_max": float(candidate_metrics["total_capacity_2040_gw"].max()),
        "price_2040_min": float(
            candidate_metrics["demand_weighted_price_2040_usd_per_kw"].min()
        ),
        "price_2040_max": float(
            candidate_metrics["demand_weighted_price_2040_usd_per_kw"].max()
        ),
        **pca_info,
    }
    capacity_range_rows = []
    for year in CAPACITY_YEARS:
        values = candidate_metrics[f"total_capacity_{year}_gw"].to_numpy(dtype=float)
        capacity_range_rows.append(
            {
                "year": year,
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        )
    capacity_range = pd.DataFrame(capacity_range_rows)
    price_range = system_summary[
        system_summary["metric"] == "demand_weighted_price_usd_per_kw"
    ][["year", "min", "median", "max"]]
    top_associations = associations[
        associations["scope"] == "candidate-level system outcome"
    ].copy()
    top_associations["absolute_spearman"] = top_associations["spearman_rho"].abs()
    top_associations = top_associations.sort_values("absolute_spearman", ascending=False).head(7)

    bundle = {
        "headline": headline,
        "report_markdown": report,
        "capacity_range": table_records(capacity_range),
        "price_range": table_records(price_range),
        "candidate_metrics": table_records(candidate_metrics),
        "price_summary": table_records(price_summary),
        "capacity_summary": table_records(capacity_summary),
        "associations": table_records(associations),
        "matched_summary": table_records(matched_summary),
        "pass_rates": table_records(pass_rates),
        "clusters": table_records(clusters),
        "price_rows": table_records(price_rows),
        "capacity_rows": table_records(capacity_rows),
        "branch_results": table_records(branch_results),
        "top_associations": table_records(top_associations),
        "output_directory": str(output_dir),
    }
    workbook_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workbook-json",
        type=Path,
        default=ROOT / "tmp" / "equilibrium_stats_workbook" / "workbook_data.json",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    workbook_json = args.workbook_json.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    candidates = collect_candidates()
    (
        candidate_metrics,
        capacity_rows,
        price_rows,
        regional_metrics,
        system_metrics,
    ) = candidate_and_observation_tables(candidates)
    price_summary = describe_groups(
        price_rows, ["year", "region", "region_label"], "price_usd_per_kw"
    )
    capacity_summary = describe_groups(
        capacity_rows, ["year", "region", "region_label"], "capacity_gw"
    )
    system_summary = metric_summary(system_metrics)
    associations, spearman_matrix = build_associations(
        candidate_metrics, regional_metrics
    )
    raw_contrasts, contrast_summary = matched_contrasts(candidate_metrics)
    branch_results = collect_branch_results()
    pass_rates = pass_rate_summary(branch_results)
    movement_history, penalized_profile_index = collect_penalized_movement_history()
    direct_anchor_audits = collect_direct_anchor_audits(penalized_profile_index)
    clusters, pca_loadings, silhouette, pca_info = pca_and_clusters(
        candidates, candidate_metrics
    )
    candidate_metrics = candidate_metrics.merge(
        clusters[["candidate", "figure_code", "family", "pc1_score", "pc2_score"]],
        on="candidate",
        how="left",
        validate="one_to_one",
    )
    candidate_metrics = add_horizon_capacity_price_indicators(candidate_metrics)

    tables = {
        "candidate_metrics.csv": candidate_metrics,
        "capacity_observations.csv": capacity_rows,
        "price_observations.csv": price_rows,
        "regional_market_metrics.csv": regional_metrics,
        "system_metrics.csv": system_metrics,
        "price_summary.csv": price_summary,
        "capacity_summary.csv": capacity_summary,
        "system_summary.csv": system_summary,
        "associations.csv": associations,
        "spearman_correlation_matrix.csv": spearman_matrix,
        "matched_contrasts.csv": raw_contrasts,
        "matched_contrast_summary.csv": contrast_summary,
        "branch_results.csv": branch_results,
        "pass_rate_summary.csv": pass_rates,
        "penalized_movement_history.csv": movement_history,
        "direct_anchor_audits.csv": direct_anchor_audits,
        "cluster_assignments.csv": clusters,
        "pca_loadings.csv": pca_loadings,
        "cluster_silhouette_scores.csv": silhouette,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")

    draw_boxplots(
        price_rows,
        "price_usd_per_kw",
        MARKET_YEARS,
        "Clearing price (USD/kW)",
        "Regional clearing-price distributions across 16 candidates",
        output_dir,
        "boxplots_market_prices",
    )
    draw_boxplots(
        capacity_rows,
        "capacity_gw",
        CAPACITY_YEARS,
        "Installed manufacturing capacity (GW)",
        "Regional capacity distributions across 16 candidates",
        output_dir,
        "boxplots_manufacturing_capacity",
    )
    plot_capacity_by_region(capacity_rows, output_dir)
    plot_prices_by_region(price_rows, output_dir)
    plot_capacity_bands_by_region(capacity_rows, output_dir)
    plot_price_bands_by_region(price_rows, output_dir)
    plot_capacity_price_scatter(candidate_metrics, associations, output_dir)
    plot_horizon_capacity_price_equilibria(candidate_metrics, output_dir)
    plot_pass_heatmap(branch_results, output_dir)
    plot_algorithm_convergence(
        movement_history,
        direct_anchor_audits,
        branch_results,
        output_dir,
    )
    plot_matched_contrasts(raw_contrasts, output_dir)
    plot_pca(clusters, pca_info, output_dir)

    report = build_report(
        candidate_metrics,
        price_summary,
        capacity_summary,
        associations,
        contrast_summary,
        pass_rates,
        pca_info,
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    build_workbook_bundle(
        output_dir,
        workbook_json,
        report,
        candidate_metrics,
        price_summary,
        capacity_summary,
        associations,
        contrast_summary,
        pass_rates,
        clusters,
        price_rows,
        capacity_rows,
        branch_results,
        system_summary,
        pca_info,
    )
    print(json.dumps({"output_dir": str(output_dir), "workbook_json": str(workbook_json)}, indent=2))


if __name__ == "__main__":
    main()
