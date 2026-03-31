from __future__ import annotations

import math
import numbers
from typing import Dict, List, Tuple

import pandas as pd

try:
    from .model_main import ModelData
except ImportError:
    from model_main import ModelData


# =============================================================================
# Exogenous Learning-By-Doing (Swanson's Law) schedule
# =============================================================================

# IEA Net Zero Emissions scenario — global cumulative solar PV installed capacity (GW).
# Source: IEA World Energy Outlook 2023, Net Zero Emissions by 2050 Scenario.
_IEA_NZE_GLOBAL_CAPACITY_GW: Dict[int, float] = {
    2025:  2_000,   # ~2 TW baseline (end-2025 estimate)
    2030:  7_300,   # IEA NZE 2030 milestone
    2035: 10_500,   # IEA NZE 2035 (interpolated)
    2040: 14_400,   # IEA NZE 2040 milestone
    2045: 18_200,   # IEA NZE 2045 (interpolated)
    2050: 22_000,   # IEA NZE 2050 milestone
}

_LBD_LEARNING_RATE: float = 0.20   # 20% cost reduction per doubling of cumulative capacity
_LBD_BASE_YEAR:     int   = 2025


def compute_lbd_schedule(
    c_man_base_by_region: Dict[str, float],
    times: List[str],
    *,
    base_year: int = _LBD_BASE_YEAR,
    learning_rate: float = _LBD_LEARNING_RATE,
    global_capacity_gw: Dict[int, float] | None = None,
) -> Dict[Tuple[str, str], float]:
    """Pre-compute exogenous Swanson's-Law cost schedule for each (region, time).

    Formula: C_t = C_0 * (X_t / X_0)^b
      C_0  — regional base cost in base_year  (USD/kW)
      X_t  — global cumulative capacity at year t  (GW)
      X_0  — global cumulative capacity in base_year  (GW)
      b    — learning index = log(1 - learning_rate) / log(2)

    Learning is *global*: the same cost scalar applies to every region's base
    cost.  Regional cost differences arise solely from the per-region c_man
    values, not from the learning curve itself.

    Returns a dict keyed by (region_str, time_str) → cost (USD/kW).
    """
    cap_gw = global_capacity_gw if global_capacity_gw is not None else _IEA_NZE_GLOBAL_CAPACITY_GW
    if base_year not in cap_gw:
        raise ValueError(
            f"base_year {base_year} not found in global_capacity_gw. "
            f"Available years: {sorted(cap_gw.keys())}"
        )
    b: float = math.log(1.0 - learning_rate) / math.log(2.0)
    X_0: float = cap_gw[base_year]

    years_sorted = sorted(cap_gw.keys())

    def _global_capacity(year: int) -> float:
        if year in cap_gw:
            return cap_gw[year]
        # Log-linear interpolation/extrapolation.
        # Extrapolation uses the slope of the nearest segment.
        if year < years_sorted[0]:
            y_lo, y_hi = years_sorted[0], years_sorted[1]
        elif year > years_sorted[-1]:
            y_lo, y_hi = years_sorted[-2], years_sorted[-1]
        else:
            y_lo = max(y for y in years_sorted if y <= year)
            y_hi = min(y for y in years_sorted if y >= year)
            if y_lo == y_hi:
                return float(cap_gw[y_lo])
        frac = (year - y_lo) / (y_hi - y_lo)
        return math.exp(
            (1.0 - frac) * math.log(cap_gw[y_lo])
            + frac * math.log(cap_gw[y_hi])
        )

    result: Dict[Tuple[str, str], float] = {}
    for tp in times:
        year = int(tp)
        X_t = _global_capacity(year)
        scalar = (X_t / X_0) ** b          # < 1 for years after base_year
        for r, c_base in c_man_base_by_region.items():
            result[(r, tp)] = c_base * scalar
    return result


def _require_columns(df: pd.DataFrame, cols: List[str], sheet: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{sheet}': {missing}")


def _norm_region(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _resolve_region_alias(code: str, valid_regions: set[str]) -> str:
    """Resolve known region aliases to whichever code exists in valid_regions."""
    if not code:
        return code
    if code in valid_regions:
        return code

    # Treat APAC and ROA as equivalent labels and map to the configured region set.
    if code == "roa" and "apac" in valid_regions:
        return "apac"
    if code == "apac" and "roa" in valid_regions:
        return "roa"
    return code


def _get_optional_float(row: pd.Series, col: str, default: float) -> float:
    if col not in row.index:
        return float(default)
    value = row[col]
    if pd.isna(value):
        return float(default)
    return float(value)


def _get_setting_float(settings: Dict[str, object], key: str, default: float) -> float:
    if key not in settings:
        return float(default)
    try:
        value = float(settings[key])
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(value):
        return float(default)
    return float(value)


def _get_setting_bool(settings: Dict[str, object], key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if value is None or pd.isna(value):
        return bool(default)
    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
        raise ValueError(f"Invalid boolean for setting '{key}': {value!r}")

    if isinstance(value, numbers.Integral):
        iv = int(value)
        if iv == 0:
            return False
        if iv == 1:
            return True
        raise ValueError(f"Invalid boolean for setting '{key}': {value!r} (expected 0/1)")

    if isinstance(value, numbers.Real):
        fv = float(value)
        if pd.isna(fv):
            return bool(default)
        if fv == 0.0:
            return False
        if fv == 1.0:
            return True
        raise ValueError(f"Invalid boolean for setting '{key}': {value!r} (expected 0/1)")

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean for setting '{key}': {value!r}")


# ---------------------------------------------------------------------------
# Intertemporal data loader for Offer Model
# ---------------------------------------------------------------------------
_TIMES = ["2025", "2030", "2035", "2040", "2045"]
_FUTURE_FALLBACK_YEAR = "2040"

# Map clean year label → possible column names in the Excel sheet
_DMAX_COL_CANDIDATES = {
    "2025": ["Dmax_2025 (GW)", "Dmax_2025(GW)", "Dmax_2025"],
    "2030": ["Dmax_2030 (GW)", "Dmax_2030(GW)", "Dmax_2030"],
    "2035": ["Dmax_2035 (GW)", "Dmax_2035(GW)", "Dmax_2035"],
    "2040": ["Dmax_2040 (GW)", "Dmax_2040(GW)", "Dmax_2040"],
    "2045": ["Dmax_2045 (GW)", "Dmax_2045(GW)", "Dmax_2045"],
}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data_from_excel(path: str, params_region_sheet: str = "params_region") -> ModelData:
    """Load intertemporal ModelData for the offer model from an Excel file.

    Parameters
    ----------
    params_region_sheet:
        Name of the Excel sheet containing regional parameters (capacity, costs, demand).
        Defaults to ``"params_region"``; set to e.g. ``"params_region_new"`` to load
        an alternative scenario sheet with the same column structure.
    """
    def _opt_float(row: "pd.Series", col: str, default: float) -> float:
        """Read an optional float column from a pandas Series, returning default if missing/NaN."""
        if col not in row.index:
            return default
        val = row[col]
        try:
            f = float(val)
            return default if pd.isna(f) else f
        except (TypeError, ValueError):
            return default

    # regions
    try:
        df_regions = pd.read_excel(path, sheet_name="regions")
    except ValueError as exc:
        raise ValueError("Missing required sheet: regions") from exc

    df_regions = df_regions.rename(columns={"region": "r"})

    _require_columns(df_regions, ["r"], "regions")
    regions = [_norm_region(v) for v in df_regions["r"].tolist()]
    regions = [r for r in regions if r]
    if not regions:
        raise ValueError("No regions found in sheet 'regions'.")
    region_set = set(regions)

    # params_region
    try:
        df_params = pd.read_excel(path, sheet_name=params_region_sheet)
    except ValueError as exc:
        raise ValueError(f"Missing required sheet: '{params_region_sheet}'") from exc

    col_mapping = {
        "region": "r",
        "qcap_exist_gw": "Qcap_exist (GW)",
        "c_man_usd_per_kw": "c_man (USD/kW)",
        "c_inv_musd_per_gw_per_yr": "c_inv (USD/kW)",
        "f_hold_musd_per_gw_per_yr": "f_hold (USD/kW)",
        "c_hold_musd_per_gw_per_yr": "f_hold (USD/kW)",
        "kcap_2025_gw": "Kcap_2025",
        "g_exp_ub_per_yr": "g_exp_ub",
        "g_exp_ub_per_yr_gw": "g_exp_ub_per_yr_gw",
        "g_dec_ub_per_yr": "g_dec_ub",
        "p_offer_max_usd_per_kw": "p_offer_max",
        "p_full_usd_per_kw": "p_full",
        "eps_abs_base_scalar": "eps_abs_base",
    }
    for yr in _TIMES:
        col_mapping[f"dmax_{yr}_gw"] = f"Dmax_{yr}"
        col_mapping[f"a_dem_{yr}_usd_per_kw"] = f"a_dem_{yr}"
        col_mapping[f"b_dem_{yr}_usd_per_kw_per_gw"] = f"b_dem_{yr}"

    df_params = df_params.rename(columns=col_mapping)

    _require_columns(df_params, ["r", "Qcap_exist (GW)", "c_man (USD/kW)"], params_region_sheet)

    d_col = next((c for c in ["D", "D (GW)"] if c in df_params.columns), None)
    dmax_col = next((c for c in ["Dmax", "Dmax (GW)", "Dmax_2025 (GW)", "Dmax_2025(GW)", "Dmax_2025"] if c in df_params.columns), None)
    if dmax_col is None and d_col is None:
        raise ValueError(f"Missing demand-cap column in sheet '{params_region_sheet}'. Add 'Dmax' or legacy 'D'.")

    df_params = df_params.copy()
    df_params["r"] = df_params["r"].map(
        lambda v: _resolve_region_alias(_norm_region(v), region_set)
    )
    df_params = df_params[df_params["r"].isin(regions)]

    row_map: Dict[str, pd.Series] = {}
    for _, row in df_params.iterrows():
        r = row["r"]
        if r and r not in row_map:
            row_map[r] = row

    missing_regions = [r for r in regions if r not in row_map]
    if missing_regions:
        raise ValueError(f"Sheet '{params_region_sheet}' missing rows for regions: {missing_regions}")

    Qcap: Dict[str, float] = {}
    D: Dict[str, float] = {}
    Dmax_base: Dict[str, float] = {}
    a_dem: Dict[str, float] = {}
    b_dem: Dict[str, float] = {}
    c_man: Dict[str, float] = {}
    for r, row in row_map.items():
        qcap_v = row["Qcap_exist (GW)"]
        if pd.isna(qcap_v):
            raise ValueError(f"Column 'Qcap_exist (GW)' has missing value for region '{r}'.")
        qcap_f = float(qcap_v)
        if qcap_f < 0.0:
            raise ValueError(f"Column 'Qcap_exist (GW)' must be >= 0 for region '{r}'. Got: {qcap_f}")
        Qcap[r] = qcap_f

        dmax_name = dmax_col or d_col
        assert dmax_name is not None
        dmax_v = row[dmax_name]
        if pd.isna(dmax_v):
            raise ValueError(f"Column '{dmax_name}' has missing value for region '{r}'.")
        dmax_f = float(dmax_v)
        if dmax_f <= 0.0:
            raise ValueError(f"Demand cap must be > 0 for region '{r}'. Got: {dmax_f} from '{dmax_name}'.")
        Dmax_base[r] = dmax_f

        if d_col is not None:
            d_v = row[d_col]
            if pd.isna(d_v):
                raise ValueError(f"Column '{d_col}' has missing value for region '{r}'.")
            d_f = float(d_v)
        else:
            d_f = dmax_f
        # a_dem and b_dem base columns are removed
        a_v = row.get("a_dem (USD/kW)", row.get("a_dem"))
        b_v = row.get("b_dem (USD/kW)", row.get("b_dem"))
        a_f = float(a_v) if not pd.isna(a_v) else float("nan")
        b_f = float(b_v) if not pd.isna(b_v) else float("nan")
        a_dem[r] = a_f
        b_dem[r] = b_f

        c_v = row["c_man (USD/kW)"]
        if pd.isna(c_v):
            raise ValueError(f"Column 'c_man (USD/kW)' has missing value for region '{r}'.")
        c_man[r] = float(c_v)

    # --- Time-indexed Dmax ---
    Dmax_t: Dict[Tuple[str, str], float] = {}
    
    for tp in _TIMES:
        col = _find_col(df_params, _DMAX_COL_CANDIDATES[tp])
        for r in regions:
            fallback_2040 = row_map[r].get(f"Dmax_{_FUTURE_FALLBACK_YEAR}", float("nan"))
            fallback_2040_val = float(fallback_2040) if not pd.isna(fallback_2040) else float(Dmax_base[r])
            if col is not None:
                raw = row_map[r].get(col, float("nan"))
                if not pd.isna(raw):
                    val = float(raw)
                elif int(tp) > int(_FUTURE_FALLBACK_YEAR):
                    val = fallback_2040_val
                else:
                    val = float(Dmax_base[r])
            else:
                if int(tp) > int(_FUTURE_FALLBACK_YEAR):
                    val = fallback_2040_val
                else:
                    val = float(Dmax_base[r])
            if val <= 0.0:
                raise ValueError(
                    f"Dmax_t for region '{r}', year '{tp}' must be > 0. Got: {val}"
                )
            Dmax_t[(r, tp)] = val

    # c_ship
    try:
        df_ship = pd.read_excel(path, sheet_name="c_ship")
    except ValueError as exc:
        raise ValueError("Missing required sheet: c_ship") from exc

    if df_ship.shape[1] < 2:
        raise ValueError("Sheet 'c_ship' must have an exporter column and at least one importer column.")

    df_ship = df_ship.copy()
    first_col = df_ship.columns[0]
    df_ship = df_ship.rename(columns={first_col: "exporter"})
    df_ship["exporter"] = df_ship["exporter"].map(
        lambda v: _resolve_region_alias(_norm_region(v), region_set)
    )
    importers = [
        _resolve_region_alias(_norm_region(c), region_set)
        for c in df_ship.columns
        if c != "exporter"
    ]
    df_ship.columns = ["exporter"] + importers

    missing_cols = [r for r in regions if r not in importers]
    if missing_cols:
        raise ValueError(f"c_ship is missing importer columns for regions: {missing_cols}")

    df_ship = df_ship.set_index("exporter", drop=True)
    c_ship: Dict[Tuple[str, str], float] = {}
    for exp in regions:
        # Default intra-region ship back to 0 if exporter not in table, or missing
        for imp in regions:
            if exp not in df_ship.index:
                c_ship[(exp, imp)] = 0.0 if exp == imp else 9999.0 # Arbitrary high cost if no data
                continue
                
            value = df_ship.at[exp, imp]
            if pd.isna(value):
                if exp == imp:
                    c_ship[(exp, imp)] = 0.0
                else:
                    raise ValueError(f"c_ship missing value for ({exp}, {imp}).")
            else:
                c_ship[(exp, imp)] = float(value)

    # settings
    try:
        df_settings = pd.read_excel(path, sheet_name="settings")
    except ValueError as exc:
        raise ValueError("Missing required sheet: settings") from exc
    _require_columns(df_settings, ["setting", "value"], "settings")

    settings: Dict[str, object] = {}
    for _, row in df_settings.iterrows():
        key = str(row["setting"]).strip().lower()
        if not key or key == "nan":
            continue
        settings[key] = row["value"]

    p_offer_max_default = _get_setting_float(settings, "p_offer_max", 200.0)
    eps_comp = _get_setting_float(settings, "eps_comp", 1e-6)
    eps_x = _get_setting_float(settings, "eps_x", 1e-3)
    # Discount rate for NPV computation.  0.0 = undiscounted (block-length weighting only).
    discount_rate = _get_setting_float(settings, "discount_rate", 0.0)
    base_year = int(_get_setting_float(settings, "base_year", 2025.0))

    players_raw = settings.get("players", None)
    players_list: List[str] = []
    if players_raw is not None and str(players_raw).strip():
        seen = set()
        for t in [t.strip().lower() for t in str(players_raw).split(",")]:
            t = _resolve_region_alias(t, region_set)
            if t in regions and t not in seen:
                players_list.append(t)
                seen.add(t)
    if not players_list:
        players_list = list(regions)

    p_offer_max: Dict[str, float] = {}
    
    g_exp_ub: Dict[str, float] = {}
    g_exp_ub_is_absolute = False
    g_dec_ub: Dict[str, float] = {}
    f_hold: Dict[str, float] = {}
    c_inv: Dict[str, float] = {}
    Kcap_2025: Dict[str, float] = {}



    col_p_offer_max = _find_col(df_params, ["p_offer_max (USD/kW)", "p_offer_max"])
    col_g_exp_abs = _find_col(
        df_params,
        ["g_exp_ub_per_yr_gw", "g_exp_ub (GW/yr)", "g_exp_ub_gw_per_yr"],
    )
    col_g_exp_rate = _find_col(df_params, ["g_exp_ub (rate/yr)", "g_exp_ub (%/yr)", "g_exp_ub"])
    col_g_dec = _find_col(df_params, ["g_dec_ub (rate/yr)", "g_dec_ub (%/yr)", "g_dec_ub"])
    col_f_hold = _find_col(df_params, ["f_hold (mUSD/GW-yr)", "f_hold (USD/kW/yr)", "f_hold (USD/kW)", "f_hold"])
    col_c_inv = _find_col(df_params, ["c_inv (mUSD/GW-yr)", "c_inv (USD/kW/yr)", "c_inv (USD/kW)", "c_inv"])

    if col_g_exp_abs is not None:
        g_exp_ub_is_absolute = True

    for r in regions:
        row = row_map[r]
        p_offer_max[r] = _opt_float(row, col_p_offer_max, p_offer_max_default) if col_p_offer_max else p_offer_max_default
        
        if g_exp_ub_is_absolute:
            g_exp_ub[r] = _opt_float(row, col_g_exp_abs, 0.0)
        else:
            g_exp_ub[r] = _opt_float(row, col_g_exp_rate, 0.05) if col_g_exp_rate else 0.05
        g_dec_ub[r] = _opt_float(row, col_g_dec, 0.05) if col_g_dec else 0.05
        f_hold[r] = _opt_float(row, col_f_hold, 0.0) if col_f_hold else 0.0
        c_inv[r] = _opt_float(row, col_c_inv, 0.0) if col_c_inv else 0.0
        
        kcap_raw = _opt_float(row, "Kcap_2025", float("nan"))
        if not pd.isna(kcap_raw):
            Kcap_2025[r] = float(kcap_raw)
        else:
            Kcap_2025[r] = float(Qcap[r])

    p_offer_ub: Dict[Tuple[str, str], float] = {}
    for exp in regions:
        for imp in regions:
            p_offer_ub[(exp, imp)] = float(p_offer_max[exp])

    # Time-indexed demand curve generation
    demand_time_indexed = _get_setting_bool(settings, "demand_time_indexed", True)
    eps_ref = _get_setting_float(settings, "eps_ref", 0.10)
    qref_frac = _get_setting_float(settings, "qref_frac", 0.95)
    pref_markup = _get_setting_float(settings, "pref_markup", 1.5)

    a_dem_t_impl: Dict[Tuple[str, str], float] | None = None
    b_dem_t_impl: Dict[Tuple[str, str], float] | None = None

    if demand_time_indexed:
        a_dem_t_impl = {}
        b_dem_t_impl = {}

        a_cols_by_tp = {
            tp: _find_col(df_params, [f"a_dem_{tp}", f"a_dem_{tp} (USD/kW)"])
            for tp in _TIMES
        }
        b_cols_by_tp = {
            tp: _find_col(df_params, [f"b_dem_{tp}", f"b_dem_{tp} (USD/kW)", f"b_dem_{tp} (USD/GW)"])
            for tp in _TIMES
        }
        a_col_2040 = a_cols_by_tp.get(_FUTURE_FALLBACK_YEAR)
        b_col_2040 = b_cols_by_tp.get(_FUTURE_FALLBACK_YEAR)
        
        for tp in _TIMES:
            a_vals, b_vals = [], []
            for r in regions:
                dmax_val = Dmax_t[(r, tp)]
                
                # Look for explicit year columns first.
                a_col = a_cols_by_tp.get(tp)
                a_val_raw = row_map[r].get(a_col, float("nan")) if a_col else float("nan")
                
                b_col = b_cols_by_tp.get(tp)
                b_val_raw = row_map[r].get(b_col, float("nan")) if b_col else float("nan")

                # For future periods beyond 2040, inherit 2040 demand-curve parameters
                # when explicit columns are not provided in the workbook.
                if int(tp) > int(_FUTURE_FALLBACK_YEAR):
                    if pd.isna(a_val_raw) and a_col_2040:
                        a_val_raw = row_map[r].get(a_col_2040, float("nan"))
                    if pd.isna(b_val_raw) and b_col_2040:
                        b_val_raw = row_map[r].get(b_col_2040, float("nan"))

                if not pd.isna(a_val_raw):
                    a_val = float(a_val_raw)
                else:
                    a_val = float(a_dem.get(r, 500.0))
                    if pd.isna(a_val):
                        a_val = 500.0

                if not pd.isna(b_val_raw):
                    b_val = float(b_val_raw)
                else:
                    # Fallback to horizontal stretching if b_dem_t isn't explicitly provided
                    b_val = a_val / max(dmax_val, 1e-9)
                
                a_dem_t_impl[(r, tp)] = a_val
                b_dem_t_impl[(r, tp)] = b_val
                a_vals.append(a_val)
                b_vals.append(b_val)

            if a_vals:
                print(f"[DEMAND CAL] t={tp}: a min/mean/max = {min(a_vals):.2f}/{sum(a_vals)/len(a_vals):.2f}/{max(a_vals):.2f} "
                      f"b min/mean/max = {min(b_vals):.4f}/{sum(b_vals)/len(b_vals):.4f}/{max(b_vals):.4f}")

    # --- NPV discount factors and period lengths ---
    years_to_next: Dict[str, float] = {tp: 5.0 for tp in _TIMES}

    beta_t: Dict[str, float] = {}
    for tp in _TIMES:
        t_int = int(tp)
        if discount_rate == 0.0:
            beta_t[tp] = 1.0
        else:
            beta_t[tp] = 1.0 / ((1.0 + discount_rate) ** (t_int - base_year))

    print(f"[DATA LOAD] discount_rate={discount_rate} base_year={base_year}")
    print(f"[DATA LOAD] beta_t={beta_t}")

    # Debug print as requested
    def _print_stats(name: str, d: dict):
        vals = [v for v in d.values() if not pd.isna(v)]
        if not vals:
            print(f"[DATA LOAD] {name}: No values loaded")
            return
        print(f"[DATA LOAD] {name}: min={min(vals):.4g}, mean={sum(vals)/len(vals):.4g}, max={max(vals):.4g}")
        
    _print_stats("c_inv", c_inv)
    _print_stats("f_hold", f_hold)
    if g_exp_ub_is_absolute:
        _print_stats("g_exp_ub_abs (GW/yr)", g_exp_ub)
    else:
        _print_stats("g_exp_ub_rate (1/yr)", g_exp_ub)
    _print_stats("g_dec_ub", g_dec_ub)

    return ModelData(
        regions=regions,
        players=players_list,
        non_strategic=set(regions) - set(players_list),
        D=D,
        a_dem=a_dem,
        b_dem=b_dem,
        Dmax=Dmax_base,
        Qcap=Qcap,
        c_man=c_man,
        c_ship=c_ship,
        
        # New components
        p_offer_ub=p_offer_ub,
        g_exp_ub=g_exp_ub,
        g_exp_ub_is_absolute=g_exp_ub_is_absolute,
        g_dec_ub=g_dec_ub,
        
        eps_x=float(eps_x),
        eps_comp=float(eps_comp),
        settings=settings,
        times=list(_TIMES),
        Dmax_t=Dmax_t,
        Kcap_2025=Kcap_2025,
        f_hold=f_hold,
        c_inv=c_inv,
        a_dem_t=a_dem_t_impl,
        b_dem_t=b_dem_t_impl,
        beta_t=beta_t,
        years_to_next=years_to_next,
        c_man_t=compute_lbd_schedule(c_man, list(_TIMES)),
    )


# =============================================================================
# Load optional initial_state sheet for EPEC warm-start
# =============================================================================
def load_initial_state(
    path: str,
    data: "ModelData",
) -> Dict[str, Dict] | None:
    """Read the ``initial_state`` sheet from the input workbook.

    Returns a warm-start dict with keys ``Q_offer``, ``p_offer``, ``a_bid``,
    ``dK_net`` — or ``None`` if the sheet does not exist.

    Expected columns:
        region, t, Q_offer, dK_net, a_bid, p_offer_<imp1>, p_offer_<imp2>, ...
    where ``<imp>`` names match the region codes exactly.
    """
    try:
        df = pd.read_excel(path, sheet_name="initial_state")
    except ValueError:
        return None

    if df.empty:
        return None

    # Normalise
    df.columns = [str(c).strip() for c in df.columns]
    if "region" not in df.columns or "t" not in df.columns:
        raise ValueError("initial_state sheet must have 'region' and 't' columns.")

    df["region"] = df["region"].apply(_norm_region)
    df["t"] = df["t"].astype(str).str.strip()

    regions = data.regions
    times = data.times or list(_TIMES)

    Q_offer: Dict[Tuple[str, str], float] = {}
    dK_net: Dict[Tuple[str, str], float] = {}
    a_bid: Dict[Tuple[str, str], float] = {}
    p_offer: Dict[Tuple[str, str, str], float] = {}

    # Identify p_offer columns: p_offer_<importer>
    p_offer_cols = {
        c: c.replace("p_offer_", "")
        for c in df.columns
        if c.startswith("p_offer_")
    }

    for _, row in df.iterrows():
        r = str(row["region"])
        t = str(row["t"])
        if r not in regions or t not in times:
            continue

        if "Q_offer" in df.columns and not pd.isna(row["Q_offer"]):
            Q_offer[(r, t)] = float(row["Q_offer"])

        if "dK_net" in df.columns and not pd.isna(row["dK_net"]):
            dK_net[(r, t)] = float(row["dK_net"])

        if "a_bid" in df.columns and not pd.isna(row["a_bid"]):
            a_bid[(r, t)] = float(row["a_bid"])

        for col, imp in p_offer_cols.items():
            imp_norm = _norm_region(imp)
            if imp_norm in regions and not pd.isna(row[col]):
                p_offer[(r, imp_norm, t)] = float(row[col])

    if not Q_offer:
        return None

    return {
        "Q_offer": Q_offer,
        "p_offer": p_offer,
        "a_bid":   a_bid,
        "dK_net":  dK_net,
    }
