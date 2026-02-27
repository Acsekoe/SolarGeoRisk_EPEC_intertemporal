from __future__ import annotations

import numbers
from typing import Dict, List, Tuple

import pandas as pd

from model_main import ModelData


def _require_columns(df: pd.DataFrame, cols: List[str], sheet: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{sheet}': {missing}")


def _norm_region(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


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
_TIMES = ["2025", "2030", "2035", "2040"]

# Map clean year label → possible column names in the Excel sheet
_DMAX_COL_CANDIDATES = {
    "2025": ["Dmax_2025 (GW)", "Dmax_2025(GW)", "Dmax_2025"],
    "2030": ["Dmax_2030 (GW)", "Dmax_2030(GW)", "Dmax_2030"],
    "2035": ["Dmax_2035 (GW)", "Dmax_2035(GW)", "Dmax_2035"],
    "2040": ["Dmax_2040 (GW)", "Dmax_2040(GW)", "Dmax_2040"],
}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data_from_excel(path: str) -> ModelData:
    """Load intertemporal ModelData for the offer model from an Excel file.
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

    _require_columns(df_regions, ["r"], "regions")
    regions = [_norm_region(v) for v in df_regions["r"].tolist()]
    regions = [r for r in regions if r]
    if not regions:
        raise ValueError("No regions found in sheet 'regions'.")

    # params_region
    try:
        df_params = pd.read_excel(path, sheet_name="params_region")
    except ValueError as exc:
        raise ValueError("Missing required sheet: params_region") from exc

    df_params.columns = [str(c).strip() for c in df_params.columns]
    _require_columns(df_params, ["r", "Qcap_exist (GW)", "c_man (USD/kW)", "a_dem (USD/kW)", "b_dem (USD/kW)"], "params_region")

    d_col = next((c for c in ["D", "D (GW)"] if c in df_params.columns), None)
    dmax_col = next((c for c in ["Dmax", "Dmax (GW)", "Dmax_2025 (GW)", "Dmax_2025(GW)", "Dmax_2025"] if c in df_params.columns), None)
    if dmax_col is None and d_col is None:
        raise ValueError("Missing demand-cap column in sheet 'params_region'. Add 'Dmax' or legacy 'D'.")

    df_params = df_params.copy()
    df_params["r"] = df_params["r"].map(_norm_region)
    df_params = df_params[df_params["r"].isin(regions)]

    row_map: Dict[str, pd.Series] = {}
    for _, row in df_params.iterrows():
        r = row["r"]
        if r and r not in row_map:
            row_map[r] = row

    missing_regions = [r for r in regions if r not in row_map]
    if missing_regions:
        raise ValueError(f"params_region missing rows for regions: {missing_regions}")

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
        if d_f <= 0.0:
            raise ValueError(f"Column 'D' must be > 0 for region '{r}'. Got: {d_f}")
        a_v = row["a_dem (USD/kW)"]
        b_v = row["b_dem (USD/kW)"]
        a_f = float(a_v) if not pd.isna(a_v) else float("nan")
        b_f = float(b_v) if not pd.isna(b_v) else float("nan")
        if not pd.isna(a_f) and a_f <= 0.0:
            raise ValueError(f"Column 'a_dem (USD/kW)' must be > 0 for region '{r}'. Got: {a_f}")
        if not pd.isna(b_f) and b_f <= 0.0:
            raise ValueError(f"Column 'b_dem (USD/kW)' must be > 0 for region '{r}'. Got: {b_f}")
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
            if col is not None:
                raw = row_map[r].get(col, float("nan"))
                val = float(raw) if not pd.isna(raw) else float(Dmax_base[r])
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
    df_ship["exporter"] = df_ship["exporter"].map(_norm_region)
    importers = [str(c).strip().lower() for c in df_ship.columns if c != "exporter"]
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
    rho_p_default = _get_setting_float(settings, "rho_p", 1e-3)
    eps_comp = _get_setting_float(settings, "eps_comp", 1e-6)
    eps_x = _get_setting_float(settings, "eps_x", 1e-3)

    players_raw = settings.get("players", None)
    players_list: List[str] = []
    if players_raw is not None and str(players_raw).strip():
        seen = set()
        for t in [t.strip().lower() for t in str(players_raw).split(",")]:
            if t in regions and t not in seen:
                players_list.append(t)
                seen.add(t)
    if not players_list:
        players_list = list(regions)

    rho_p: Dict[str, float] = {}
    p_offer_max: Dict[str, float] = {}
    kappa_Q: Dict[str, float] = {r: 0.0 for r in regions}
    
    g_exp_ub: Dict[str, float] = {}
    g_dec_ub: Dict[str, float] = {}
    f_hold: Dict[str, float] = {}
    c_inv: Dict[str, float] = {}
    Kcap_2025: Dict[str, float] = {}

    kappa_col = next((c for c in df_params.columns if str(c).strip().lower() == "kappa_q"), None)
    if kappa_col is not None:
        for r in regions:
            row = row_map[r]
            value = row[kappa_col]
            value_f = 0.0 if pd.isna(value) else float(value)
            kappa_Q[r] = value_f

    col_rho_p = _find_col(df_params, ["rho_p (scalar)", "rho_p"])
    col_p_offer_max = _find_col(df_params, ["p_offer_max (USD/kW)", "p_offer_max"])
    col_g_exp = _find_col(df_params, ["g_exp_ub (rate/yr)", "g_exp_ub (%/yr)", "g_exp_ub"])
    col_g_dec = _find_col(df_params, ["g_dec_ub (rate/yr)", "g_dec_ub (%/yr)", "g_dec_ub"])
    col_f_hold = _find_col(df_params, ["f_hold (mUSD/GW-yr)", "f_hold (USD/kW/yr)", "f_hold (USD/kW)", "f_hold"])
    col_c_inv = _find_col(df_params, ["c_inv (mUSD/GW-yr)", "c_inv (USD/kW/yr)", "c_inv (USD/kW)", "c_inv"])

    for r in regions:
        row = row_map[r]
        rho_p[r] = _opt_float(row, col_rho_p, float(rho_p_default)) if col_rho_p else float(rho_p_default)
        p_offer_max[r] = _opt_float(row, col_p_offer_max, p_offer_max_default) if col_p_offer_max else p_offer_max_default
        
        g_exp_ub[r] = _opt_float(row, col_g_exp, 0.05) if col_g_exp else 0.05
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
        print(f"[DEMAND CAL] eps_ref={eps_ref} qref_frac={qref_frac} pref_markup={pref_markup}")
        for tp in _TIMES:
            a_vals, b_vals = [], []
            for r in regions:
                dmax_val = Dmax_t[(r, tp)]
                Qref = qref_frac * dmax_val
                Pref = pref_markup * c_man[r]
                b_val = Pref / (eps_ref * max(Qref, 1e-9))
                a_val = Pref + b_val * Qref
                
                a_dem_t_impl[(r, tp)] = a_val
                b_dem_t_impl[(r, tp)] = b_val
                a_vals.append(a_val)
                b_vals.append(b_val)
                
                if r in {"ch", "eu", "roa"} and tp == "2030":
                    q_pref = (a_val - Pref)/b_val if b_val > 0 else 0
                    q_2pref = (a_val - 2*Pref)/b_val if b_val > 0 else 0
                    print(f"[DEMAND CAL] {r},2030: Dmax={dmax_val:.2f} Qref={Qref:.2f} Pref={Pref:.2f} a={a_val:.2f} b={b_val:.4f} Q(Pref)={q_pref:.2f} Q(2Pref)={q_2pref:.2f}")

            if a_vals:
                print(f"[DEMAND CAL] t={tp}: a min/mean/max = {min(a_vals):.2f}/{sum(a_vals)/len(a_vals):.2f}/{max(a_vals):.2f} "
                      f"b min/mean/max = {min(b_vals):.4f}/{sum(b_vals)/len(b_vals):.4f}/{max(b_vals):.4f}")

    # Debug print as requested
    def _print_stats(name: str, d: dict):
        vals = [v for v in d.values() if not pd.isna(v)]
        if not vals:
            print(f"[DATA LOAD] {name}: No values loaded")
            return
        print(f"[DATA LOAD] {name}: min={min(vals):.4g}, mean={sum(vals)/len(vals):.4g}, max={max(vals):.4g}")
        
    _print_stats("c_inv", c_inv)
    _print_stats("f_hold", f_hold)
    _print_stats("g_exp_ub", g_exp_ub)
    _print_stats("g_dec_ub", g_dec_ub)

    return ModelData(
        regions=regions,
        players=players_list,
        non_strategic=set(),
        D=D,
        a_dem=a_dem,
        b_dem=b_dem,
        Dmax=Dmax_base,
        Qcap=Qcap,
        c_man=c_man,
        c_ship=c_ship,
        
        # New components
        p_offer_ub=p_offer_ub,
        rho_p=rho_p,
        g_exp_ub=g_exp_ub,
        g_dec_ub=g_dec_ub,
        
        eps_x=float(eps_x),
        eps_comp=float(eps_comp),
        kappa_Q=kappa_Q,
        settings=settings,
        times=list(_TIMES),
        Dmax_t=Dmax_t,
        Kcap_2025=Kcap_2025,
        f_hold=f_hold,
        c_inv=c_inv,
        a_dem_t=a_dem_t_impl,
        b_dem_t=b_dem_t_impl,
    )
