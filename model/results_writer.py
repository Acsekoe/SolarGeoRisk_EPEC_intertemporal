from __future__ import annotations

import os
from typing import Dict, List, Any

import pandas as pd

try:
    from .model_main import ModelData
except ImportError:
    from model_main import ModelData


def _safe_get(d: Dict, k, default=0.0) -> float:
    try:
        v = d.get(k, default)
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def write_results_excel(
    *,
    data: ModelData,
    state: Dict[str, Dict],
    iter_rows: List[Dict[str, object]],
    detailed_iter_rows: List[Dict[str, object]] | None = None,
    output_path: str,
    meta: Dict[str, object] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    Q_offer = state.get("Q_offer", {})
    Kcap = state.get("Kcap", {})
    dK_net = state.get("dK_net", {})
    x_dem = state.get("x_dem", {})
    lam = state.get("lam", {})
    x = state.get("x", {})
    p_offer = state.get("p_offer", {})
    obj = state.get("obj", {})
    # Solved c_man_var from the model (not reconstructed).
    c_man_var = state.get("c_man_var", {})

    times = data.times or ["2025", "2030", "2035", "2040"]

    # Regions Sheet
    region_rows: List[Dict[str, object]] = []

    for r in data.regions:
        for t in times:
            imports = sum(_safe_get(x, (exp, r, t), 0.0) for exp in data.regions)
            exports = sum(_safe_get(x, (r, imp, t), 0.0) for imp in data.regions)
            utilized_capacity = float(exports)
            kcap_val = _safe_get(Kcap, (r, t), float((data.Kcap_2025 or data.Qcap).get(r, 0.0)))
            utilization_rate = utilized_capacity / kcap_val if kcap_val > 0.0 else 0.0
            a_dem_used = data.a_dem_t.get((r, t), data.a_dem.get(r, 0.0)) if data.a_dem_t else data.a_dem.get(r, 0.0)
            b_dem_used = data.b_dem_t.get((r, t), data.b_dem.get(r, 0.0)) if data.b_dem_t else data.b_dem.get(r, 0.0)
            lam_val = _safe_get(lam, (r, t), 0.0)
            Q_implied = max(0.0, (a_dem_used - lam_val) / b_dem_used) if b_dem_used > 0 else 0.0

            region_rows.append(
                {
                    "r": r,
                    "t": t,
                    "Kcap": kcap_val,
                    "net_cap_change": _safe_get(dK_net, (r, t), 0.0),
                    "Icap_report": max(_safe_get(dK_net, (r, t), 0.0), 0.0),
                    "Dcap_report": max(-_safe_get(dK_net, (r, t), 0.0), 0.0),
                    "Q_offer": _safe_get(Q_offer, (r, t), 0.0),
                    "utilized_capacity": utilized_capacity,
                    "utilization_rate": utilization_rate,
                    "x_dem": _safe_get(x_dem, (r, t), 0.0),
                    "lam": lam_val,
                    "obj": _safe_get(obj, r, 0.0), # Objective is scalar per region
                    "imports": float(imports),
                    "exports": float(exports),
                    "Kcap_init": float((data.Kcap_2025 or data.Qcap).get(r, 0.0)),
                    "Qcap_init": float(data.Qcap.get(r, 0.0)),
                    "a_dem_used": a_dem_used,
                    "b_dem_used": b_dem_used,
                    "Q_implied": Q_implied,
                    # Solved manufacturing cost variable (actual model level, not reconstructed).
                    "c_man_var": _safe_get(c_man_var, (r, t), float(data.c_man.get(r, 0.0))),
                    # Exogenous base manufacturing cost for reference.
                    "c_man_base": float(data.c_man.get(r, 0.0)),
                }
            )
            
    df_regions = pd.DataFrame(region_rows)

    # Flows Sheet
    flow_rows: List[Dict[str, object]] = []
    
    for exp in data.regions:
        for imp in data.regions:
            for t in times:
                flow_rows.append(
                    {
                        "exp": exp,
                        "imp": imp,
                        "t": t,
                        "x": _safe_get(x, (exp, imp, t), 0.0),
                        "p_offer": _safe_get(p_offer, (exp, imp, t), 0.0),
                        "c_ship": float(data.c_ship.get((exp, imp), 0.0)),
                        "c_man": float(data.c_man.get(exp, 0.0)),
                    }
                )

    df_flows = pd.DataFrame(flow_rows)

    # Iteration History
    df_iters = pd.DataFrame(iter_rows)
    
    # Meta
    df_meta = pd.DataFrame(list((meta or {}).items()), columns=["key", "value"])
    
    # Detailed Iterations
    df_detailed = pd.DataFrame(detailed_iter_rows) if detailed_iter_rows else pd.DataFrame()

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Define formats
        header_fmt = workbook.add_format({'bold': True, 'text_wrap': False, 'valign': 'top', 'border': 1})

        def write_sheet(df: pd.DataFrame, sheet_name: str):
            if df.empty:
                return
                
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            
            # Apply header format and auto-width
            for idx, col in enumerate(df.columns):
                # Write header with format
                worksheet.write(0, idx, col, header_fmt)
                
                # Estimate width
                max_len = max(
                    len(str(col)),
                    df[col].astype(str).map(len).max() if not df.empty else 0
                )
                width = min(max(max_len + 2, 10), 30) # clamp between 10 and 30
                worksheet.set_column(idx, idx, width)
            
            # Freeze top row
            worksheet.freeze_panes(1, 0)
            
            # Add simple autofilter
            (max_row, max_col) = df.shape
            if max_row > 0:
                worksheet.autofilter(0, 0, max_row, max_col - 1)

        write_sheet(df_regions, "regions")
        write_sheet(df_flows, "flows")
        write_sheet(df_iters, "iters")
        if not df_detailed.empty:
            write_sheet(df_detailed, "detailed_iters")
        write_sheet(df_meta, "meta")

