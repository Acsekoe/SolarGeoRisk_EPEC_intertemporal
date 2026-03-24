import re

def go():
    def normalize(s):
        return s.replace('\\r\\n', '\\n')

    # ---- 1. model_main.py ----
    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/model_main.py', 'r', encoding='utf-8') as f:
        text = normalize(f.read())
    
    t1 = normalize("""    p_offer_ub: Dict[Tuple[str, str], float]

    rho_p: Dict[str, float]

    eps_x: float
    eps_comp: float

    kappa_Q: Dict[str, float] | None = None""")
    r1 = normalize("""    p_offer_ub: Dict[Tuple[str, str], float]

    eps_x: float
    eps_comp: float""")
    if t1 in text:
        text = text.replace(t1, r1)
    else:
        print("model_main t1 not found")

    t2 = normalize("""    kappa_map = getattr(data, "kappa_Q", None) or {}
    bad_kappa = sorted([k for k in kappa_map if float(kappa_map[k]) < 0.0])
    if bad_kappa:
        raise ValueError(f"All kappa_Q must be >= 0. Invalid regions: {bad_kappa}")""")
    if t2 in text:
        text = text.replace(t2, "")
    else:
        print("model_main t2 not found")

    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/model_main.py', 'w', encoding='utf-8') as f:
        f.write(text)
    
    # ---- 2. data_prep.py ----
    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/data_prep.py', 'r', encoding='utf-8') as f:
        text = normalize(f.read())
    
    # Remove col_mapping entry
    text = text.replace('        "rho_p_scalar": "rho_p",\n', "")

    # Remove var definitions
    text = text.replace('    rho_p_default = _get_setting_float(settings, "rho_p", 1e-3)\n', "")
    text = text.replace('    rho_p: Dict[str, float] = {}\n', "")
    text = text.replace('    kappa_Q: Dict[str, float] = {r: 0.0 for r in regions}\n', "")

    # Remove exact blocks
    t3 = normalize("""    kappa_col = next((c for c in df_params.columns if str(c).strip().lower() == "kappa_q"), None)
    if kappa_col is not None:
        for r in regions:
            row = row_map[r]
            value = row[kappa_col]
            value_f = 0.0 if pd.isna(value) else float(value)
            kappa_Q[r] = value_f""")
    if t3 in text:
        text = text.replace(t3, "")
    else:
        print("data_prep t3 not found")
        
    text = text.replace('    col_rho_p = _find_col(df_params, ["rho_p (scalar)", "rho_p"])\n', "")
    text = text.replace('        rho_p[r] = _opt_float(row, col_rho_p, float(rho_p_default)) if col_rho_p else float(rho_p_default)\n', "")
    text = text.replace('        rho_p=rho_p,\n', "")
    text = text.replace('        kappa_Q=kappa_Q,\n', "")

    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/data_prep.py', 'w', encoding='utf-8') as f:
        f.write(text)

    print("Replacements done for model_main & data_prep!")

if __name__ == '__main__':
    go()
