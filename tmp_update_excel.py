import pandas as pd
import openpyxl

path = 'inputs/input_data_intertemporal_final.xlsx'

a_dem_values = {
    "ch": 400.0, "eu": 450.0, "us": 500.0, "apac": 400.0, "roa": 400.0, "af": 350.0, "row": 400.0
}

b_dem_dynamic = {
    "ch": {"2025": 1.007, "2030": 0.683, "2035": 0.519, "2040": 0.467},
    "eu": {"2025": 5.594, "2030": 4.853, "2035": 2.276, "2040": 1.320},
    "us": {"2025": 9.930, "2030": 5.429, "2035": 4.000, "2040": 3.304},
    "apac": {"2025": 4.311, "2030": 2.800, "2035": 1.806, "2040": 1.333},
    "roa": {"2025": 19.113, "2030": 14.000, "2035": 6.222, "2040": 3.733},
    "af": {"2025": 60.209, "2030": 32.857, "2035": 15.333, "2040": 7.667},
    "row": {"2025": 8.309, "2030": 5.600, "2035": 3.111, "2040": 2.154}
}

df_params = pd.read_excel(path, sheet_name='params_region')

# Map values
def apply_a_dem(r):
    return a_dem_values.get(str(r).strip().lower(), pd.NA)

def apply_b_dem(r, year):
    b_dict = b_dem_dynamic.get(str(r).strip().lower(), {})
    return b_dict.get(year, pd.NA)

df_params['a_dem_2025'] = df_params['r'].apply(apply_a_dem)
df_params['a_dem_2030'] = df_params['r'].apply(apply_a_dem)
df_params['a_dem_2035'] = df_params['r'].apply(apply_a_dem)
df_params['a_dem_2040'] = df_params['r'].apply(apply_a_dem)

df_params['b_dem_2025'] = df_params['r'].apply(lambda r: apply_b_dem(r, "2025"))
df_params['b_dem_2030'] = df_params['r'].apply(lambda r: apply_b_dem(r, "2030"))
df_params['b_dem_2035'] = df_params['r'].apply(lambda r: apply_b_dem(r, "2035"))
df_params['b_dem_2040'] = df_params['r'].apply(lambda r: apply_b_dem(r, "2040"))

with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_params.to_excel(writer, sheet_name='params_region', index=False)

print("Injected dynamic demand parameters into FINAL successfully.")
