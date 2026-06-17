# Market Power and Strategic Price Formation in the Global Solar PV Supply Chain: An EPEC Framework

Intertemporal EPEC model comparing strategic regional behavior against a global welfare-maximizing planner across six regions (CH, EU, US, APAC, AF, ROW) over 2025–2040.

## Requirements

**Python packages**

```
pip install -r requirements.txt
```

**GAMS + IPOPT**

The model requires a working [GAMS](https://www.gams.com/) installation with the IPOPT solver licensed and accessible. GAMS is invoked via the `gamspy` Python package. Ensure `gams` is on your system PATH and that IPOPT is available in your GAMS solver suite.

## Repository structure

```
inputs/
  input_data_intertemporal.xlsx   # all model parameters (demand, costs, capacities)
model/
  run_gs.py                       # main entry point — RunConfig + run()
  gauss_seidel.py                 # Gauss-Seidel diagonalization solver
  model_main.py                   # per-player MPEC formulation (GAMS/GAMSPy)
  model_llp_planner.py            # global welfare-maximizing planner benchmark
  data_prep.py                    # input parsing + learning-by-doing cost schedule
  results_writer.py               # Excel output writer
  plot_results.py                 # default diagnostic plots
  compute_welfare_comparison.py   # welfare comparison table (planner vs. EPEC)
plots/
  plot_prices.py                  # Fig. 2 — market-clearing prices
  plot_welfare.py                 # Fig. 3 — regional welfare gains/losses
  plot_capacity_epec_demand.py    # Fig. 4 — capacity evolution with global demand
  plot_convergence.py             # Fig. 1 — Gauss-Seidel convergence metric
  plot_iter21_capacity_chords.py  # trade flow chord diagrams
outputs/
  llp_planner_results.xlsx        # pre-computed planner solution
  welfare_comparison.xlsx         # pre-computed welfare comparison
  sens/converged/                 # pre-computed EPEC results (3 converged runs)
  figures/                        # generated figures (PNG)
```

## Running the model

### Step 1 — solve the EPEC

Edit `PLAYER_ORDER` at the top of [model/run_gs.py](model/run_gs.py) to set the Gauss-Seidel update sequence (the paper uses `["ch", "row", "apac", "us", "eu", "af"]`). Then run:

```bash
python -m model.run_gs
```

Output is written to `outputs/` as a timestamped Excel file. To redirect output to `outputs/sens/converged/`, adjust `RunConfig.out_dir` in the same file.

Key `RunConfig` settings to tune convergence:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iters` | 30 | Maximum Gauss-Seidel iterations |
| `omega` | 0.8 | Initial damping factor |
| `omega_min` | 0.4 | Minimum damping (paper uses 0.6) |
| `tol_strat` | 1e-2 | Convergence tolerance on the maximum relative strategy change (`Delta theta`) |
| `stable_iters` | 3 | Number of consecutive sweeps below `tol_strat` required for convergence |

### Step 2 — solve the planner benchmark

The planner is solved automatically during `run_gs` if `llp_planner_results.xlsx` is absent. To re-solve it explicitly, call `solve_llp_planner()` from `model/model_llp_planner.py`.

### Step 3 — compute welfare comparison

```bash
python model/compute_welfare_comparison.py
```

Reads `outputs/llp_planner_results.xlsx` and `outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx`, writes `outputs/welfare_comparison.xlsx`.

### Step 4 — generate paper figures

Run each plotting script from the project root:

```bash
python plots/plot_convergence.py          # Fig. 1
python plots/plot_prices.py               # Fig. 2
python plots/plot_welfare.py              # Fig. 3
python plots/plot_capacity_epec_demand.py # Fig. 4
python plots/plot_iter21_capacity_chords.py
```

Figures are saved to `outputs/figures/`.

## Pre-computed results

The `outputs/` directory contains pre-computed results so figures can be reproduced without re-running the full solver:

- `outputs/llp_planner_results.xlsx` — planner benchmark solution
- `outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx` — selected EPEC equilibrium (Equilibrium 3 in the paper, player order CH→ROW→APAC→US→EU→AF)
- `outputs/welfare_comparison.xlsx` — welfare decomposition table
