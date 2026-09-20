# Penalized search profiles

This folder consolidates the saved penalized and damped Gauss--Seidel
histories for all seven update orders, including the orders that did not meet
the strategy-movement criterion.

The layout is:

`update-order/penalized_sweeps_XXX_YYY.xlsx`

Each workbook is an unchanged copy of the original result workbook. Its
`detailed_iters` sheet contains the full profile after every completed sweep,
with 30 rows per sweep (six regions times five periods). The `iters` sheet
contains sweep-level convergence and penalty diagnostics.

The continuation workbooks retain their original local iteration numbering
from 1 to 10. In this archive, their filenames map those iterations to global
sweeps 31 to 40.

`profile_index.csv` records coverage, final movement, convergence status, and
the solve-quality flag for every update order.

These are penalized search trajectories. Movement convergence in these files
is not an independent equilibrium verification.
