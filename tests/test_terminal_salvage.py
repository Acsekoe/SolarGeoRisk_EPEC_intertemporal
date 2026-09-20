from __future__ import annotations

import unittest

from model.model_main import (
    ModelData,
    _implied_capacity_path,
    _operating_times,
    _terminal_salvage_credit,
    _terminal_salvage_discount_factor,
)


def model_data(*, fraction: float = 0.5, state_only: bool = True) -> ModelData:
    times = ["2025", "2030", "2035", "2040", "2045"]
    return ModelData(
        regions=["r"],
        players=["r"],
        non_strategic=set(),
        D={"r": 1.0},
        a_dem={"r": 2.0},
        b_dem={"r": 1.0},
        Dmax={"r": 1.0},
        Qcap={"r": 1.0},
        c_man={"r": 1.0},
        c_ship={("r", "r"): 0.0},
        p_offer_ub={("r", "r"): 10.0},
        eps_x=1e-3,
        eps_comp=1e-3,
        settings={
            "discount_rate": 0.02,
            "terminal_salvage_fraction": fraction,
            "terminal_capacity_state_only": state_only,
        },
        times=times,
        c_inv={"r": 200.0},
        beta_t={tp: 1.0 / (1.02 ** (int(tp) - 2025)) for tp in times},
        years_to_next={tp: 5.0 for tp in times},
    )


class TerminalSalvageTests(unittest.TestCase):
    def test_terminal_state_is_discounted_to_2045(self) -> None:
        data = model_data()
        self.assertAlmostEqual(
            _terminal_salvage_discount_factor(data),
            1.0 / (1.02**20),
            places=12,
        )

    def test_legacy_terminal_market_is_discounted_to_block_end(self) -> None:
        data = model_data(state_only=False)
        self.assertAlmostEqual(
            _terminal_salvage_discount_factor(data),
            1.0 / (1.02**25),
            places=12,
        )

    def test_credit_is_fraction_of_investment_cost_times_terminal_stock(self) -> None:
        data = model_data(fraction=0.5)
        expected = (1.0 / (1.02**20)) * 0.5 * 200.0 * 10.0
        self.assertAlmostEqual(
            _terminal_salvage_credit(data, "r", 10.0),
            expected,
            places=12,
        )

    def test_zero_fraction_preserves_baseline_objective(self) -> None:
        data = model_data(fraction=0.0)
        self.assertEqual(_terminal_salvage_credit(data, "r", 10.0), 0.0)

    def test_operating_horizon_excludes_terminal_capacity_state(self) -> None:
        self.assertEqual(
            _operating_times(model_data()),
            ["2025", "2030", "2035", "2040"],
        )

    def test_2040_move_determines_2045_terminal_stock(self) -> None:
        data = model_data()
        path = _implied_capacity_path(
            data,
            list(data.times or []),
            {("r", "2040"): 2.0},
        )
        self.assertEqual(path[("r", "2040")], 1.0)
        self.assertEqual(path[("r", "2045")], 11.0)


if __name__ == "__main__":
    unittest.main()
