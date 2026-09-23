from __future__ import annotations

import unittest
from types import SimpleNamespace

from model.run_gs import (
    OBJECTIVE_MODE_WITH_MU_AND_PENALTIES,
    OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
    RunConfig,
    _apply_data_overrides,
    _effective_run_config,
)


class ObjectiveModeTests(unittest.TestCase):
    def test_default_preserves_mu_subtraction_and_configured_penalties(self) -> None:
        cfg = _effective_run_config(RunConfig())
        self.assertEqual(cfg.objective_mode, OBJECTIVE_MODE_WITH_MU_AND_PENALTIES)
        self.assertEqual(cfg.c_pen_p, 1.0)
        self.assertEqual(cfg.c_quad_p, 0.1)

    def test_comparison_mode_zeros_all_objective_penalties(self) -> None:
        cfg = _effective_run_config(
            RunConfig(objective_mode=OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES)
        )
        for name in ("c_pen_q", "c_pen_p", "c_pen_a", "c_pen_dk"):
            self.assertEqual(getattr(cfg, name), 0.0)
        for name in ("c_quad_q", "c_quad_p", "c_quad_a"):
            self.assertEqual(getattr(cfg, name), 0.0)
        for name in (
            "c_pen_q_mid",
            "c_pen_p_mid",
            "c_pen_a_mid",
            "c_pen_dk_mid",
            "c_pen_q_final",
            "c_pen_p_final",
            "c_pen_a_final",
            "c_pen_dk_final",
        ):
            self.assertIsNone(getattr(cfg, name))

    def test_comparison_mode_sets_objective_flags_on_model_data(self) -> None:
        data = SimpleNamespace(
            eps_x=0.0,
            eps_comp=0.0,
            settings={},
            beta_t={},
            times=["2025"],
        )
        _apply_data_overrides(
            data,
            RunConfig(objective_mode=OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES),
        )
        self.assertFalse(data.settings["subtract_mu_offer_from_producer_margin"])
        self.assertFalse(data.settings["economic_quadratic_penalties_enabled"])
        self.assertFalse(data.settings["algorithmic_proximal_penalties_enabled"])
        self.assertEqual(data.settings["c_quad_p"], 0.0)
        self.assertEqual(data.settings["c_pen_p"], 0.0)


if __name__ == "__main__":
    unittest.main()
