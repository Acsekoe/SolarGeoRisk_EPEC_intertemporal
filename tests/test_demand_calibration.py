from __future__ import annotations

import unittest

from model.data_prep import _expected_demand_coefficients


class DemandCalibrationTests(unittest.TestCase):
    def test_documented_eu_2030_coefficients(self) -> None:
        a_dem, b_dem = _expected_demand_coefficients(967.65, 0.90, 90.0)
        self.assertAlmostEqual(a_dem, 2042.8166666666666, places=12)
        self.assertAlmostEqual(b_dem, 11.946296296296296, places=12)

    def test_documented_africa_2030_slope(self) -> None:
        _, b_dem = _expected_demand_coefficients(1135.6, 0.90, 10.0)
        self.assertAlmostEqual(b_dem, 126.17777777777778, places=12)

    def test_calibration_inputs_must_be_positive(self) -> None:
        for args in ((0.0, 0.9, 10.0), (100.0, 0.0, 10.0), (100.0, 0.9, 0.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                _expected_demand_coefficients(*args)


if __name__ == "__main__":
    unittest.main()
