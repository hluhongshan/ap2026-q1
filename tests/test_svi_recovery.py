"""Synthetic recovery checks for the Q1A SVI pipeline."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "approach1_q1a_svi.py"
SPEC = importlib.util.spec_from_file_location("approach1_q1a_svi", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SVI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SVI)

TRUE_PARAMS = np.array([-0.005, 0.15, -0.40, -0.03, 0.12])
TRUE_F = 4000.0
TRUE_R = 0.02
TRUE_T = 0.25


def synthetic_chain(noise: float = 0.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    DF = math.exp(-TRUE_R * TRUE_T)
    S0 = TRUE_F * DF
    k = np.round(np.linspace(-0.20, 0.20, 41), 12)
    K = TRUE_F * np.exp(k)
    w = SVI.svi_w(k, TRUE_PARAMS)
    iv = np.sqrt(w / TRUE_T)
    C = np.array([SVI.bs_price("C", TRUE_F, strike, TRUE_T, DF, vol) for strike, vol in zip(K, iv)])
    P = np.array([SVI.bs_price("P", TRUE_F, strike, TRUE_T, DF, vol) for strike, vol in zip(K, iv)])
    if noise:
        C = C * (1.0 + noise * rng.standard_normal(len(C)))
        P = P * (1.0 + noise * rng.standard_normal(len(P)))
    vols = np.full_like(C, 100.0)
    return k, K, C, P, vols, DF, S0, w


class TestSVIRecovery(unittest.TestCase):
    def test_forward_recovery_within_ten_basis_points(self):
        k, K, C, P, vols, DF, S0, _ = synthetic_chain(noise=1e-6)
        estimate = SVI.estimate_forward_pair(K, C, P, vols, vols, DF, S0, TRUE_R, TRUE_T)
        self.assertLess(abs(estimate["F"] / TRUE_F - 1.0), 1e-3)
        self.assertGreaterEqual(estimate["in_band"].sum(), len(K) - 1)

    def test_svi_parameter_recovery_within_five_percent(self):
        k, K, C, P, vols, DF, S0, w = synthetic_chain()
        estimate = SVI.estimate_forward_pair(K, C, P, vols, vols, DF, S0, TRUE_R, TRUE_T)
        self.assertLess(abs(estimate["F"] / TRUE_F - 1.0), 1e-3)
        fit = SVI.fit_svi(k, w)
        self.assertIsNotNone(fit)
        params = fit["params"]
        for name, true, got in zip(("a", "b", "rho", "m", "eta"), TRUE_PARAMS, params):
            scale = max(abs(true), 1e-3)
            self.assertLess(abs(got - true) / scale, 0.05, msg=f"{name}: true={true} got={got}")
        self.assertLess(fit["rmse"], 1e-10)

    def test_analytic_g_matches_finite_difference(self):
        k = np.linspace(-0.13, 0.13, 11)
        h = 1e-5
        w = SVI.svi_w(k, TRUE_PARAMS)
        w_plus = SVI.svi_w(k + h, TRUE_PARAMS)
        w_minus = SVI.svi_w(k - h, TRUE_PARAMS)
        w1_fd = (w_plus - w_minus) / (2.0 * h)
        w2_fd = (w_plus - 2.0 * w + w_minus) / (h * h)
        g_fd = (
            (1.0 - k * w1_fd / (2.0 * w)) ** 2
            - (w1_fd * w1_fd / 4.0) * (1.0 / w + 1.0 / 4.0)
            + w2_fd / 2.0
        )
        _, _, _, g, _ = SVI.svi_g(k, TRUE_PARAMS)
        self.assertTrue(np.allclose(g, g_fd, rtol=1e-6, atol=1e-6), msg=f"max diff={np.max(np.abs(g - g_fd))}")

        w1_analytic = SVI.svi_g(k, TRUE_PARAMS)[1]
        w2_analytic = SVI.svi_g(k, TRUE_PARAMS)[2]
        self.assertTrue(np.allclose(w1_analytic, w1_fd, rtol=1e-5, atol=1e-6))
        self.assertTrue(np.allclose(w2_analytic, w2_fd, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
