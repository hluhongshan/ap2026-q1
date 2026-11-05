"""Verification tests for the Q1B differentiable SVI and neural surface."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "approach2_q1b_neural.py"
SPEC = importlib.util.spec_from_file_location("approach2_q1b_neural", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
Q1B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q1B)
A1 = Q1B.A1

torch.set_num_threads(1)
torch.manual_seed(0)

TRUE_PARAMS = np.array([-0.005, 0.15, -0.40, -0.03, 0.12])
TRUE_F = 4000.0
TRUE_R = 0.02
TRUE_T = 0.50


class TestNeuralQ1B(unittest.TestCase):
    def test_autograd_w_and_dwdT_match_finite_differences(self):
        k = np.linspace(-0.12, 0.12, 9)
        params = np.array([0.02, 0.12, -0.45, -0.02, 0.14])
        model = Q1B.TorchSVI(params)
        k_tensor = torch.tensor(k, dtype=torch.float64, requires_grad=True)
        w, w1, _, _ = Q1B.torch_svi_g(k_tensor, model)
        w1_auto = torch.autograd.grad(w.sum(), k_tensor, create_graph=True)[0]
        w2_auto = torch.autograd.grad(w1_auto.sum(), k_tensor)[0]

        h = 1e-5
        w_plus = A1.svi_w(k + h, params)
        w_minus = A1.svi_w(k - h, params)
        w1_fd = (w_plus - w_minus) / (2.0 * h)
        w2_fd = (w_plus - 2.0 * A1.svi_w(k, params) + w_minus) / (h * h)
        np.testing.assert_allclose(w1_auto.detach().numpy(), w1_fd, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(w2_auto.detach().numpy(), w2_fd, rtol=1e-5, atol=1e-7)

        network = Q1B.IVSurface(hidden=12, hidden_layers=2)
        k_point = torch.tensor(k, dtype=torch.float64, requires_grad=True)
        T_point = torch.tensor(np.linspace(0.08, 1.1, len(k)), dtype=torch.float64, requires_grad=True)
        t_scale = math.sqrt(1.1)
        sigma = network(k_point, T_point, t_scale)
        w_network = sigma * sigma * T_point
        w_k_auto = torch.autograd.grad(w_network.sum(), k_point, create_graph=True)[0]
        w_T_auto = torch.autograd.grad(w_network.sum(), T_point, create_graph=True)[0]

        h_k = 1e-5
        with torch.no_grad():
            w_k_plus = network(k_point + h_k, T_point, t_scale) ** 2 * T_point
            w_k_minus = network(k_point - h_k, T_point, t_scale) ** 2 * T_point
            w_T_plus = network(k_point, T_point + h_k, t_scale) ** 2 * (T_point + h_k)
            w_T_minus = network(k_point, T_point - h_k, t_scale) ** 2 * (T_point - h_k)
        w_k_fd = (w_k_plus - w_k_minus) / (2.0 * h_k)
        w_T_fd = (w_T_plus - w_T_minus) / (2.0 * h_k)
        np.testing.assert_allclose(w_k_auto.detach().numpy(), w_k_fd.numpy(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(w_T_auto.detach().numpy(), w_T_fd.numpy(), rtol=1e-6, atol=1e-8)

    def test_torch_g_matches_approach1_analytic_g(self):
        k = np.linspace(-0.14, 0.14, 29)
        model = Q1B.TorchSVI(TRUE_PARAMS)
        k_tensor = torch.tensor(k, dtype=torch.float64)
        _, _, _, g_torch = Q1B.torch_svi_g(k_tensor, model)
        _, _, _, g_analytic, _ = A1.svi_g(k, TRUE_PARAMS)
        np.testing.assert_allclose(g_torch.detach().numpy(), g_analytic, rtol=1e-9, atol=1e-10)

    def test_torch_price_matches_approach1_black(self):
        DF = math.exp(-TRUE_R * TRUE_T)
        cases = [
            ("C", 1.0, 0.90, TRUE_T, DF, 0.22),
            ("C", 1.0, 1.00, TRUE_T, DF, 0.18),
            ("C", 1.0, 1.15, TRUE_T, DF, 0.31),
            ("P", 1.0, 0.85, TRUE_T, DF, 0.27),
            ("P", 1.0, 1.00, TRUE_T, DF, 0.18),
            ("P", 1.0, 1.20, TRUE_T, DF, 0.35),
            ("C", 4000.0, 3800.0, 0.08, 0.999, 0.15),
            ("P", 4000.0, 4300.0, 1.50, 0.970, 0.25),
        ]
        for cp, F, K, T, DF, sigma in cases:
            expected = A1.bs_price(cp, F, K, T, DF, sigma)
            got = float(Q1B.bs_price_torch(cp, F, K, T, DF, sigma))
            self.assertLess(abs(got - expected), 1e-10, msg=f"{cp} F={F} K={K} T={T}")

    def test_torch_svi_recovers_known_parameters(self):
        k = np.linspace(-0.25, 0.25, 81)
        w = A1.svi_w(k, TRUE_PARAMS)
        x0 = np.array([-0.010, 0.14, -0.42, -0.02, 0.13])
        fit = Q1B.fit_svi_torch(
            k, w, x0,
            adam_steps=30000,
            lbfgs_steps=500,
            lr=0.03,
            seed=42,
        )
        self.assertLess(fit["objective"], 1e-14)
        for name, true, got in zip(("a", "b", "rho", "m", "eta"), TRUE_PARAMS, fit["params"]):
            self.assertLess(abs(got - true), 1e-5, msg=f"{name}: true={true} got={got}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
