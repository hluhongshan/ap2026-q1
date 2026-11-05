"""Verification tests for the Q1B differentiable SVI and neural surface."""

from __future__ import annotations

import importlib.util
import math
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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

    def test_positive_scaling_preserves_quadratic_minimizer(self):
        target = torch.tensor([1.25, -0.75], dtype=torch.float64)

        def fit(scale: float) -> np.ndarray:
            x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.LBFGS(
                [x],
                lr=1.0,
                max_iter=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = scale * ((x - target) ** 2).sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            return x.detach().numpy().copy()

        np.testing.assert_allclose(fit(1.0), fit(9.0), atol=1e-8)

    def test_neural_loss_scaling_is_a_constant_multiple(self):
        torch.manual_seed(7)
        model = Q1B.IVSurface(hidden=4, hidden_layers=1)
        k = torch.tensor([-0.10, 0.0, 0.10], dtype=torch.float64)
        T = torch.tensor([0.10, 0.20, 0.50], dtype=torch.float64)
        unit = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        colloc_k = torch.tensor(
            np.tile(np.linspace(Q1B.COLLOC_K_START, Q1B.COLLOC_K_STOP, 5), 3),
            dtype=torch.float64,
        )
        colloc_T = torch.tensor(
            np.repeat(np.array([0.10, 0.20, 0.50]), 5),
            dtype=torch.float64,
        )

        def evaluate(scale: float):
            model.zero_grad(set_to_none=True)
            total, price, butterfly, calendar = Q1B.neural_loss(
                model, k, T, unit, colloc_k, colloc_T,
                math.sqrt(0.50), 0.3, 0.2, scale=scale,
            )
            total.backward()
            gradients = [param.grad.detach().clone() for param in model.parameters()]
            return float(total.detach()), price, butterfly, calendar, gradients

        total_a, price_a, butterfly_a, calendar_a, gradients_a = evaluate(1.0)
        total_b, price_b, butterfly_b, calendar_b, gradients_b = evaluate(4.0)
        self.assertAlmostEqual(total_b, 4.0 * total_a, places=12)
        self.assertAlmostEqual(price_b, price_a, places=15)
        self.assertAlmostEqual(butterfly_b, butterfly_a, places=15)
        self.assertAlmostEqual(calendar_b, calendar_a, places=15)
        for grad_a, grad_b in zip(gradients_a, gradients_b):
            np.testing.assert_allclose(grad_b.numpy(), 4.0 * grad_a.numpy(), rtol=1e-12, atol=1e-15)

    def test_lambda_pairs_share_one_initialization(self):
        date = pd.Timestamp("2025-09-15")
        records = []
        for expiry, expiry_T in (("2025-10-17", 0.08), ("2025-11-21", 0.18)):
            for k in (-0.10, 0.10):
                records.append({
                    "date": date,
                    "expiry": pd.Timestamp(expiry),
                    "date_str": "2025-09-15",
                    "expiry_str": expiry,
                    "T": expiry_T,
                    "k": k,
                    "cp": "P" if k < 0 else "C",
                    "price": 10.0,
                    "unit_price": 0.01,
                    "F": 4000.0,
                    "K": 4000.0 * math.exp(k),
                    "DF": 0.99,
                })
        otm = pd.DataFrame(records)
        captured = []

        def fake_fit(group, fit_args, lambda1, lambda2, seed, design, rep, *rest, **kwargs):
            captured.append((float(lambda1), float(lambda2), int(seed)))
            t_scale = math.sqrt(float(group["T"].max()))
            info = {
                "t_scale": t_scale,
                "loss": 0.0,
                "price_loss": 0.0,
                "total_loss_unscaled": 0.0,
                "scale": 1.0,
                "wall_seconds": 0.0,
                "max_butterfly_penalty": 0.0,
                "max_calendar_penalty": 0.0,
                "max_weighted_butterfly": 0.0,
                "max_weighted_calendar": 0.0,
                "expiry_T": sorted(group["T"].unique().astype(float).tolist()),
            }
            model = Q1B.IVSurface(hidden=4, hidden_layers=1)
            return {
                "status": "ok",
                "model": model,
                "info": info,
                "from_cache": False,
                "attempts": 1,
            }

        class FakeLog:
            def check_runtime(self, limit, phase):
                return None

            def write(self, message):
                return None

        run_args = SimpleNamespace(
            seed=42,
            max_runtime_minutes=60.0,
            fit_timeout_seconds=120.0,
            fit_retries=0,
        )
        with unittest.mock.patch.object(Q1B, "run_neural_fit_cached", fake_fit), \
                unittest.mock.patch.object(Q1B, "LAMBDA_VALUES", (0.0, 1.0)):
            Q1B.run_ablation(
                otm, [date], run_args, ROOT / ".cache" / "test_q1b", [], [], FakeLog(),
            )
        self.assertEqual(len(captured), 4)
        self.assertEqual(len({item[2] for item in captured}), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
