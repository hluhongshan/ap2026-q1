"""Cold-start recovery checks for the three SVI solvers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "approach2_svi_cold_start", ROOT / "code" / "approach2_svi_cold_start.py"
)
assert SPEC is not None and SPEC.loader is not None
COLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLD)

TRUE_PARAMS = np.array([-0.005, 0.15, -0.40, -0.03, 0.12])


class TestSVIColdStart(unittest.TestCase):
    def test_three_solvers_recover_known_parameters(self):
        k = np.round(np.linspace(-0.20, 0.20, 41), 12)
        w = COLD.A1.svi_w(k, TRUE_PARAMS)
        bounds = COLD.A1.svi_bounds(COLD.A1.DEFAULT_ETA_MIN, COLD.A1.DEFAULT_ETA_MAX)
        x0 = COLD.cold_start(k, w, bounds)
        runs = {
            "adam": COLD.run_adam(k, w, x0, steps=5000, lr=0.03, seed=42),
            "lbfgs": COLD.run_lbfgs(k, w, x0, steps=500, seed=42),
            "scipy": COLD.run_scipy(k, w, x0, bounds),
        }
        for name, run in runs.items():
            self.assertLess(run["objective"], 1e-8, msg=f"{name} objective={run['objective']:.3e}")
            np.testing.assert_allclose(
                run["params"],
                TRUE_PARAMS,
                rtol=0.05,
                atol=5e-3,
                err_msg=f"{name} did not recover the known parameters",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
