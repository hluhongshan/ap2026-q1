"""Q1 Option B: differentiable SVI and a neural implied-vol surface for CFFEX IO.

This module reuses the Option A (approach1) loading, rate, parity-forward,
filter and implied-volatility code by importing ``approach1_q1a_svi.py``.
It adds:

* a torch (float64) SVI fit with b=softplus(u), rho=tanh(v), eta=softplus(s),
  a and m free, solved by Adam then L-BFGS, compared with scipy least_squares
  from the same starting point;
* a smooth MLP sigma_theta(k, T) fitted to OTM prices with butterfly and
  calendar penalties computed by autograd on a collocation grid;
* an in-sample lambda ablation, a random-drop and wing holdout robustness
  study, and a like-for-like comparison with approach1 on the same quotes.

All randomness is seeded and torch threads are set explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

# Neural fits run in freshly spawned workers.  These BLAS/OpenMP settings are
# applied before numpy/torch import because repeated high-order autograd fits
# in this environment can stall without them.  A minimal repeated high-order
# autograd reproduction completed 6/6 runs with these settings, versus stalls
# on the third run without them.  They also make each worker single-threaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as torch_functional
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
APPROACH1_PATH = ROOT / "code" / "approach1_q1a_svi.py"
DEFAULT_OPTIONS = ROOT / "data" / "clean" / "options_io.csv"
DEFAULT_SHIBOR = ROOT / "data" / "clean" / "shibor.csv"
DEFAULT_METADATA = ROOT / "data" / "metadata.json"
DEFAULT_APPROACH1_DIR = ROOT / "results" / "approach1" / "io"
DEFAULT_OUT = ROOT / "results" / "approach2" / "io"
DEFAULT_LOG = ROOT / ".cache" / "approach2_q1b.log"

LAMBDA_VALUES = (0.0, 0.01, 1.0)
COLLOC_K_START = -0.14
COLLOC_K_STOP = 0.14
DTYPE = torch.float64
CACHE_VERSION = "q1b_v1"


def load_approach1():
    spec = importlib.util.spec_from_file_location("approach1_q1a_svi", APPROACH1_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {APPROACH1_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A1 = load_approach1()
GRID_K = A1.GRID_K
SURFACE_K = A1.SURFACE_K
BUTTERFLY_TOL = A1.BUTTERFLY_TOL
CALENDAR_TOL = A1.CALENDAR_TOL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 Option B: differentiable SVI and neural surface")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--price-col", choices=["close", "settle"], default="settle")
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--kband", type=float, default=0.20)
    parser.add_argument("--min-days", type=int, default=7)
    parser.add_argument("--min-strikes", type=int, default=6)
    parser.add_argument("--eta-min", type=float, default=A1.DEFAULT_ETA_MIN)
    parser.add_argument("--eta-max", type=float, default=A1.DEFAULT_ETA_MAX)

    parser.add_argument("--svi-adam-steps", type=int, default=1200)
    parser.add_argument("--svi-lbfgs-steps", type=int, default=300)
    parser.add_argument("--svi-adam-lr", type=float, default=0.03)
    parser.add_argument("--nn-adam-steps", type=int, default=1500)
    parser.add_argument("--nn-lbfgs-steps", type=int, default=100)
    parser.add_argument("--nn-lbfgs-evals", type=int, default=300)
    parser.add_argument("--nn-lbfgs-seconds", type=float, default=30.0)
    parser.add_argument("--nn-lbfgs-history", type=int, default=20)
    parser.add_argument("--nn-adam-lr", type=float, default=0.01)
    parser.add_argument("--fit-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--fit-retries", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "approach2_cache")
    parser.add_argument("--nn-hidden", type=int, default=24)
    parser.add_argument("--nn-hidden-layers", type=int, default=3)
    parser.add_argument("--colloc-k-points", type=int, default=29)
    parser.add_argument("--colloc-t-points", type=int, default=17)
    parser.add_argument("--robust-seeds", type=int, default=3)
    parser.add_argument("--drop-fraction", type=float, default=0.30)
    parser.add_argument("--wing-fraction", type=float, default=0.15)
    parser.add_argument("--compare-tol-relative", type=float, default=1e-3)
    parser.add_argument("--max-runtime-minutes", type=float, default=45.0)

    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--shibor", type=Path, default=DEFAULT_SHIBOR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--approach1-dir", type=Path, default=DEFAULT_APPROACH1_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_torch_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError("--torch-threads must be >= 1")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


def ensure_inside_repo(path: Path) -> Path:
    root = ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must stay inside {root}: {resolved}")
    return resolved


class RunLog:
    def __init__(self, path: Path, console: Console) -> None:
        self.path = ensure_inside_repo(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.console = console
        self.start = time.perf_counter()
        self.lines: list[str] = []

    def write(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start
        line = f"[{elapsed:8.2f}s] {message}"
        self.lines.append(line)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def check_runtime(self, limit_minutes: float, phase: str) -> None:
        elapsed = time.perf_counter() - self.start
        if elapsed > limit_minutes * 60.0:
            raise TimeoutError(
                f"runtime guard: {phase} exceeded {limit_minutes:.1f} minutes "
                f"(elapsed {elapsed / 60.0:.1f} min); stop and report before cutting scope"
            )


def select_sample_dates(options: pd.DataFrame, metadata: dict) -> tuple[list[pd.Timestamp], list[dict]]:
    metadata_dates = [pd.Timestamp(item) for item in metadata.get("sample_dates", [])]
    months = sorted(dict.fromkeys(item.strftime("%Y-%m") for item in metadata_dates))
    if len(months) != 6:
        raise ValueError(f"expected 6 sample months from metadata, got {months}")
    counts = options.groupby([options.date.dt.strftime("%Y-%m"), "date"]).size()
    selected: list[pd.Timestamp] = []
    rows: list[dict] = []
    for month in months:
        series = counts.loc[month]
        maximum = int(series.max())
        threshold = 0.8 * maximum
        eligible = series[series >= threshold]
        mid = pd.Timestamp(f"{month}-15")
        eligible_dates = [pd.Timestamp(item) for item in eligible.index]
        chosen = min(eligible_dates, key=lambda item: (abs((item - mid).days), item))
        metadata_date = next((item for item in metadata_dates if item.strftime("%Y-%m") == month), None)
        selected.append(chosen)
        rows.append({
            "month": month,
            "max_quotes": maximum,
            "threshold_80pct": threshold,
            "eligible_dates": len(eligible),
            "date": chosen.strftime("%Y-%m-%d"),
            "date_rows": int(series.loc[chosen]),
            "days_from_mid_month": int(abs((chosen - mid).days)),
            "metadata_date": metadata_date.strftime("%Y-%m-%d") if metadata_date is not None else None,
            "matches_metadata": bool(metadata_date is not None and metadata_date == chosen),
        })
    return selected, rows


def build_otm_quotes(
    data: pd.DataFrame,
    iv: pd.DataFrame,
    sample_dates: list[pd.Timestamp],
    min_strikes: int,
) -> tuple[pd.DataFrame, dict]:
    liquid = data[data.liquid].copy()
    merged = liquid.merge(
        iv[["date", "expiry", "K", "cp", "iv", "status"]],
        on=["date", "expiry", "K", "cp"],
        how="left",
    )
    otm = merged[
        ((merged.cp == "C") & (merged.K >= merged.F))
        | ((merged.cp == "P") & (merged.K < merged.F))
    ].copy()
    otm = otm[(otm.status == "ok") & np.isfinite(otm.iv) & (otm.iv > 0)].copy()
    otm["w"] = otm.iv ** 2 * otm["T"]
    otm["unit_price"] = otm.price / (otm.DF * otm.F)
    otm = otm.sort_values(["date", "expiry", "K", "cp"]).drop_duplicates(["date", "expiry", "K"], keep="first")
    sample_set = set(sample_dates)
    otm = otm[otm.date.isin(sample_set)].copy()

    counts = otm.groupby(["date", "expiry"]).size()
    valid = set(counts[counts >= min_strikes].index)
    skipped = []
    for (date, expiry), count in counts.items():
        if count < min_strikes:
            skipped.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "expiry": pd.Timestamp(expiry).strftime("%Y-%m-%d"),
                "n_quotes": int(count),
                "reason": "fewer_than_min_strikes",
            })
    otm["keep_expiry"] = [
        (pd.Timestamp(date), pd.Timestamp(expiry)) in valid
        for date, expiry in zip(otm.date, otm.expiry)
    ]
    otm = otm[otm.keep_expiry].copy()
    otm["date_str"] = otm.date.dt.strftime("%Y-%m-%d")
    otm["expiry_str"] = otm.expiry.dt.strftime("%Y-%m-%d")
    info = {
        "otm_quotes": int(len(otm)),
        "date_expiry_fits": int(otm.groupby(["date", "expiry"]).ngroups),
        "skipped_expiries": skipped,
        "quotes_by_date": {
            row["date"]: int(row["n_quotes"])
            for _, row in otm.groupby("date", sort=True).size().rename("n_quotes").reset_index().iterrows()
        },
    }
    return otm, info


def date_quotes(otm: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    group = otm[otm.date == date].sort_values(["T", "K", "cp"]).copy()
    if group.empty:
        raise ValueError(f"no OTM quotes for sample date {date}")
    return group.reset_index(drop=True)


FIT_ARG_KEYS = (
    "nn_adam_steps",
    "nn_lbfgs_steps",
    "nn_lbfgs_evals",
    "nn_lbfgs_seconds",
    "nn_lbfgs_history",
    "nn_adam_lr",
    "nn_hidden",
    "nn_hidden_layers",
    "colloc_k_points",
    "colloc_t_points",
)


def derive_seed(base_seed: int, date: str, lambda1: float, lambda2: float, design: str, rep) -> int:
    text = f"{int(base_seed)}|{date}|{float(lambda1):.12g}|{float(lambda2):.12g}|{design}|{rep}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def fit_args_dict(args: argparse.Namespace) -> dict:
    return {key: getattr(args, key) for key in FIT_ARG_KEYS}


def group_fingerprint(group: pd.DataFrame) -> str:
    payload = {
        "date": str(group.date_str.iloc[0]) if "date_str" in group.columns else "",
        "expiries": sorted(str(item) for item in group.expiry_str.unique()) if "expiry_str" in group.columns else [],
        "k": np.round(group.k.to_numpy(dtype=float), 12).tolist(),
        "T": np.round(group["T"].to_numpy(dtype=float), 12).tolist(),
        "cp": group.cp.tolist(),
        "unit_price": np.round(group.unit_price.to_numpy(dtype=float), 14).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def neural_cache_key(
    date: str,
    lambda1: float,
    lambda2: float,
    seed: int,
    design: str,
    rep,
    args: argparse.Namespace,
    fingerprint: str,
) -> str:
    config = {
        "version": CACHE_VERSION,
        "date": date,
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "seed": int(seed),
        "design": design,
        "rep": rep,
        "args": fit_args_dict(args),
        "fingerprint": fingerprint,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _inv_softplus(value: float) -> float:
    value = float(value)
    if value <= 1e-12:
        return -30.0
    if value > 30.0:
        return value
    return math.log(math.expm1(value))


class TorchSVI(torch.nn.Module):
    """Raw SVI with b=softplus(u), rho=tanh(v), eta=softplus(s)."""

    def __init__(self, params: np.ndarray | list[float]) -> None:
        super().__init__()
        a, b, rho, m, eta = (float(item) for item in params)
        b = max(b, 1e-10)
        eta = max(eta, 1e-10)
        rho = min(max(rho, -0.999999), 0.999999)
        raw = [a, _inv_softplus(b), math.atanh(rho), m, _inv_softplus(eta)]
        self.raw = torch.nn.Parameter(torch.tensor(raw, dtype=DTYPE))

    def params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, u, v, m, s = self.raw.unbind()
        b = torch_functional.softplus(u) + 1e-12
        rho = torch.tanh(v)
        eta = torch_functional.softplus(s) + 1e-12
        return a, b, rho, m, eta

    def w(self, k: torch.Tensor) -> torch.Tensor:
        a, b, rho, m, eta = self.params()
        d = k - m
        return a + b * (rho * d + torch.sqrt(d * d + eta * eta))

    def numpy_params(self) -> np.ndarray:
        with torch.no_grad():
            return np.array([item.item() for item in self.params()], dtype=float)


def torch_svi_g(k: torch.Tensor, model: TorchSVI) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b, rho, m, eta = model.params()
    d = k - m
    s = torch.sqrt(d * d + eta * eta)
    w = a + b * (rho * d + s)
    w1 = b * (rho + d / s)
    w2 = b * eta * eta / (s * s * s)
    g = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 1.0 / 4.0) + w2 / 2.0
    return w, w1, w2, g


def svi_objective_mse(k: np.ndarray, w: np.ndarray, params: np.ndarray) -> float:
    return float(np.mean((A1.svi_w(k, params) - w) ** 2))


def fit_svi_torch(
    k: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    adam_steps: int,
    lbfgs_steps: int,
    lr: float,
    seed: int,
) -> dict:
    set_seed(seed)
    k_tensor = torch.tensor(np.asarray(k, dtype=float), dtype=DTYPE)
    w_tensor = torch.tensor(np.asarray(w, dtype=float), dtype=DTYPE)
    model = TorchSVI(x0)
    initial_loss = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    history: list[float] = [initial_loss]
    phases: list[str] = ["initial"]
    best_loss = initial_loss
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    start = time.perf_counter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(adam_steps, 1), eta_min=lr * 1e-3)
    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model.w(k_tensor) - w_tensor) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        value = float(loss.detach())
        history.append(value)
        phases.append("adam")
        if value < best_loss:
            best_loss = value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if value < 1e-30:
            break
    adam_evals = len(history)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=lbfgs_steps,
        max_eval=max(2 * lbfgs_steps, 20),
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
    )

    def closure() -> torch.Tensor:
        nonlocal best_loss, best_state
        optimizer_lbfgs.zero_grad(set_to_none=True)
        loss = torch.mean((model.w(k_tensor) - w_tensor) ** 2)
        loss.backward()
        value = float(loss.detach())
        history.append(value)
        phases.append("lbfgs")
        if value < best_loss:
            best_loss = value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        return loss

    try:
        optimizer_lbfgs.step(closure)
    except (RuntimeError, ValueError):
        pass
    final_loss = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    if not math.isfinite(final_loss) or final_loss > best_loss:
        if best_state is not None:
            model.load_state_dict(best_state)
        final_loss = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    wall = time.perf_counter() - start
    return {
        "params": model.numpy_params(),
        "objective": final_loss,
        "history": history,
        "phases": phases,
        "adam_evals": adam_evals,
        "lbfgs_evals": len(history) - adam_evals,
        "wall_seconds": wall,
        "iterations": len(history) - 1,
    }


def fit_svi_scipy(k: np.ndarray, w: np.ndarray, x0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> dict:
    start = time.perf_counter()

    def residual(params: np.ndarray) -> np.ndarray:
        return A1.svi_w(k, params) - w

    sol = least_squares(
        residual,
        x0,
        bounds=bounds,
        method="trf",
        x_scale="jac",
        max_nfev=4000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    wall = time.perf_counter() - start
    params = np.asarray(sol.x, dtype=float)
    return {
        "params": params,
        "objective": svi_objective_mse(k, w, params),
        "nfev": int(sol.nfev),
        "wall_seconds": wall,
        "success": bool(sol.success),
        "message": str(sol.message),
    }


def load_reference_starts(
    approach1_dir: Path,
    otm: pd.DataFrame,
    sample_dates: list[pd.Timestamp],
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[dict, dict]:
    frame, availability = approach1_params(approach1_dir)
    starts: dict[tuple[str, str], dict] = {}
    if frame is not None:
        for row in frame.itertuples(index=False):
            starts[(row.date, row.expiry)] = {
                "raw": np.array([row.a_raw, row.b_raw, row.rho_raw, row.m_raw, row.eta_raw], dtype=float),
                "repaired": np.array([row.a_repaired, row.b_repaired, row.rho_repaired, row.m_repaired, row.eta_repaired], dtype=float),
                "source": "approach1_svi_params_raw",
            }
    computed = 0
    missing = []
    sample_keys = set()
    for date in sample_dates:
        group = date_quotes(otm, date)
        for expiry, slice_group in group.groupby("expiry", sort=True):
            key = (pd.Timestamp(date).strftime("%Y-%m-%d"), pd.Timestamp(expiry).strftime("%Y-%m-%d"))
            sample_keys.add(key)
            if key in starts:
                continue
            fit = A1.fit_svi(
                slice_group.k.to_numpy(dtype=float),
                slice_group.w.to_numpy(dtype=float),
                bounds=bounds,
            )
            if fit is None:
                missing.append({"date": key[0], "expiry": key[1]})
                continue
            starts[key] = {
                "raw": np.asarray(fit["params"], dtype=float),
                "repaired": np.asarray(fit["params"], dtype=float),
                "source": f"computed_{fit['start']}",
            }
            computed += 1
    info = {
        "available": availability,
        "from_approach1": int(sum(1 for key, item in starts.items() if key in sample_keys and item["source"] == "approach1_svi_params_raw")),
        "computed": int(computed),
        "missing": missing,
        "note": "Both solvers start from the same full-sample SVI parameters: approach1's best raw fit when available, otherwise a local multi-start fit.",
    }
    return starts, info


def run_svi_solver_comparison(
    otm: pd.DataFrame,
    sample_dates: list[pd.Timestamp],
    args: argparse.Namespace,
    reference_starts: dict,
    log: RunLog,
) -> tuple[pd.DataFrame, list[dict], dict]:
    bounds = A1.svi_bounds(args.eta_min, args.eta_max)
    rows: list[dict] = []
    histories: list[dict] = []
    for date in sample_dates:
        group = date_quotes(otm, date)
        for expiry, slice_group in group.groupby("expiry", sort=True):
            k = slice_group.k.to_numpy(dtype=float)
            w = slice_group.w.to_numpy(dtype=float)
            key = (pd.Timestamp(date).strftime("%Y-%m-%d"), pd.Timestamp(expiry).strftime("%Y-%m-%d"))
            if key not in reference_starts:
                continue
            x0 = np.asarray(reference_starts[key]["raw"], dtype=float)
            torch_fit = fit_svi_torch(
                k, w, x0,
                adam_steps=args.svi_adam_steps,
                lbfgs_steps=args.svi_lbfgs_steps,
                lr=args.svi_adam_lr,
                seed=args.seed,
            )
            scipy_fit = fit_svi_scipy(k, w, x0, bounds)
            best_objective = min(torch_fit["objective"], scipy_fit["objective"])
            tolerance = best_objective * (1.0 + args.compare_tol_relative) + 1e-15
            first_tol = next(
                (index for index, value in enumerate(torch_fit["history"]) if value <= tolerance),
                None,
            )
            torch_reached = bool(torch_fit["objective"] <= tolerance)
            scipy_reached = bool(scipy_fit["objective"] <= tolerance)
            row = {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "expiry": pd.Timestamp(expiry).strftime("%Y-%m-%d"),
                "T": float(slice_group["T"].iloc[0]),
                "n_quotes": int(len(slice_group)),
                "start": reference_starts[key]["source"],
                "start_objective": svi_objective_mse(k, w, x0),
                "reference_objective": svi_objective_mse(k, w, x0),
                "torch_objective": float(torch_fit["objective"]),
                "scipy_objective": float(scipy_fit["objective"]),
                "best_objective": float(best_objective),
                "tolerance_objective": float(tolerance),
                "torch_reached": torch_reached,
                "scipy_reached": scipy_reached,
                "torch_iterations_to_tolerance": int(first_tol) if first_tol is not None else None,
                "torch_total_iterations": int(torch_fit["iterations"]),
                "torch_adam_evals": int(torch_fit["adam_evals"]),
                "torch_lbfgs_evals": int(torch_fit["lbfgs_evals"]),
                "torch_wall_seconds": float(torch_fit["wall_seconds"]),
                "scipy_nfev": int(scipy_fit["nfev"]),
                "scipy_iterations_to_tolerance": int(scipy_fit["nfev"]) if scipy_reached else None,
                "scipy_wall_seconds": float(scipy_fit["wall_seconds"]),
                "torch_a": float(torch_fit["params"][0]),
                "torch_b": float(torch_fit["params"][1]),
                "torch_rho": float(torch_fit["params"][2]),
                "torch_m": float(torch_fit["params"][3]),
                "torch_eta": float(torch_fit["params"][4]),
                "scipy_a": float(scipy_fit["params"][0]),
                "scipy_b": float(scipy_fit["params"][1]),
                "scipy_rho": float(scipy_fit["params"][2]),
                "scipy_m": float(scipy_fit["params"][3]),
                "scipy_eta": float(scipy_fit["params"][4]),
            }
            rows.append(row)
            histories.append({
                "date": row["date"],
                "expiry": row["expiry"],
                "T": row["T"],
                "history": torch_fit["history"],
                "adam_evals": torch_fit["adam_evals"],
                "scipy_objective": float(scipy_fit["objective"]),
                "best_objective": float(best_objective),
                "tolerance_objective": float(tolerance),
            })
            log.write(
                f"svi {row['date']} {row['expiry']} torch={row['torch_objective']:.3e} "
                f"scipy={row['scipy_objective']:.3e} tol={tolerance:.3e}"
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no SVI solver comparisons were produced")
    summary = {
        "fits": int(len(frame)),
        "torch_reached": int(frame.torch_reached.sum()),
        "scipy_reached": int(frame.scipy_reached.sum()),
        "both_reached": int((frame.torch_reached & frame.scipy_reached).sum()),
        "torch_objective_median": float(frame.torch_objective.median()),
        "scipy_objective_median": float(frame.scipy_objective.median()),
        "best_objective_median": float(frame.best_objective.median()),
        "torch_objective_max": float(frame.torch_objective.max()),
        "scipy_objective_max": float(frame.scipy_objective.max()),
        "torch_relative_gap_median": float(((frame.torch_objective - frame.best_objective) / frame.best_objective).median()),
        "scipy_relative_gap_median": float(((frame.scipy_objective - frame.best_objective) / frame.best_objective).median()),
        "torch_iterations_to_tolerance_median": float(frame.torch_iterations_to_tolerance.median()),
        "scipy_nfev_median": float(frame.scipy_nfev.median()),
        "torch_wall_seconds_median": float(frame.torch_wall_seconds.median()),
        "scipy_wall_seconds_median": float(frame.scipy_wall_seconds.median()),
        "torch_wall_seconds_total": float(frame.torch_wall_seconds.sum()),
        "scipy_wall_seconds_total": float(frame.scipy_wall_seconds.sum()),
        "tolerance_relative": float(args.compare_tol_relative),
        "protocol": "Both solvers start from the same full-sample SVI parameters: approach1 raw fit from results/approach1/io/svi_params.csv when available, otherwise a computed multi-start fit.",
    }
    return frame, histories, summary


class IVSurface(torch.nn.Module):
    """Smooth MLP: inputs k/0.2 and sqrt(T)/sqrt(T_max), output softplus(sigma)."""

    def __init__(self, hidden: int = 24, hidden_layers: int = 3) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[torch.nn.Module] = []
        width = 2
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(width, hidden, dtype=DTYPE))
            layers.append(torch.nn.Tanh())
            width = hidden
        layers.append(torch.nn.Linear(width, 1, dtype=DTYPE))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, k: torch.Tensor, T: torch.Tensor, t_scale: float) -> torch.Tensor:
        inputs = torch.stack([k / 0.20, torch.sqrt(T) / t_scale], dim=-1)
        raw = self.net(inputs).squeeze(-1)
        return torch_functional.softplus(raw) + 1e-6


def black_otm_unit(k: torch.Tensor, T: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Undiscounted OTM Black price with forward 1, strike exp(k)."""
    sqrt_t = torch.sqrt(T)
    d1 = -k / (sigma * sqrt_t) + 0.5 * sigma * sqrt_t
    d2 = d1 - sigma * sqrt_t
    call = torch.special.ndtr(d1) - torch.exp(k) * torch.special.ndtr(d2)
    put = torch.exp(k) * torch.special.ndtr(-d2) - torch.special.ndtr(-d1)
    return torch.where(k >= 0.0, call, put)


def bs_price_torch(cp: str, F: float, K: float, T: float, DF: float, sigma: torch.Tensor | float) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, dtype=DTYPE)
    T_tensor = torch.as_tensor(T, dtype=DTYPE)
    sqrt_t = torch.sqrt(T_tensor)
    d1 = (math.log(F / K) + 0.5 * sigma_tensor * sigma_tensor * T_tensor) / (sigma_tensor * sqrt_t)
    d2 = d1 - sigma_tensor * sqrt_t
    if cp == "C":
        return DF * (F * torch.special.ndtr(d1) - K * torch.special.ndtr(d2))
    return DF * (K * torch.special.ndtr(-d2) - F * torch.special.ndtr(-d1))


def neural_w_derivatives(
    model: IVSurface,
    k: torch.Tensor,
    T: torch.Tensor,
    t_scale: float,
    need_k: bool = True,
    need_T: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """w, w_k, w_kk and w_T by torch.func autograd (safe for repeated fits)."""

    def w_point(k_point: torch.Tensor, T_point: torch.Tensor) -> torch.Tensor:
        sigma = model(k_point, T_point, t_scale)
        return sigma * sigma * T_point

    w = torch.func.vmap(w_point)(k, T)
    w_k = None
    w_kk = None
    w_T = None
    if need_k:
        first = torch.func.grad(w_point, argnums=0)
        second = torch.func.grad(first, argnums=0)
        w_k = torch.func.vmap(first)(k, T)
        w_kk = torch.func.vmap(second)(k, T)
    if need_T:
        w_T = torch.func.vmap(torch.func.grad(w_point, argnums=1))(k, T)
    return w, w_k, w_kk, w_T


def neural_loss(
    model: IVSurface,
    k_quotes: torch.Tensor,
    T_quotes: torch.Tensor,
    unit_market: torch.Tensor,
    colloc_k: torch.Tensor,
    colloc_T: torch.Tensor,
    t_scale: float,
    lambda1: float,
    lambda2: float,
) -> tuple[torch.Tensor, float, float, float]:
    sigma = model(k_quotes, T_quotes, t_scale)
    unit_model = black_otm_unit(k_quotes, T_quotes, sigma)
    price_loss = torch.mean((unit_model - unit_market) ** 2)
    reg_butterfly = torch.zeros((), dtype=DTYPE)
    reg_calendar = torch.zeros((), dtype=DTYPE)
    need_butterfly = lambda1 != 0.0
    need_calendar = lambda2 != 0.0
    if need_butterfly or need_calendar:
        w, w_k, w_kk, w_T = neural_w_derivatives(
            model, colloc_k, colloc_T, t_scale,
            need_k=need_butterfly, need_T=need_calendar,
        )
        if need_butterfly:
            g = (
                (1.0 - colloc_k * w_k / (2.0 * w)) ** 2
                - (w_k * w_k / 4.0) * (1.0 / w + 1.0 / 4.0)
                + w_kk / 2.0
            )
            reg_butterfly = torch.mean(torch.relu(-g) ** 2)
        if need_calendar:
            reg_calendar = torch.mean(torch.relu(-w_T) ** 2)
    total = price_loss + lambda1 * reg_butterfly + lambda2 * reg_calendar
    return total, float(price_loss.detach()), float(reg_butterfly.detach()), float(reg_calendar.detach())


def _flat_parameters(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in params])


def _set_flat_parameters(params: list[torch.nn.Parameter], values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in params:
            count = param.numel()
            param.copy_(values[offset:offset + count].view_as(param))
            offset += count


def _flat_gradient(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.grad.detach().reshape(-1) for param in params])


def _loss_and_gradient(model: IVSurface, loss_function) -> tuple[float, torch.Tensor]:
    params = list(model.parameters())
    for param in params:
        param.grad = None
    loss = loss_function()
    loss.backward()
    return float(loss.detach()), _flat_gradient(params)


def lbfgs_bounded(
    model: IVSurface,
    loss_function,
    max_iter: int,
    max_evals: int,
    max_seconds: float,
    history_size: int,
    tolerance_grad: float = 1e-10,
) -> dict:
    """L-BFGS with two-loop recursion, Armijo backtracking and hard caps."""
    params = list(model.parameters())
    start = time.perf_counter()
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []
    rho_history: list[float] = []

    loss_value, gradient = _loss_and_gradient(model, loss_function)
    evaluations = 1
    iterations = 0
    history = [loss_value]
    best_loss = loss_value
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    stop_reason = "max_iter"

    for _ in range(max_iter):
        if evaluations >= max_evals:
            stop_reason = "max_evals"
            break
        if time.perf_counter() - start > max_seconds:
            stop_reason = "max_seconds"
            break
        if not torch.isfinite(gradient).all():
            stop_reason = "nonfinite_gradient"
            break
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        if gradient_norm <= tolerance_grad:
            stop_reason = "gradient_tolerance"
            break

        q = gradient.clone()
        alphas: list[torch.Tensor] = []
        for s_vec, y_vec, rho in reversed(list(zip(s_history, y_history, rho_history))):
            alpha = rho * torch.dot(s_vec, q)
            q = q - alpha * y_vec
            alphas.append(alpha)
        if s_history:
            last_s = s_history[-1]
            last_y = y_history[-1]
            yy = torch.dot(last_y, last_y)
            gamma = torch.dot(last_s, last_y) / yy if float(yy) > 0.0 else 1.0
        else:
            gamma = 1.0
        direction = gamma * q
        for (s_vec, y_vec, rho), alpha in zip(
            zip(s_history, y_history, rho_history), reversed(alphas)
        ):
            beta = rho * torch.dot(y_vec, direction)
            direction = direction + s_vec * (alpha - beta)
        direction = -direction
        directional_derivative = torch.dot(gradient, direction)
        if not torch.isfinite(directional_derivative) or float(directional_derivative) >= 0.0:
            direction = -gradient
            directional_derivative = -torch.dot(gradient, gradient)

        x_start = _flat_parameters(params)
        step = 1.0
        accepted = False
        new_loss = loss_value
        new_gradient = gradient
        x_new = x_start
        for _ in range(20):
            if evaluations >= max_evals:
                break
            if time.perf_counter() - start > max_seconds:
                break
            x_trial = x_start + step * direction
            _set_flat_parameters(params, x_trial)
            trial_loss, trial_gradient = _loss_and_gradient(model, loss_function)
            evaluations += 1
            history.append(trial_loss)
            if trial_loss < best_loss:
                best_loss = trial_loss
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            if (
                math.isfinite(trial_loss)
                and trial_loss <= loss_value + 1e-4 * step * float(directional_derivative)
            ):
                accepted = True
                new_loss = trial_loss
                new_gradient = trial_gradient
                x_new = x_trial
                break
            step *= 0.5
        if not accepted:
            _set_flat_parameters(params, x_start)
            stop_reason = "line_search_failed"
            break

        s_vec = x_new - x_start
        y_vec = new_gradient - gradient
        sy = torch.dot(s_vec, y_vec)
        if float(sy) > 1e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            rho_history.append(1.0 / float(sy))
            if len(s_history) > history_size:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
        loss_value = new_loss
        gradient = new_gradient
        iterations += 1

    if evaluations >= max_evals:
        stop_reason = "max_evals"
    if time.perf_counter() - start > max_seconds:
        stop_reason = "max_seconds"
    model.load_state_dict(best_state)
    return {
        "loss": best_loss,
        "history": history,
        "iterations": iterations,
        "evaluations": evaluations,
        "seconds": time.perf_counter() - start,
        "stop_reason": stop_reason,
    }


def fit_neural_surface(
    group: pd.DataFrame,
    args: argparse.Namespace,
    lambda1: float,
    lambda2: float,
    seed: int,
) -> tuple[IVSurface, dict]:
    set_seed(seed)
    t_scale = math.sqrt(float(group["T"].max()))
    k_quotes = torch.tensor(group.k.to_numpy(dtype=float), dtype=DTYPE)
    T_quotes = torch.tensor(group["T"].to_numpy(dtype=float), dtype=DTYPE)
    unit_market = torch.tensor(group.unit_price.to_numpy(dtype=float), dtype=DTYPE)
    expiry_T = np.sort(group["T"].unique().astype(float))
    k_grid = np.linspace(COLLOC_K_START, COLLOC_K_STOP, args.colloc_k_points)
    T_grid = np.unique(np.concatenate([
        expiry_T,
        np.linspace(float(expiry_T.min()), float(expiry_T.max()), args.colloc_t_points),
    ]))
    colloc_k = torch.tensor(np.tile(k_grid, len(T_grid)), dtype=DTYPE)
    colloc_T = torch.tensor(np.repeat(T_grid, len(k_grid)), dtype=DTYPE)

    model = IVSurface(hidden=args.nn_hidden, hidden_layers=args.nn_hidden_layers)
    initial = neural_loss(
        model, k_quotes, T_quotes, unit_market, colloc_k, colloc_T,
        t_scale, lambda1, lambda2,
    )
    initial_loss = float(initial[0].detach())
    history: list[float] = [initial_loss]
    price_history: list[float] = [initial[1]]
    best_loss = initial_loss
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    start = time.perf_counter()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.nn_adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.nn_adam_steps, 1), eta_min=args.nn_adam_lr * 1e-3
    )
    for _ in range(args.nn_adam_steps):
        optimizer.zero_grad(set_to_none=True)
        total, price_loss, _, _ = neural_loss(
            model, k_quotes, T_quotes, unit_market, colloc_k, colloc_T,
            t_scale, lambda1, lambda2,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        scheduler.step()
        value = float(total.detach())
        history.append(value)
        price_history.append(price_loss)
        if value < best_loss:
            best_loss = value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if value < 1e-18:
            break
    adam_evals = len(history)
    if best_state is not None:
        model.load_state_dict(best_state)

    def loss_function() -> torch.Tensor:
        return neural_loss(
            model, k_quotes, T_quotes, unit_market, colloc_k, colloc_T,
            t_scale, lambda1, lambda2,
        )[0]

    lbfgs_result = lbfgs_bounded(
        model,
        loss_function,
        max_iter=args.nn_lbfgs_steps,
        max_evals=args.nn_lbfgs_evals,
        max_seconds=args.nn_lbfgs_seconds,
        history_size=args.nn_lbfgs_history,
    )
    history.extend(lbfgs_result["history"])
    final_loss = float(neural_loss(
        model, k_quotes, T_quotes, unit_market, colloc_k, colloc_T,
        t_scale, lambda1, lambda2,
    )[0].detach())
    model.eval()
    wall = time.perf_counter() - start
    info = {
        "t_scale": t_scale,
        "loss": final_loss,
        "adam_evals": int(adam_evals),
        "lbfgs_evals": int(lbfgs_result["evaluations"]),
        "lbfgs_iterations": int(lbfgs_result["iterations"]),
        "lbfgs_stop_reason": lbfgs_result["stop_reason"],
        "lbfgs_seconds": float(lbfgs_result["seconds"]),
        "iterations": int(adam_evals - 1 + lbfgs_result["evaluations"]),
        "wall_seconds": float(wall),
        "history": history,
        "price_history": price_history,
        "colloc_points": int(len(colloc_k)),
        "expiry_T": expiry_T.tolist(),
    }
    return model, info


CACHE_INFO_KEYS = (
    "t_scale",
    "loss",
    "adam_evals",
    "lbfgs_evals",
    "lbfgs_iterations",
    "lbfgs_stop_reason",
    "lbfgs_seconds",
    "iterations",
    "wall_seconds",
    "colloc_points",
    "expiry_T",
)


def _cache_info(info: dict) -> dict:
    return {key: info[key] for key in CACHE_INFO_KEYS if key in info}


def _neural_fit_worker(payload: dict, queue) -> None:
    """Fresh spawned worker: one neural fit, then save it to the cache."""
    set_torch_threads(1)
    key = payload["cache_key"]
    try:
        group = pd.DataFrame(payload["records"])
        args = argparse.Namespace(**payload["args"])
        model, info = fit_neural_surface(
            group,
            args,
            payload["lambda1"],
            payload["lambda2"],
            payload["seed"],
        )
        small_info = _cache_info(info)
        cache_dir = Path(payload["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        state_path = cache_dir / f"{key}.pt"
        meta_path = cache_dir / f"{key}.json"
        state_tmp = cache_dir / f".{key}.pt.tmp-{os.getpid()}"
        meta_tmp = cache_dir / f".{key}.json.tmp-{os.getpid()}"
        torch.save(model.state_dict(), state_tmp)
        os.replace(state_tmp, state_path)
        metadata = {
            "cache_key": key,
            "config": payload["config"],
            "seed": int(payload["seed"]),
            "info": small_info,
            "created_at": time.time(),
        }
        meta_tmp.write_text(json.dumps(A1.to_jsonable(metadata), indent=2, allow_nan=False), encoding="utf-8")
        os.replace(meta_tmp, meta_path)
        queue.put({"status": "ok", "cache_key": key, "seed": int(payload["seed"]), "info": small_info})
    except BaseException as exc:  # noqa: BLE001 - failures must be reported, not lost in the worker
        queue.put({"status": "failed", "cache_key": key, "seed": int(payload.get("seed", -1)), "error": repr(exc)})


def _run_worker_process(payload: dict, timeout: float) -> tuple[dict | None, str | None]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_neural_fit_worker, args=(payload, queue))
    process.start()
    process.join(timeout)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
    result = None
    try:
        result = queue.get(timeout=5)
    except Exception:
        result = None
    try:
        queue.close()
        queue.join_thread()
    except Exception:
        pass
    if timed_out:
        return None, "timeout"
    if result is None:
        return None, f"worker_exit_{process.exitcode}"
    return result, None


def load_cached_fit(cache_dir: Path, key: str) -> tuple[IVSurface | None, dict | None]:
    state_path = cache_dir / f"{key}.pt"
    meta_path = cache_dir / f"{key}.json"
    if not state_path.exists() or not meta_path.exists():
        return None, None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        config = metadata["config"]
        model = IVSurface(
            hidden=int(config["args"]["nn_hidden"]),
            hidden_layers=int(config["args"]["nn_hidden_layers"]),
        )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        info = dict(metadata["info"])
        info["from_cache"] = True
        info["cache_key"] = key
        info["seed"] = int(metadata["seed"])
        return model, info
    except Exception:
        return None, None


def run_neural_fit_cached(
    group: pd.DataFrame,
    args: argparse.Namespace,
    lambda1: float,
    lambda2: float,
    seed: int,
    design: str,
    rep,
    cache_dir: Path,
    timeout: float,
    retries: int,
    failed_fits: list[dict],
    cache_stats: dict,
    force_recompute: bool = False,
) -> dict:
    date_str = str(group.date_str.iloc[0]) if "date_str" in group.columns else str(group.date.iloc[0])
    fingerprint = group_fingerprint(group)
    key = neural_cache_key(date_str, lambda1, lambda2, seed, design, rep, args, fingerprint)
    config = {
        "date": date_str,
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "seed": int(seed),
        "design": design,
        "rep": rep,
        "args": fit_args_dict(args),
        "fingerprint": fingerprint,
        "cache_key": key,
    }
    if not force_recompute:
        model, info = load_cached_fit(cache_dir, key)
        if model is not None:
            cache_stats["hits"] = cache_stats.get("hits", 0) + 1
            return {
                "status": "ok",
                "model": model,
                "info": info,
                "cache_key": key,
                "seed": int(seed),
                "attempts": 0,
                "from_cache": True,
                "config": config,
            }
    cache_stats["misses"] = cache_stats.get("misses", 0) + 1
    records = group[["k", "T", "cp", "price", "unit_price"]].to_dict(orient="records")
    payload = {
        "records": records,
        "args": fit_args_dict(args),
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "seed": int(seed),
        "cache_dir": str(cache_dir),
        "cache_key": key,
        "config": config,
    }
    last_error = None
    for attempt in range(1 + max(retries, 0)):
        result, error = _run_worker_process(payload, timeout)
        model, info = load_cached_fit(cache_dir, key)
        if model is not None:
            return {
                "status": "ok",
                "model": model,
                "info": info,
                "cache_key": key,
                "seed": int(seed),
                "attempts": attempt + 1,
                "from_cache": False,
                "config": config,
            }
        if result is not None and result.get("status") == "failed":
            last_error = result.get("error", "worker_failed")
        elif error is not None:
            last_error = error
            if error == "timeout":
                cache_stats["timeouts"] = cache_stats.get("timeouts", 0) + 1
        else:
            last_error = "cache_missing_after_worker"
        if attempt < retries:
            cache_stats["retries"] = cache_stats.get("retries", 0) + 1
    failure = {
        **config,
        "attempts": 1 + max(retries, 0),
        "error": last_error or "unknown_failure",
    }
    failed_fits.append(failure)
    cache_stats["failed"] = cache_stats.get("failed", 0) + 1
    return {
        "status": "failed",
        "cache_key": key,
        "seed": int(seed),
        "attempts": 1 + max(retries, 0),
        "error": last_error or "unknown_failure",
        "config": config,
    }


def neural_sigma_numpy(model: IVSurface, k: np.ndarray, T: np.ndarray, t_scale: float) -> np.ndarray:
    k_tensor = torch.tensor(np.asarray(k, dtype=float), dtype=DTYPE)
    T_tensor = torch.tensor(np.asarray(T, dtype=float), dtype=DTYPE)
    with torch.no_grad():
        sigma = model(k_tensor, T_tensor, t_scale)
    return sigma.detach().numpy()


def neural_repricing_errors(
    model: IVSurface,
    group: pd.DataFrame,
    t_scale: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    sigma = neural_sigma_numpy(model, group.k.to_numpy(dtype=float), group["T"].to_numpy(dtype=float), t_scale)
    model_prices = np.array([
        A1.bs_price(cp, float(F), float(K), float(T), float(DF), float(vol))
        for cp, F, K, T, DF, vol in zip(group.cp, group.F, group.K, group["T"], group.DF, sigma)
    ])
    market = group.price.to_numpy(dtype=float)
    valid = np.isfinite(model_prices) & np.isfinite(sigma) & (market > 0)
    rel = np.abs(model_prices[valid] - market[valid]) / market[valid]
    frame = group.loc[valid, ["date_str", "expiry_str", "k", "T", "cp", "price"]].copy()
    frame["model_price"] = model_prices[valid]
    frame["rel_error"] = rel
    return rel, frame


def svi_repricing_errors(params: np.ndarray, group: pd.DataFrame) -> np.ndarray:
    k = group.k.to_numpy(dtype=float)
    T = group["T"].to_numpy(dtype=float)
    w = A1.svi_w(k, params)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt(w / T)
    model_prices = np.array([
        A1.bs_price(cp, float(F), float(K), float(t), float(DF), float(vol))
        for cp, F, K, t, DF, vol in zip(group.cp, group.F, group.K, T, group.DF, sigma)
    ])
    market = group.price.to_numpy(dtype=float)
    valid = np.isfinite(model_prices) & np.isfinite(sigma) & (market > 0)
    return np.abs(model_prices[valid] - market[valid]) / market[valid]


def neural_audit(model: IVSurface, expiry_T: np.ndarray, t_scale: float, tol: float = BUTTERFLY_TOL) -> dict:
    Ts = np.sort(np.asarray(expiry_T, dtype=float))
    n_k = len(GRID_K)
    k = torch.tensor(np.tile(GRID_K, len(Ts)), dtype=DTYPE)
    T = torch.tensor(np.repeat(Ts, n_k), dtype=DTYPE)
    w_tensor, w_k, w_kk, _ = neural_w_derivatives(model, k, T, t_scale, need_k=True, need_T=False)
    w = w_tensor.detach().numpy().reshape(len(Ts), n_k)
    w1 = w_k.detach().numpy().reshape(len(Ts), n_k)
    w2 = w_kk.detach().numpy().reshape(len(Ts), n_k)
    k_np = GRID_K.reshape(1, n_k)
    g = (1.0 - k_np * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 1.0 / 4.0) + w2 / 2.0
    diff = w[1:] - w[:-1]
    return {
        "butterfly_violations_strict": int((g < 0.0).sum()),
        "butterfly_violations": int((g < -tol).sum()),
        "min_g": float(np.min(g)),
        "calendar_violations_strict": int((diff < 0.0).sum()),
        "calendar_violations": int((diff < -CALENDAR_TOL).sum()),
        "min_forward_variance": float(np.min(diff / np.diff(Ts)[:, None])) if len(Ts) > 1 else float("nan"),
        "checks_butterfly": int(g.size),
        "checks_calendar": int(diff.size),
    }


def pooled_stats(errors: np.ndarray) -> dict:
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def run_ablation(
    otm: pd.DataFrame,
    sample_dates: list[pd.Timestamp],
    args: argparse.Namespace,
    cache_dir: Path,
    failed_fits: list[dict],
    cache_stats: dict,
    log: RunLog,
) -> tuple[pd.DataFrame, dict, dict, list[dict], dict]:
    models: dict = {}
    rows: list[dict] = []
    error_pool: dict[tuple[float, float], list[float]] = {}
    requested = len(sample_dates) * len(LAMBDA_VALUES) * len(LAMBDA_VALUES)
    for date in sample_dates:
        log.check_runtime(args.max_runtime_minutes, f"ablation {date.date()}")
        group = date_quotes(otm, date)
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        expiry_T = np.sort(group["T"].unique().astype(float))
        for lambda1 in LAMBDA_VALUES:
            for lambda2 in LAMBDA_VALUES:
                seed = derive_seed(args.seed, date_str, lambda1, lambda2, "ablation", 0)
                fit_result = run_neural_fit_cached(
                    group, args, lambda1, lambda2, seed, "ablation", 0,
                    cache_dir, args.fit_timeout_seconds, args.fit_retries,
                    failed_fits, cache_stats,
                )
                if fit_result["status"] != "ok":
                    log.write(
                        f"nn FAILED {date_str} l1={lambda1} l2={lambda2} "
                        f"seed={seed} error={fit_result['error']}"
                    )
                    continue
                model = fit_result["model"]
                info = fit_result["info"]
                rel, _ = neural_repricing_errors(model, group, info["t_scale"])
                audit = neural_audit(model, expiry_T, info["t_scale"])
                models[(date_str, lambda1, lambda2)] = {
                    "model": model,
                    "t_scale": info["t_scale"],
                    "info": info,
                }
                error_pool.setdefault((lambda1, lambda2), []).extend(rel.tolist())
                rows.append({
                    "status": "ok",
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "date": date_str,
                    "seed": int(seed),
                    "from_cache": bool(fit_result["from_cache"]),
                    "attempts": int(fit_result["attempts"]),
                    "n_quotes": int(len(rel)),
                    "median_rel_error": float(np.median(rel)),
                    "max_rel_error": float(np.max(rel)),
                    "p95_rel_error": float(np.quantile(rel, 0.95)),
                    "fit_loss": float(info["loss"]),
                    "fit_seconds": float(info["wall_seconds"]),
                    **{key: value for key, value in audit.items() if key.startswith(("butterfly", "calendar", "min_"))},
                })
                log.write(
                    f"nn {date_str} l1={lambda1} l2={lambda2} seed={seed} "
                    f"median={np.median(rel):.4f} max={np.max(rel):.4f} "
                    f"bf={audit['butterfly_violations']} cal={audit['calendar_violations']}"
                )
    frame = pd.DataFrame(rows)
    aggregate = []
    for lambda1 in LAMBDA_VALUES:
        for lambda2 in LAMBDA_VALUES:
            subset = frame[(frame.lambda1 == lambda1) & (frame.lambda2 == lambda2)]
            pooled = pooled_stats(np.asarray(error_pool.get((lambda1, lambda2), []), dtype=float))
            n_succeeded = int(len(subset))
            aggregate.append({
                "lambda1": lambda1,
                "lambda2": lambda2,
                "n_fits_requested": int(len(sample_dates)),
                "n_fits_succeeded": n_succeeded,
                "n_fits_failed": int(len(sample_dates) - n_succeeded),
                "n_dates": n_succeeded,
                "n_quotes": pooled["n"],
                "median_rel_error": pooled["median"],
                "p95_rel_error": pooled["p95"],
                "max_rel_error": pooled["max"],
                "mean_rel_error": pooled["mean"],
                "butterfly_violations_strict": int(subset.butterfly_violations_strict.sum()) if n_succeeded else 0,
                "butterfly_violations": int(subset.butterfly_violations.sum()) if n_succeeded else 0,
                "calendar_violations_strict": int(subset.calendar_violations_strict.sum()) if n_succeeded else 0,
                "calendar_violations": int(subset.calendar_violations.sum()) if n_succeeded else 0,
                "min_g": float(subset.min_g.min()) if n_succeeded else float("nan"),
                "min_forward_variance": float(subset.min_forward_variance.min()) if n_succeeded else float("nan"),
                "fit_seconds_total": float(subset.fit_seconds.sum()) if n_succeeded else 0.0,
                "fit_seconds_median": float(subset.fit_seconds.median()) if n_succeeded else float("nan"),
            })
    aggregate_frame = pd.DataFrame(aggregate)
    complete = aggregate_frame[
        (aggregate_frame.n_fits_failed == 0)
        & (aggregate_frame.butterfly_violations == 0)
        & (aggregate_frame.calendar_violations == 0)
        & (aggregate_frame.butterfly_violations_strict == 0)
        & (aggregate_frame.calendar_violations_strict == 0)
        & (aggregate_frame.n_fits_succeeded > 0)
    ]
    if len(complete):
        ordered = complete.sort_values(
            ["median_rel_error", "max_rel_error", "lambda1", "lambda2"]
        )
    else:
        ordered = aggregate_frame.sort_values(
            ["n_fits_failed", "butterfly_violations", "calendar_violations",
             "median_rel_error", "max_rel_error"]
        )
    chosen_row = ordered.iloc[0].to_dict()
    chosen = (float(chosen_row["lambda1"]), float(chosen_row["lambda2"]))
    unpenalized = aggregate_frame[
        (aggregate_frame.lambda1 == 0.0) & (aggregate_frame.lambda2 == 0.0)
    ].iloc[0].to_dict()
    selection = {
        "rule": (
            "Choose the lambda pair with complete fit coverage, zero butterfly and calendar "
            "violations at 1e-8 on the approach1 audit grid (strict counts included), and the "
            "smallest pooled in-sample OTM relative repricing median; break ties by pooled max "
            "error, then by (lambda1, lambda2). If no pair has complete coverage and zero "
            "violations, minimize failed fits, then total violations, then repricing error. "
            "Only in-sample ablation data are used; robustness holdouts are never used for selection."
        ),
        "chosen_lambda1": chosen[0],
        "chosen_lambda2": chosen[1],
        "chosen": chosen_row,
        "unpenalized": unpenalized,
        "feasible_pairs": int(len(complete)),
        "coverage": {
            "fits_requested": int(requested),
            "fits_succeeded": int(len(frame)),
            "fits_failed": int(requested - len(frame)),
            "dates_with_complete_coverage": int(
                frame.groupby("date").size().eq(len(LAMBDA_VALUES) ** 2).sum()
            ),
        },
        "aggregate": aggregate,
    }
    return frame, models, selection, aggregate, error_pool

def fit_and_evaluate_svi(
    train: pd.DataFrame,
    held: pd.DataFrame,
    args: argparse.Namespace,
    reference_starts: dict,
) -> tuple[list[float], dict]:
    bounds = A1.svi_bounds(args.eta_min, args.eta_max)
    errors: list[float] = []
    fitted_expiries = []
    failed_expiries = []
    for expiry, held_group in held.groupby("expiry", sort=True):
        train_group = train[train.expiry == expiry]
        if len(train_group) < args.min_strikes:
            failed_expiries.append(pd.Timestamp(expiry).strftime("%Y-%m-%d"))
            continue
        key = (
            pd.Timestamp(train_group.date.iloc[0]).strftime("%Y-%m-%d"),
            pd.Timestamp(expiry).strftime("%Y-%m-%d"),
        )
        x0 = reference_starts.get(key, {}).get("raw") if reference_starts else None
        fit = A1.fit_svi(
            train_group.k.to_numpy(dtype=float),
            train_group.w.to_numpy(dtype=float),
            bounds=bounds,
            x0_params=x0,
            include_starts=x0 is None,
        )
        if fit is None:
            failed_expiries.append(pd.Timestamp(expiry).strftime("%Y-%m-%d"))
            continue
        rel = svi_repricing_errors(fit["params"], held_group)
        errors.extend(rel.tolist())
        fitted_expiries.append(pd.Timestamp(expiry).strftime("%Y-%m-%d"))
    return errors, {"fitted_expiries": fitted_expiries, "failed_expiries": failed_expiries}


def run_robustness(
    otm: pd.DataFrame,
    sample_dates: list[pd.Timestamp],
    args: argparse.Namespace,
    chosen: tuple[float, float],
    reference_starts: dict,
    cache_dir: Path,
    failed_fits: list[dict],
    cache_stats: dict,
    log: RunLog,
) -> tuple[pd.DataFrame, dict]:
    lambda1, lambda2 = chosen
    rows: list[dict] = []
    pooled: dict[tuple[str, str], list[float]] = {}
    per_seed: dict[tuple[str, str, str], list[float]] = {}
    requested_neural = 0
    succeeded_neural = 0
    failed_neural = 0
    requested_svi = 0
    succeeded_svi = 0
    for date in sample_dates:
        log.check_runtime(args.max_runtime_minutes, f"robustness {date.date()}")
        group = date_quotes(otm, date)
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        designs: list[tuple[str, int, pd.DataFrame, pd.DataFrame]] = []
        for seed_offset in range(args.robust_seeds):
            split_seed = derive_seed(args.seed, date_str, 0.0, 0.0, "split_random", seed_offset)
            rng = np.random.default_rng(split_seed)
            indices = np.arange(len(group))
            rng.shuffle(indices)
            n_drop = int(round(args.drop_fraction * len(group)))
            held = group.iloc[np.sort(indices[:n_drop])].copy()
            train = group.iloc[np.sort(indices[n_drop:])].copy()
            designs.append(("random", seed_offset, train, held))
        held_parts = []
        train_parts = []
        for _, expiry_group in group.groupby("expiry", sort=True):
            order = np.argsort(-np.abs(expiry_group.k.to_numpy(dtype=float)))
            n_hold = max(1, int(math.ceil(args.wing_fraction * len(expiry_group))))
            held_parts.append(expiry_group.iloc[order[:n_hold]])
            train_parts.append(expiry_group.iloc[order[n_hold:]])
        designs.append((
            "wing",
            0,
            pd.concat(train_parts, ignore_index=False),
            pd.concat(held_parts, ignore_index=False),
        ))

        for design, rep, train, held in designs:
            expiries_with_enough = {
                expiry for expiry, subset in train.groupby("expiry") if len(subset) >= args.min_strikes
            }
            skipped = sorted(
                pd.Timestamp(expiry).strftime("%Y-%m-%d")
                for expiry in train.expiry.unique()
                if expiry not in expiries_with_enough
            )
            train = train[train.expiry.isin(expiries_with_enough)].copy()
            held = held[held.expiry.isin(expiries_with_enough)].copy()
            if held.empty:
                continue
            requested_svi += 1
            svi_errors, svi_info = fit_and_evaluate_svi(train, held, args, reference_starts)
            svi_valid_expiries = set(pd.Timestamp(item) for item in svi_info["fitted_expiries"])
            if not svi_valid_expiries:
                continue
            train = train[train.expiry.isin(svi_valid_expiries)].copy()
            held = held[held.expiry.isin(svi_valid_expiries)].copy()
            if held.empty:
                continue
            svi_rel = np.asarray(svi_errors, dtype=float)
            succeeded_svi += 1
            seed_key = f"{design}:{rep}"
            seed = derive_seed(args.seed, date_str, lambda1, lambda2, design, rep)
            requested_neural += 1
            fit_result = run_neural_fit_cached(
                train, args, lambda1, lambda2, seed, design, rep,
                cache_dir, args.fit_timeout_seconds, args.fit_retries,
                failed_fits, cache_stats,
            )
            neural_rel = None
            if fit_result["status"] == "ok":
                succeeded_neural += 1
                neural_rel, _ = neural_repricing_errors(
                    fit_result["model"], held, fit_result["info"]["t_scale"]
                )
                pooled.setdefault((design, "neural"), []).extend(neural_rel.tolist())
                per_seed.setdefault((design, seed_key, "neural"), []).extend(neural_rel.tolist())
            else:
                failed_neural += 1
            pooled.setdefault((design, "svi"), []).extend(svi_rel.tolist())
            per_seed.setdefault((design, seed_key, "svi"), []).extend(svi_rel.tolist())
            for model_name, values, status in (
                ("svi", svi_rel, "ok"),
                ("neural", neural_rel, "ok" if neural_rel is not None else "failed"),
            ):
                if values is None:
                    stats = {"n": 0, "median": float("nan"), "p95": float("nan"), "mean": float("nan"), "max": float("nan")}
                else:
                    stats = pooled_stats(values)
                rows.append({
                    "design": design,
                    "seed": seed_key,
                    "fit_seed": int(seed) if model_name == "neural" else None,
                    "date": date_str,
                    "model": model_name,
                    "status": status,
                    "from_cache": bool(fit_result.get("from_cache", False)) if model_name == "neural" and status == "ok" else False,
                    "attempts": int(fit_result.get("attempts", 0)) if model_name == "neural" else 0,
                    "n_eval": stats["n"],
                    "median_rel_error": stats["median"],
                    "p95_rel_error": stats["p95"],
                    "mean_rel_error": stats["mean"],
                    "max_rel_error": stats["max"],
                    "n_train": int(len(train)),
                    "n_held": int(len(held)),
                    "skipped_expiries": ";".join(skipped),
                    "fit_seconds": float(fit_result["info"]["wall_seconds"]) if model_name == "neural" and status == "ok" else float("nan"),
                })
            log.write(
                f"robust {date_str} {design} rep={rep} fit_seed={seed} "
                f"svi_med={np.median(svi_rel):.4f} "
                f"nn_med={np.median(neural_rel) if neural_rel is not None else float('nan'):.4f}"
            )
    frame = pd.DataFrame(rows)
    summary_rows = []
    for design in ("random", "wing"):
        for model_name in ("svi", "neural"):
            values = np.asarray(pooled.get((design, model_name), []), dtype=float)
            stats = pooled_stats(values)
            fit_rows = frame[
                (frame.design == design) & (frame.model == model_name) & (frame.status == "ok")
            ]
            summary_rows.append({
                "design": design,
                "model": model_name,
                "fits": int(len(fit_rows)),
                "fits_requested": int(requested_neural if model_name == "neural" else requested_svi),
                **stats,
            })
    summary_frame = pd.DataFrame(summary_rows)
    random_rows = summary_frame[summary_frame.design == "random"].set_index("model")
    wing_rows = summary_frame[summary_frame.design == "wing"].set_index("model")
    degradation = {}
    for model_name in ("svi", "neural"):
        if model_name in random_rows.index and model_name in wing_rows.index:
            degradation[model_name] = {
                "median_delta": float(wing_rows.loc[model_name, "median"] - random_rows.loc[model_name, "median"]),
                "median_ratio": float(wing_rows.loc[model_name, "median"] / random_rows.loc[model_name, "median"])
                if random_rows.loc[model_name, "median"] > 0 else float("nan"),
                "p95_delta": float(wing_rows.loc[model_name, "p95"] - random_rows.loc[model_name, "p95"]),
                "p95_ratio": float(wing_rows.loc[model_name, "p95"] / random_rows.loc[model_name, "p95"])
                if random_rows.loc[model_name, "p95"] > 0 else float("nan"),
            }
    neural_wing_worse_median = (
        degradation.get("neural", {}).get("median_delta", float("nan"))
        > degradation.get("svi", {}).get("median_delta", float("nan"))
    )
    neural_wing_worse_p95 = (
        degradation.get("neural", {}).get("p95_delta", float("nan"))
        > degradation.get("svi", {}).get("p95_delta", float("nan"))
    )
    summary = {
        "pooled": summary_rows,
        "coverage": {
            "neural_fits_requested": int(requested_neural),
            "neural_fits_succeeded": int(succeeded_neural),
            "neural_fits_failed": int(failed_neural),
            "svi_fits_requested": int(requested_svi),
            "svi_fits_succeeded": int(succeeded_svi),
            "svi_fits_failed": int(requested_svi - succeeded_svi),
        },
        "degradation_from_random_to_wing": degradation,
        "wing_degrades_more": {
            "median": "neural" if neural_wing_worse_median else "svi",
            "p95": "neural" if neural_wing_worse_p95 else "svi",
        },
        "design_note": (
            "random drops 30% of quotes per sample date using 3 seeds; wing holds out the "
            "outer 15% of |k| within each expiry. Both models are refit on the remaining "
            "quotes and scored only on held-out quotes. Expiries with fewer than min_strikes "
            "training quotes, or where SVI fit failed, are excluded from both models. SVI refits "
            "are warm-started from the full-sample approach1 fit when available; neural refits "
            "are fresh spawned fits with explicit seeds. Pooled summaries cover successful fits only."
        ),
        "per_seed": [
            {"design": design, "seed": seed, "model": model_name, **pooled_stats(np.asarray(values, dtype=float))}
            for (design, seed, model_name), values in sorted(per_seed.items())
        ],
    }
    return frame, summary

def build_surface_frame(
    models: dict,
    sample_dates: list[pd.Timestamp],
    chosen: tuple[float, float],
) -> pd.DataFrame:
    rows = []
    for date in sample_dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        key = (date_str, chosen[0], chosen[1])
        if key not in models:
            continue
        item = models[key]
        model = item["model"]
        t_scale = item["t_scale"]
        expiry_T = np.sort(np.asarray(item["info"]["expiry_T"], dtype=float))
        T_grid = np.unique(np.concatenate([
            expiry_T,
            np.linspace(float(expiry_T.min()), float(expiry_T.max()), 25),
        ]))
        k_grid, t_mesh = np.meshgrid(SURFACE_K, T_grid)
        sigma = neural_sigma_numpy(model, k_grid.reshape(-1), t_mesh.reshape(-1), t_scale).reshape(k_grid.shape)
        w = sigma * sigma * t_mesh
        rows.append(pd.DataFrame({
            "date": date_str,
            "k": k_grid.reshape(-1),
            "T": t_mesh.reshape(-1),
            "sigma": sigma.reshape(-1),
            "w": w.reshape(-1),
            "is_expiry": np.isin(np.round(t_mesh.reshape(-1), 12), np.round(expiry_T, 12)),
        }))
    if not rows:
        return pd.DataFrame(columns=["date", "k", "T", "sigma", "w", "is_expiry"])
    return pd.concat(rows, ignore_index=True)


def build_svi_convergence(histories: list[dict], out_path: Path) -> None:
    dates = sorted({item["date"] for item in histories})
    max_expiries = max(len([item for item in histories if item["date"] == date]) for date in dates)
    titles = []
    for date in dates:
        items = sorted([item for item in histories if item["date"] == date], key=lambda item: item["T"])
        titles.extend([f"{date} exp {index + 1}" for index in range(len(items))])
        titles.extend([""] * (max_expiries - len(items)))
    fig = make_subplots(
        rows=len(dates),
        cols=max_expiries,
        subplot_titles=titles,
        vertical_spacing=0.035,
        horizontal_spacing=0.035,
    )
    for row_index, date in enumerate(dates, start=1):
        items = sorted([item for item in histories if item["date"] == date], key=lambda item: item["T"])
        for col_index, item in enumerate(items, start=1):
            history = np.asarray(item["history"], dtype=float)
            x = np.arange(len(history))
            fig.add_trace(
                go.Scatter(x=x, y=history, mode="lines", name="torch", line=dict(width=1.1, color="#1f77b4"), showlegend=(row_index == 1 and col_index == 1)),
                row=row_index, col=col_index,
            )
            fig.add_trace(
                go.Scatter(x=[0, max(1, len(history) - 1)], y=[item["scipy_objective"]] * 2, mode="lines", name="scipy final", line=dict(width=1.0, color="#d62728", dash="dash"), showlegend=(row_index == 1 and col_index == 1)),
                row=row_index, col=col_index,
            )
            fig.add_trace(
                go.Scatter(x=[0, max(1, len(history) - 1)], y=[item["tolerance_objective"]] * 2, mode="lines", name="1.001 x best", line=dict(width=0.8, color="#2ca02c", dash="dot"), showlegend=(row_index == 1 and col_index == 1)),
                row=row_index, col=col_index,
            )
            if 0 < item["adam_evals"] < len(history):
                fig.add_vline(x=item["adam_evals"], line=dict(width=0.6, color="#7f7f7f"), row=row_index, col=col_index)
            fig.update_xaxes(title_text="objective evaluation", row=row_index, col=col_index)
            fig.update_yaxes(title_text="MSE(w)", type="log", row=row_index, col=col_index)
    fig.update_layout(
        height=max(500, 340 * len(dates)),
        width=1900,
        title="Torch SVI convergence (Adam then L-BFGS) vs scipy least_squares",
    )
    fig.write_html(out_path, include_plotlyjs=True)


def build_surface_html(surface: pd.DataFrame, out_path: Path) -> None:
    dates = sorted(surface.date.unique())
    frames = []
    first_z = None
    first_T = None
    k_axis = np.sort(surface.k.unique())
    for date in dates:
        subset = surface[surface.date == date]
        T_axis = np.sort(subset["T"].unique())
        z = subset.pivot(index="T", columns="k", values="sigma").reindex(index=T_axis, columns=k_axis).to_numpy()
        if first_z is None:
            first_z = z
            first_T = T_axis
        frames.append(go.Frame(
            name=date,
            data=[go.Surface(x=k_axis, y=T_axis, z=z, colorscale="Viridis", colorbar=dict(title="sigma"))],
        ))
    fig = go.Figure(
        data=[go.Surface(x=k_axis, y=first_T, z=first_z, colorscale="Viridis", colorbar=dict(title="sigma"))],
        frames=frames,
    )
    steps = [
        dict(method="animate", args=[[date], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=date)
        for date in dates
    ]
    fig.update_layout(
        title="Neural implied-volatility surface (chosen lambdas; use slider to change date)",
        scene=dict(xaxis_title="k = ln(K/F)", yaxis_title="T (years)", zaxis_title="sigma"),
        sliders=[dict(active=0, currentvalue=dict(prefix="Date: "), steps=steps)],
        height=850,
    )
    fig.write_html(out_path, include_plotlyjs=True)


def approach1_params(approach1_dir: Path) -> tuple[pd.DataFrame | None, dict]:
    path = approach1_dir / "svi_params.csv"
    if not path.exists():
        return None, {"available": False, "reason": f"missing {path}"}
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame["expiry"] = pd.to_datetime(frame.expiry).dt.strftime("%Y-%m-%d")
    return frame, {"available": True, "path": str(path), "rows": int(len(frame))}


def compare_with_approach1(
    otm: pd.DataFrame,
    models: dict,
    sample_dates: list[pd.Timestamp],
    chosen: tuple[float, float],
    approach1_dir: Path,
) -> dict:
    frame, availability = approach1_params(approach1_dir)
    if frame is None:
        return {"available": False, "reason": availability.get("reason")}
    params_by_key = {
        (row.date, row.expiry): np.array([row.a_repaired, row.b_repaired, row.rho_repaired, row.m_repaired, row.eta_repaired], dtype=float)
        for row in frame.itertuples(index=False)
    }
    neural_errors: list[float] = []
    approach1_errors: list[float] = []
    per_date = []
    missing = []
    missing_neural_dates = []
    for date in sample_dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        if (date_str, chosen[0], chosen[1]) not in models:
            missing_neural_dates.append(date_str)
            continue
        group = date_quotes(otm, date)
        common_expiries = []
        for expiry in sorted(group.expiry.unique()):
            expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d")
            if (date_str, expiry_str) in params_by_key:
                common_expiries.append(expiry)
            else:
                missing.append({"date": date_str, "expiry": expiry_str})
        if not common_expiries:
            continue
        subset = group[group.expiry.isin(common_expiries)].copy()
        item = models[(date_str, chosen[0], chosen[1])]
        neural_rel, _ = neural_repricing_errors(item["model"], subset, item["t_scale"])
        a1_rel = []
        for expiry, expiry_group in subset.groupby("expiry", sort=True):
            expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d")
            a1_rel.extend(svi_repricing_errors(params_by_key[(date_str, expiry_str)], expiry_group))
        neural_errors.extend(neural_rel.tolist())
        approach1_errors.extend(a1_rel)
        neural_stats = pooled_stats(neural_rel)
        a1_stats = pooled_stats(np.asarray(a1_rel, dtype=float))
        per_date.append({
            "date": date_str,
            "n_quotes": int(len(subset)),
            "neural_median": neural_stats["median"],
            "neural_p95": neural_stats["p95"],
            "neural_max": neural_stats["max"],
            "approach1_median": a1_stats["median"],
            "approach1_p95": a1_stats["p95"],
            "approach1_max": a1_stats["max"],
        })
    return {
        "available": True,
        "source": availability,
        "basis": "approach1 repaired SVI parameters from svi_params.csv, re-priced on this run's OTM quote set",
        "common_quotes": int(len(neural_errors)),
        "neural": pooled_stats(np.asarray(neural_errors, dtype=float)),
        "approach1": pooled_stats(np.asarray(approach1_errors, dtype=float)),
        "per_date": per_date,
        "missing_approach1_expiries": missing,
        "missing_neural_dates": missing_neural_dates,
    }


def run_determinism_check(
    group: pd.DataFrame,
    date_str: str,
    args: argparse.Namespace,
    lambda1: float,
    lambda2: float,
    cache_dir: Path,
    failed_fits: list[dict],
    cache_stats: dict,
) -> dict:
    seed = derive_seed(args.seed, date_str, lambda1, lambda2, "determinism", 0)
    results = {}
    for label in ("determinism_a", "determinism_b"):
        results[label] = run_neural_fit_cached(
            group, args, lambda1, lambda2, seed, label, 0,
            cache_dir, args.fit_timeout_seconds, args.fit_retries,
            failed_fits, cache_stats, force_recompute=True,
        )
    if any(item["status"] != "ok" for item in results.values()):
        return {
            "status": "failed",
            "date": date_str,
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "seed": int(seed),
            "errors": {label: item.get("error") for label, item in results.items()},
        }
    model_a = results["determinism_a"]["model"]
    model_b = results["determinism_b"]["model"]
    loss_a = float(results["determinism_a"]["info"]["loss"])
    loss_b = float(results["determinism_b"]["info"]["loss"])
    loss_abs_diff = abs(loss_a - loss_b)
    loss_rel_diff = loss_abs_diff / max(abs(loss_a), abs(loss_b), 1e-30)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    max_state_diff = float(max(
        (state_a[name] - state_b[name]).abs().max().item()
        for name in state_a
    ))
    deterministic = bool(loss_abs_diff <= 1e-12 and max_state_diff <= 1e-10)
    return {
        "status": "ok",
        "date": date_str,
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "seed": int(seed),
        "loss_a": loss_a,
        "loss_b": loss_b,
        "loss_abs_diff": loss_abs_diff,
        "loss_rel_diff": loss_rel_diff,
        "max_state_abs_diff": max_state_diff,
        "deterministic": deterministic,
        "note": (
            "Two fresh spawned fits with the same explicit seed. A nonzero difference is "
            "reported as nondeterminism for AGENTS.md rule 3."
        ),
    }


def make_metrics_payload(
    args: argparse.Namespace,
    metadata: dict,
    filter_counts: dict,
    iv_stats: dict,
    sample_rows: list[dict],
    otm_info: dict,
    solver_summary: dict,
    selection: dict,
    ablation_frame: pd.DataFrame,
    robustness_summary: dict,
    chosen_audits: list[dict],
    chosen_errors: np.ndarray,
    chosen_per_date: pd.DataFrame,
    comparison: dict,
    failed_fits: list[dict],
    determinism: dict,
    cache_stats: dict,
    cache_dir: Path,
    missing_chosen_dates: list[str],
    fit_seconds: float,
    total_seconds: float,
    sample_dates: list[pd.Timestamp],
) -> dict:
    chosen = (selection["chosen_lambda1"], selection["chosen_lambda2"])
    unpenalized = selection["unpenalized"]
    mismatch = [row for row in sample_rows if not row["matches_metadata"]]
    params = {
        "approach": "Q1B_DIFFERENTIABLE_SVI_NEURAL",
        "dataset": "io",
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "price_col": args.price_col,
        "min_volume": args.min_volume,
        "min_price": args.min_price,
        "kband": args.kband,
        "min_days": args.min_days,
        "min_strikes": args.min_strikes,
        "eta_bounds": [args.eta_min, args.eta_max],
        "inputs": {
            "options": str(args.options),
            "shibor": str(args.shibor),
            "metadata": str(args.metadata),
            "approach1_dir": str(args.approach1_dir),
        },
        "sample_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in sample_dates],
        "sample_date_rule": (
            "For each of the 6 metadata sample months, take the date nearest mid-month "
            "(day 15) with at least 80% of that month's maximum quote count in data/clean/options_io.csv."
        ),
        "svi_form": "a+b*(rho*(k-m)+sqrt((k-m)^2+eta^2)); b=softplus(u), rho=tanh(v), eta=softplus(s), a,m free",
        "svi_solver_protocol": "torch Adam then L-BFGS (float64) vs scipy least_squares, both from the same full-sample SVI start (approach1 raw fit when available)",
        "svi_robustness": "refit by one bounded least_squares from the full-sample SVI warm start (multi-start fallback if missing)",
        "neural_architecture": {
            "layers": args.nn_hidden_layers + 1,
            "hidden_layers": args.nn_hidden_layers,
            "hidden_width": args.nn_hidden,
            "activation": "tanh",
            "output": "softplus + 1e-6",
            "inputs": "k/0.2, sqrt(T)/sqrt(max T on sample date)",
        },
        "neural_loss": (
            "mean((C_BS(sigma)-C_mkt)^2) in undiscounted OTM price per unit forward, plus "
            "lambda1*mean(max(0,-g)^2) + lambda2*mean(max(0,-dw/dT)^2) on a collocation grid "
            "k in [-0.14,0.14] crossed with T spanning the listed expiries; g includes +w''/2."
        ),
        "lambda_grid": list(LAMBDA_VALUES),
        "selection_rule": selection["rule"],
        "chosen_lambdas": {"lambda1": chosen[0], "lambda2": chosen[1]},
        "unpenalized_lambdas": {"lambda1": 0.0, "lambda2": 0.0},
        "robustness": {
            "drop_fraction": args.drop_fraction,
            "wing_fraction": args.wing_fraction,
            "seeds": [args.seed + index for index in range(args.robust_seeds)],
        },
        "price_note": "CFFEX fields: settle=今结算, close=今收盘; no settlement algorithm is inferred.",
        "rate_convention": "reused from approach1 (SHIBOR ACT/360 simple converted to continuous zero on ACT/365)",
        "forward_estimator": "reused from approach1 (volume-weighted parity forward within |ln(F_K/median)| <= 0.05)",
        "out_dir": str(args.out_dir),
        "cache_dir": str(cache_dir),
        "worker_isolation": {
            "method": "spawn one fresh process per neural fit",
            "timeout_seconds": float(args.fit_timeout_seconds),
            "retries": int(args.fit_retries),
            "thread_env": {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "NUMEXPR_NUM_THREADS": 1},
            "torch_threads": 1,
        },
        "seed_derivation": "sha256(base_seed|date|lambda1|lambda2|design|rep) mod (2^31-1); explicit per fit",
    }
    metrics = {
        "quotes_total": int(filter_counts["quotes_total"]),
        "quotes_liquid": int(filter_counts["quotes_liquid"]),
        "filter_removals": {
            "independent": filter_counts["independent_removals"],
            "sequential": filter_counts["sequential_removals"],
            "note": "Reused approach1 filters: volume>=min_volume, price>=min_price, |k|<=kband, days>=min_days.",
        },
        "sample_dates": sample_rows,
        "sample_dates_metadata_mismatch": mismatch,
        "otm": {
            **otm_info,
            "note": "OTM definition and IV inversion reused from approach1; expiries with fewer than min_strikes OTM quotes are dropped.",
        },
        "svi_solver": solver_summary,
        "ablation": {
            "aggregate": selection["aggregate"],
            "chosen": selection["chosen"],
            "unpenalized": unpenalized,
            "feasible_pairs": selection["feasible_pairs"],
            "selection_rule": selection["rule"],
            "per_date": ablation_frame.to_dict(orient="records"),
        },
        "robustness": robustness_summary,
        "chosen_model": {
            "lambdas": {"lambda1": chosen[0], "lambda2": chosen[1]},
            "in_sample_repricing": pooled_stats(chosen_errors),
            "per_date_repricing": chosen_per_date.to_dict(orient="records"),
            "missing_dates": missing_chosen_dates,
            "violations": {
                "per_date": chosen_audits,
                "butterfly_violations_strict": int(sum(item["butterfly_violations_strict"] for item in chosen_audits)),
                "butterfly_violations": int(sum(item["butterfly_violations"] for item in chosen_audits)),
                "calendar_violations_strict": int(sum(item["calendar_violations_strict"] for item in chosen_audits)),
                "calendar_violations": int(sum(item["calendar_violations"] for item in chosen_audits)),
                "min_g": float(min((item["min_g"] for item in chosen_audits), default=float("nan"))),
                "min_forward_variance": float(min((item["min_forward_variance"] for item in chosen_audits), default=float("nan"))),
                "tolerance": {"butterfly": BUTTERFLY_TOL, "calendar_w": CALENDAR_TOL},
                "grid": {"start": float(GRID_K[0]), "stop": float(GRID_K[-1]), "points": int(len(GRID_K))},
                "note": "Violations counted on the approach1 k grid at each listed expiry; calendar compares adjacent expiries only.",
            },
        },
        "comparison_approach1": comparison,
        "failed_fits": failed_fits,
        "determinism_check": determinism,
        "cache": {**cache_stats, "dir": str(cache_dir)},
        "runtime": {
            "fit_seconds": float(fit_seconds),
            "total_seconds": float(total_seconds),
            "max_runtime_minutes": float(args.max_runtime_minutes),
            "torch_threads": int(args.torch_threads),
            "note": "fit_seconds covers SVI solver comparison, neural ablation and robustness refits; total_seconds is wall time for the whole run.",
        },
        "metadata_input": {
            "sample_date_report": metadata.get("sample_date_report", []),
            "sample_month_report": metadata.get("sample_month_report", []),
        },
        "iv_summary": {
            "attempted_quotes": iv_stats["attempted_quotes"],
            "ok_quotes": iv_stats["ok_quotes"],
            "status_counts": iv_stats["status_counts"],
            "gap_median": iv_stats["gap_median"],
            "gap_p95_abs": iv_stats["gap_p95_abs"],
        },
    }
    return {
        "params": params,
        "metrics": metrics,
        "fit_seconds": float(fit_seconds),
        "history": {},
    }


def print_summary(
    console: Console,
    params: dict,
    metrics: dict,
    solver_frame: pd.DataFrame,
    ablation_aggregate: pd.DataFrame,
    robustness_summary: dict,
    comparison: dict,
) -> None:
    console.print(Panel.fit(
        f"Q1B differentiable SVI + neural surface | dataset=io | seed={params['seed']} | "
        f"sample dates={len(params['sample_dates'])} | chosen lambdas={params['chosen_lambdas']}",
        title="Q1 Option B",
    ))

    table = Table(title="Sample date selection (80% rule)")
    for column in ("month", "max quotes", "80% threshold", "chosen", "rows", "metadata", "match"):
        table.add_column(column, justify="right" if column != "month" else "left")
    for row in metrics["sample_dates"]:
        table.add_row(
            row["month"], f"{row['max_quotes']}", f"{row['threshold_80pct']:.1f}",
            row["date"], f"{row['date_rows']}", row["metadata_date"], str(row["matches_metadata"]),
        )
    console.print(table)

    table = Table(title="SVI solver comparison (MSE in w)")
    table.add_column("Metric")
    table.add_column("Torch Adam+L-BFGS", justify="right")
    table.add_column("scipy least_squares", justify="right")
    summary = metrics["svi_solver"]
    table.add_row("fits", f"{summary['fits']}", f"{summary['fits']}")
    table.add_row("reached 0.1% tolerance", f"{summary['torch_reached']}", f"{summary['scipy_reached']}")
    table.add_row("median final objective", f"{summary['torch_objective_median']:.3e}", f"{summary['scipy_objective_median']:.3e}")
    table.add_row("max final objective", f"{summary['torch_objective_max']:.3e}", f"{summary['scipy_objective_max']:.3e}")
    table.add_row("median iterations to tolerance", f"{summary['torch_iterations_to_tolerance_median']:.1f}", f"{summary['scipy_nfev_median']:.1f}")
    table.add_row("median wall seconds", f"{summary['torch_wall_seconds_median']:.4f}", f"{summary['scipy_wall_seconds_median']:.4f}")
    console.print(table)

    table = Table(title="Ablation (pooled in-sample OTM relative repricing error)")
    for column in ("lambda1", "lambda2", "fits ok/req", "median", "p95", "max", "bf strict", "bf tol", "cal strict", "cal tol", "fit s"):
        table.add_column(column, justify="right")
    for row in metrics["ablation"]["aggregate"]:
        table.add_row(
            f"{row['lambda1']:.2f}", f"{row['lambda2']:.2f}",
            f"{row['n_fits_succeeded']}/{row['n_fits_requested']}",
            f"{row['median_rel_error']:.6f}",
            f"{row['p95_rel_error']:.6f}", f"{row['max_rel_error']:.6f}",
            f"{row['butterfly_violations_strict']}", f"{row['butterfly_violations']}",
            f"{row['calendar_violations_strict']}", f"{row['calendar_violations']}",
            f"{row['fit_seconds_total']:.1f}",
        )
    console.print(table)

    table = Table(title="Robustness (pooled held-out OTM relative price error)")
    table.add_column("Design")
    table.add_column("Model")
    table.add_column("fits", justify="right")
    table.add_column("n", justify="right")
    table.add_column("median", justify="right")
    table.add_column("p95", justify="right")
    for row in robustness_summary["pooled"]:
        table.add_row(
            row["design"], row["model"], f"{row['fits']}/{row['fits_requested']}",
            f"{row['n']}", f"{row['median']:.6f}", f"{row['p95']:.6f}",
        )
    console.print(table)
    wing = robustness_summary["wing_degrades_more"]
    console.print(f"Wing holdout degrades more: median -> [bold]{wing['median']}[/bold], p95 -> [bold]{wing['p95']}[/bold]")

    if comparison.get("available"):
        table = Table(title="Like-for-like comparison with approach1 on the same OTM quotes")
        table.add_column("Model")
        table.add_column("n", justify="right")
        table.add_column("median", justify="right")
        table.add_column("p95", justify="right")
        table.add_column("max", justify="right")
        for label, key in (("approach1 repaired SVI", "approach1"), ("neural chosen", "neural")):
            stats = comparison[key]
            table.add_row(label, f"{stats['n']}", f"{stats['median']:.6f}", f"{stats['p95']:.6f}", f"{stats['max']:.6f}")
        console.print(table)

    det = metrics["determinism_check"]
    if det.get("status") == "ok":
        console.print(
            f"Determinism: loss diff={det['loss_abs_diff']:.3e} max state diff="
            f"{det['max_state_abs_diff']:.3e} deterministic={det['deterministic']}"
        )
    else:
        console.print(f"Determinism: [red]{det.get('status')}[/red] {det}")
    console.print(f"Failed fits: {len(metrics['failed_fits'])}")
    console.print(f"Cache stats: {metrics['cache']}")

    violations = metrics["chosen_model"]["violations"]
    console.print(
        f"Chosen neural violations: butterfly strict={violations['butterfly_violations_strict']} "
        f"tol={violations['butterfly_violations']} | calendar strict={violations['calendar_violations_strict']} "
        f"tol={violations['calendar_violations']} | min g={violations['min_g']:.3e}"
    )
    console.print(f"Total runtime: {metrics['runtime']['total_seconds']:.1f}s")


def run(args: argparse.Namespace) -> dict:
    console = Console(width=220)
    start = time.perf_counter()
    set_seed(args.seed)
    set_torch_threads(args.torch_threads)
    out_dir = ensure_inside_repo(args.out_dir)
    cache_dir = ensure_inside_repo(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed_fits: list[dict] = []
    cache_stats: dict = {}
    log = RunLog(args.log_path, console)
    log.write(f"start seed={args.seed} torch_threads={args.torch_threads}")
    if not (0.0 < args.eta_min < args.eta_max):
        raise ValueError("require 0 < eta_min < eta_max")
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading and verifying data (approach1 loaders)[/bold]")
    options, shibor, metadata = A1.load_data(args.options, args.shibor, args.metadata)
    console.print(f"  options rows={len(options):,} dates={options.date.nunique()} expiries={options.expiry.nunique()}")
    console.print(f"  shibor rows={len(shibor):,} range={shibor.date.min().date()}..{shibor.date.max().date()}")

    console.print("[bold]Forwards, filters and implied vols (approach1 functions)[/bold]")
    forwards, _ = A1.estimate_forwards(options, shibor, args.price_col)
    data, filter_counts = A1.apply_filters(options, forwards, args)
    iv, _, iv_stats = A1.compute_implied_vols(data[data.liquid])
    console.print(f"  liquid quotes={filter_counts['quotes_liquid']:,}/{filter_counts['quotes_total']:,} | IV ok={iv_stats['ok_quotes']:,}")

    sample_dates, sample_rows = select_sample_dates(options, metadata)
    console.print("[bold]Sample dates[/bold]")
    for row in sample_rows:
        flag = "match" if row["matches_metadata"] else f"metadata={row['metadata_date']}"
        console.print(
            f"  {row['month']}: max={row['max_quotes']} threshold={row['threshold_80pct']:.1f} "
            f"chosen={row['date']} ({row['date_rows']} rows) {flag}"
        )
    otm, otm_info = build_otm_quotes(data, iv, sample_dates, args.min_strikes)
    console.print(f"  OTM quotes={otm_info['otm_quotes']:,} | date-expiry fits={otm_info['date_expiry_fits']}")

    console.print("[bold]Step 1: differentiable SVI vs scipy[/bold]")
    solver_start = time.perf_counter()
    reference_bounds = A1.svi_bounds(args.eta_min, args.eta_max)
    reference_starts, start_info = load_reference_starts(args.approach1_dir, otm, sample_dates, reference_bounds)
    console.print(
        f"  reference starts: approach1={start_info['from_approach1']} "
        f"computed={start_info['computed']} missing={len(start_info['missing'])}"
    )
    solver_frame, solver_histories, solver_summary = run_svi_solver_comparison(
        otm, sample_dates, args, reference_starts, log
    )
    solver_summary["start_info"] = start_info
    solver_seconds = time.perf_counter() - solver_start
    console.print(
        f"  fits={solver_summary['fits']} torch reached={solver_summary['torch_reached']} "
        f"scipy reached={solver_summary['scipy_reached']} | median torch={solver_summary['torch_objective_median']:.3e} "
        f"scipy={solver_summary['scipy_objective_median']:.3e}"
    )

    console.print("[bold]Steps 2-3: neural surface and lambda ablation[/bold]")
    ablation_start = time.perf_counter()
    ablation_frame, models, selection, ablation_aggregate, error_pool = run_ablation(
        otm, sample_dates, args, cache_dir, failed_fits, cache_stats, log
    )
    ablation_aggregate = pd.DataFrame(ablation_aggregate)
    ablation_seconds = time.perf_counter() - ablation_start
    chosen = (selection["chosen_lambda1"], selection["chosen_lambda2"])
    console.print(
        f"  chosen lambdas={chosen} | feasible pairs={selection['feasible_pairs']} | "
        f"fits ok={selection['coverage']['fits_succeeded']}/{selection['coverage']['fits_requested']}"
    )

    console.print("[bold]Step 4: robustness[/bold]")
    robust_start = time.perf_counter()
    robustness_frame, robustness_summary = run_robustness(
        otm, sample_dates, args, chosen, reference_starts, cache_dir, failed_fits, cache_stats, log
    )
    robust_seconds = time.perf_counter() - robust_start
    console.print(
        f"  rows={len(robustness_frame)} | neural fits ok="
        f"{robustness_summary['coverage']['neural_fits_succeeded']}/"
        f"{robustness_summary['coverage']['neural_fits_requested']}"
    )

    console.print("[bold]Determinism check (same seed, two fresh fits)[/bold]")
    det_start = time.perf_counter()
    det_date = sample_dates[0]
    det_date_str = pd.Timestamp(det_date).strftime("%Y-%m-%d")
    determinism = run_determinism_check(
        date_quotes(otm, det_date), det_date_str, args, chosen[0], chosen[1],
        cache_dir, failed_fits, cache_stats,
    )
    det_seconds = time.perf_counter() - det_start
    if determinism["status"] == "ok":
        console.print(
            f"  {det_date_str} lambdas={chosen} loss_diff={determinism['loss_abs_diff']:.3e} "
            f"max_state_diff={determinism['max_state_abs_diff']:.3e} "
            f"deterministic={determinism['deterministic']}"
        )
    else:
        console.print(f"  determinism check failed: {determinism}")

    console.print("[bold]Chosen-model diagnostics and approach1 comparison[/bold]")
    missing_chosen_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in sample_dates
        if (pd.Timestamp(date).strftime("%Y-%m-%d"), chosen[0], chosen[1]) not in models
    ]
    if missing_chosen_dates:
        console.print(f"  [yellow]missing chosen fits for {missing_chosen_dates}[/yellow]")
    chosen_errors = np.asarray(error_pool.get(chosen, []), dtype=float)
    chosen_per_date = ablation_frame[
        (ablation_frame.lambda1 == chosen[0]) & (ablation_frame.lambda2 == chosen[1])
    ].copy()
    chosen_audits = []
    for date in sample_dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        key = (date_str, chosen[0], chosen[1])
        if key not in models:
            continue
        item = models[key]
        audit = neural_audit(item["model"], np.asarray(item["info"]["expiry_T"], dtype=float), item["t_scale"])
        chosen_audits.append({"date": date_str, **audit})
    comparison = compare_with_approach1(otm, models, sample_dates, chosen, args.approach1_dir)

    surface = build_surface_frame(models, sample_dates, chosen)
    fit_seconds = solver_seconds + ablation_seconds + robust_seconds + det_seconds
    total_seconds = time.perf_counter() - start

    metrics_payload = make_metrics_payload(
        args, metadata, filter_counts, iv_stats, sample_rows, otm_info,
        solver_summary, selection, ablation_frame, robustness_summary,
        chosen_audits, chosen_errors, chosen_per_date, comparison,
        failed_fits, determinism, cache_stats, cache_dir, missing_chosen_dates,
        fit_seconds, total_seconds, sample_dates,
    )
    metrics_payload["history"] = {
        "sample_date_selection": sample_rows,
        "svi_solver_fits": solver_frame.to_dict(orient="records"),
        "ablation_fits": ablation_frame.to_dict(orient="records"),
        "robustness_fits": robustness_frame.to_dict(orient="records"),
        "failed_fits": failed_fits,
        "determinism_check": determinism,
        "cache_stats": cache_stats,
        "timings": {
            "solver_seconds": solver_seconds,
            "ablation_seconds": ablation_seconds,
            "robustness_seconds": robust_seconds,
            "determinism_seconds": det_seconds,
            "fit_seconds": fit_seconds,
            "total_seconds": total_seconds,
        },
        "rates": {
            "pillars": A1.PILLAR_DAYS,
            "convention": "simple ACT/360 SHIBOR converted to continuous zero rate on ACT/365 T (reused from approach1)",
        },
        "log_path": str(args.log_path),
    }

    console.print("[bold]Writing outputs[/bold]")
    solver_frame.to_csv(out_dir / "solver_comparison.csv", index=False)
    ablation_frame.to_csv(out_dir / "ablation.csv", index=False)
    ablation_aggregate.to_csv(out_dir / "ablation_aggregate.csv", index=False)
    robustness_frame.to_csv(out_dir / "robustness.csv", index=False)
    surface.to_csv(out_dir / "surface.csv", index=False)
    if failed_fits:
        pd.DataFrame(failed_fits).to_csv(out_dir / "failed_fits.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(A1.to_jsonable(metrics_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    build_svi_convergence(solver_histories, out_dir / "svi_convergence.html")
    if len(surface):
        build_surface_html(surface, out_dir / "surface.html")
    log.write(f"wrote {out_dir}")

    print_summary(console, metrics_payload["params"], metrics_payload["metrics"], solver_frame, ablation_aggregate, robustness_summary, comparison)
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
