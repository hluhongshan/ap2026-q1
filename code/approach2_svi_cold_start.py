#!/usr/bin/env python3
"""Cold-start SVI solver comparison for Q1 Option B."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import approach2_q1b_neural as q1b

A1 = q1b.A1
DTYPE = torch.float64
SOLVERS = ("adam", "lbfgs", "scipy")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-start SVI solver comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--price-col", choices=["close", "settle"], default="settle")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "approach2" / "io_scaled")
    parser.add_argument("--adam-steps", type=int, default=5000)
    parser.add_argument("--adam-lr", type=float, default=0.03)
    parser.add_argument("--lbfgs-steps", type=int, default=500)
    parser.add_argument("--compare-tol-relative", type=float, default=1e-3)
    parser.add_argument("--max-fits", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def cold_start(k: np.ndarray, w: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lower, upper = bounds
    params = np.array([0.5 * float(np.min(w)), 0.1, -0.3, 0.0, 0.1], dtype=float)
    return np.clip(params, lower + 1e-10, upper - 1e-10)


def svi_jacobian(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, rho, m, eta = (float(item) for item in params)
    d = k - m
    s = np.sqrt(d * d + eta * eta)
    jac = np.empty((len(k), 5), dtype=float)
    jac[:, 0] = 1.0
    jac[:, 1] = rho * d + s
    jac[:, 2] = b * d
    jac[:, 3] = -b * (rho + d / s)
    jac[:, 4] = b * eta / s
    return jac


def mse(k: np.ndarray, w: np.ndarray, params: np.ndarray) -> float:
    return float(np.mean((A1.svi_w(k, params) - w) ** 2))


def run_adam(
    k: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
) -> dict:
    q1b.set_seed(seed)
    k_tensor = torch.tensor(k, dtype=DTYPE)
    w_tensor = torch.tensor(w, dtype=DTYPE)
    model = q1b.TorchSVI(x0)
    initial = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    scale = 1.0 / initial if initial > 0.0 else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1), eta_min=lr * 1e-3
    )
    trace = [(0.0, 0, initial)]
    best = initial
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    start = time.perf_counter()
    loss = torch.mean((model.w(k_tensor) - w_tensor) ** 2)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        (scale * loss).backward()
        optimizer.step()
        scheduler.step()
        loss = torch.mean((model.w(k_tensor) - w_tensor) ** 2)
        value = float(loss.detach()) / scale
        trace.append((time.perf_counter() - start, step, value))
        if value < best:
            best = value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    wall = time.perf_counter() - start
    model.load_state_dict(best_state)
    final = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    return {
        "params": model.numpy_params(),
        "objective": final,
        "trace": trace,
        "total_iterations": int(steps),
        "wall_seconds": float(wall),
        "scale": float(scale),
    }


def run_lbfgs(
    k: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    steps: int,
    seed: int,
) -> dict:
    q1b.set_seed(seed)
    k_tensor = torch.tensor(k, dtype=DTYPE)
    w_tensor = torch.tensor(w, dtype=DTYPE)
    model = q1b.TorchSVI(x0)
    initial = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    scale = 1.0 / initial if initial > 0.0 else 1.0
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max(steps, 1),
        max_eval=max(2 * steps, 20),
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
    )
    state = optimizer.state[model.raw]
    trace: list[tuple[float, int, float]] = []
    best = initial
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    start = time.perf_counter()

    def closure() -> torch.Tensor:
        nonlocal best, best_state
        optimizer.zero_grad(set_to_none=True)
        loss = scale * torch.mean((model.w(k_tensor) - w_tensor) ** 2)
        loss.backward()
        value = float(loss.detach()) / scale
        trace.append((time.perf_counter() - start, int(state.get("n_iter", 0)), value))
        if value < best:
            best = value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        return loss

    try:
        optimizer.step(closure)
    except (RuntimeError, ValueError):
        pass
    wall = time.perf_counter() - start
    model.load_state_dict(best_state)
    final = float(torch.mean((model.w(k_tensor) - w_tensor) ** 2).detach())
    return {
        "params": model.numpy_params(),
        "objective": final,
        "trace": trace,
        "total_iterations": int(state.get("n_iter", 0)),
        "wall_seconds": float(wall),
        "scale": float(scale),
    }


def run_scipy(
    k: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> dict:
    trace: list[tuple[float, int, float]] = []
    iteration = [0]
    start = time.perf_counter()

    def residual(params: np.ndarray) -> np.ndarray:
        values = A1.svi_w(k, params) - w
        trace.append((time.perf_counter() - start, iteration[0], float(np.mean(values * values))))
        return values

    def callback(_: np.ndarray) -> None:
        iteration[0] += 1

    sol = least_squares(
        residual,
        x0,
        jac=lambda params: svi_jacobian(k, params),
        bounds=bounds,
        method="trf",
        x_scale="jac",
        max_nfev=4000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        callback=callback,
    )
    wall = time.perf_counter() - start
    params = np.asarray(sol.x, dtype=float)
    return {
        "params": params,
        "objective": mse(k, w, params),
        "trace": trace,
        "total_iterations": int(iteration[0]),
        "wall_seconds": float(wall),
        "solver_nfev": int(sol.nfev),
    }


def first_tolerance(trace: list[tuple[float, int, float]], tolerance: float) -> tuple | None:
    for index, item in enumerate(trace):
        if item[2] <= tolerance:
            return index, item
    return None


def solver_record(run: dict, tolerance: float) -> dict:
    hit = first_tolerance(run["trace"], tolerance)
    reached = bool(run["objective"] <= tolerance)
    if not reached:
        hit = None
    return {
        "reached": reached,
        "iterations_to_tolerance": int(hit[1][1]) if hit is not None else None,
        "function_evals_to_tolerance": int(hit[0] + 1) if hit is not None else None,
        "total_iterations": int(run["total_iterations"]),
        "total_function_evals": int(len(run["trace"])),
        "wall_seconds_to_tolerance": float(hit[1][0]) if hit is not None else None,
        "wall_seconds_total": float(run["wall_seconds"]),
        "final_objective": float(run["objective"]),
        "params": np.asarray(run["params"], dtype=float),
    }


def row_from_fit(
    date: str,
    expiry: str,
    T: float,
    n_quotes: int,
    k: np.ndarray,
    w: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    reference: np.ndarray,
    reference_source: str,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    x0 = cold_start(k, w, bounds)
    cold_objective = mse(k, w, x0)
    approach1_objective = mse(k, w, reference)
    adam = run_adam(k, w, x0, args.adam_steps, args.adam_lr, args.seed)
    lbfgs = run_lbfgs(k, w, x0, args.lbfgs_steps, args.seed)
    scipy = run_scipy(k, w, x0, bounds)
    best_known = min(adam["objective"], lbfgs["objective"], scipy["objective"], approach1_objective)
    tolerance = best_known * (1.0 + args.compare_tol_relative) + 1e-15
    rows = {}
    for name, run in (("adam", adam), ("lbfgs", lbfgs), ("scipy", scipy)):
        rows[name] = solver_record(run, tolerance)
    row = {
        "date": date,
        "expiry": expiry,
        "T": float(T),
        "n_quotes": int(n_quotes),
        "cold_a": float(x0[0]),
        "cold_b": float(x0[1]),
        "cold_rho": float(x0[2]),
        "cold_m": float(x0[3]),
        "cold_eta": float(x0[4]),
        "cold_objective": float(cold_objective),
        "approach1_objective": float(approach1_objective),
        "approach1_source": reference_source,
        "best_known_objective": float(best_known),
        "tolerance_objective": float(tolerance),
        "torch_scale": float(adam["scale"]),
    }
    for name in SOLVERS:
        record = rows[name]
        row[f"{name}_reached"] = record["reached"]
        row[f"{name}_iterations_to_tolerance"] = record["iterations_to_tolerance"]
        row[f"{name}_function_evals_to_tolerance"] = record["function_evals_to_tolerance"]
        row[f"{name}_total_iterations"] = record["total_iterations"]
        row[f"{name}_total_function_evals"] = record["total_function_evals"]
        row[f"{name}_wall_seconds_to_tolerance"] = record["wall_seconds_to_tolerance"]
        row[f"{name}_wall_seconds_total"] = record["wall_seconds_total"]
        row[f"{name}_final_objective"] = record["final_objective"]
        for offset, letter in enumerate(("a", "b", "rho", "m", "eta")):
            row[f"{name}_{letter}"] = float(record["params"][offset])
        if name == "scipy":
            row["scipy_solver_nfev"] = int(scipy["solver_nfev"])
    history = {
        "date": date,
        "expiry": expiry,
        "T": float(T),
        "tolerance": float(tolerance),
        "best_known_objective": float(best_known),
        "cold_objective": float(cold_objective),
        "runs": {
            "adam": adam,
            "lbfgs": lbfgs,
            "scipy": scipy,
        },
    }
    return row, history


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def censored(row: dict, name: str, field: str, total_field: str) -> float:
    value = row[f"{name}_{field}"]
    if value is not None and math.isfinite(float(value)):
        return float(value)
    return float(row[f"{name}_{total_field}"])


def summarise(rows: list[dict], args: argparse.Namespace) -> dict:
    summary = {
        "fits": int(len(rows)),
        "tolerance_relative": float(args.compare_tol_relative),
        "tolerance_rule": "final objective <= (1 + compare_tol_relative) * best_known + 1e-15",
        "best_known_rule": "minimum of the three cold-start runs and approach1's multi-start raw fit",
        "torch_objective_scaling": "c = 1 / MSE(cold start), per fit; final objectives and tolerance use unscaled MSE",
        "solvers": {},
    }
    for name in SOLVERS:
        reached = [row for row in rows if row[f"{name}_reached"]]
        summary["solvers"][name] = {
            "reached": int(len(reached)),
            "not_reached": int(len(rows) - len(reached)),
            "final_objective": stats([row[f"{name}_final_objective"] for row in rows]),
            "relative_gap_to_best": stats(
                [
                    (row[f"{name}_final_objective"] - row["best_known_objective"])
                    / max(row["best_known_objective"], 1e-300)
                    for row in rows
                ]
            ),
            "iterations_to_tolerance": {
                **stats([censored(row, name, "iterations_to_tolerance", "total_iterations") for row in rows]),
                "censored": True,
                "non_reaching_censored_at": "total_iterations",
            },
            "function_evals_to_tolerance": {
                **stats(
                    [
                        censored(row, name, "function_evals_to_tolerance", "total_function_evals")
                        for row in rows
                    ]
                ),
                "censored": True,
                "non_reaching_censored_at": "total_function_evals",
            },
            "wall_seconds_to_tolerance": {
                **stats(
                    [
                        censored(row, name, "wall_seconds_to_tolerance", "wall_seconds_total")
                        for row in rows
                    ]
                ),
                "censored": True,
                "non_reaching_censored_at": "wall_seconds_total",
            },
            "total_iterations": stats([row[f"{name}_total_iterations"] for row in rows]),
            "total_function_evals": stats([row[f"{name}_total_function_evals"] for row in rows]),
            "wall_seconds_total": stats([row[f"{name}_wall_seconds_total"] for row in rows]),
        }
    return summary


def protocol_payload(
    args: argparse.Namespace,
    base: argparse.Namespace,
    rows: list[dict],
    sample_dates: list[pd.Timestamp],
    otm_info: dict,
    bounds: tuple[np.ndarray, np.ndarray],
) -> dict:
    lower, upper = bounds
    return {
        "script": A1.rel_path(Path(__file__)),
        "seed": int(args.seed),
        "fits": int(len(rows)),
        "sample_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in sample_dates],
        "otm": {
            "quotes": int(otm_info["otm_quotes"]),
            "date_expiry_fits": int(otm_info["date_expiry_fits"]),
            "selection": "same OTM quotes and sample-date expiry fits as approach2 Q1B",
        },
        "price_col": args.price_col,
        "start": "a = 0.5 * min(w), b = 0.1, rho = -0.3, m = 0, eta = 0.1, clipped to bounds",
        "bounds": {
            "lower": [float(item) for item in lower],
            "upper": [float(item) for item in upper],
            "eta_min": float(base.eta_min),
            "eta_max": float(base.eta_max),
        },
        "objective": "mean squared error in total variance w, float64",
        "torch_objective_scaling": {
            "constant": "c = 1 / MSE(cold start), recomputed per fit from the quotes alone",
            "applies_to": "torch Adam and torch L-BFGS objective",
            "final_objectives": "unscaled MSE",
            "minimizer_unchanged": True,
        },
        "solvers": {
            "adam": {
                "max_iterations": int(args.adam_steps),
                "learning_rate": float(args.adam_lr),
                "schedule": "cosine annealing to learning_rate * 1e-3",
                "dtype": "float64",
            },
            "lbfgs": {
                "max_iterations": int(args.lbfgs_steps),
                "line_search": "strong_wolfe",
                "history_size": 50,
                "max_eval": max(2 * args.lbfgs_steps, 20),
                "dtype": "float64",
            },
            "scipy": {
                "method": "trf",
                "jacobian": "analytic",
                "max_nfev": 4000,
                "ftol": 1e-12,
                "xtol": 1e-12,
                "gtol": 1e-12,
            },
        },
        "tolerance": {
            "relative": float(args.compare_tol_relative),
            "rule": "final objective <= (1 + relative) * best_known + 1e-15",
            "best_known": "minimum of the three cold-start runs and approach1's multi-start raw fit for that fit",
        },
        "timing": "wall seconds measured only inside each solver loop; data loading, summaries and figures are excluded",
        "summary_censoring": "non-reaching fits are kept: iterations, function evaluations and time to tolerance are censored at that solver's total budget, so medians and p95 use all fits",
        "cold_start_objective_bounds": {
            "objective_min": float(min(row["cold_objective"] for row in rows)),
            "objective_max": float(max(row["cold_objective"] for row in rows)),
            "torch_scale_min": float(min(row["torch_scale"] for row in rows)),
            "torch_scale_max": float(max(row["torch_scale"] for row in rows)),
        },
    }


def write_convergence_html(histories: list[dict], out_path: Path) -> None:
    ordered = sorted(histories, key=lambda item: (item["T"], item["date"], item["expiry"]))
    indices = sorted({0, len(ordered) // 2, len(ordered) - 1})
    selected = [ordered[index] for index in indices]
    colors = {"adam": "#1f77b4", "lbfgs": "#d62728", "scipy": "#2ca02c"}
    fig = make_subplots(
        rows=len(selected),
        cols=1,
        subplot_titles=[
            f"{item['date']} {item['expiry']}  T={item['T']:.3f}  "
            f"cold={item['cold_objective']:.2e}  best={item['best_known_objective']:.2e}"
            for item in selected
        ],
        vertical_spacing=0.09,
    )
    for row_index, item in enumerate(selected, start=1):
        for name in SOLVERS:
            trace = item["runs"][name]["trace"]
            x = np.arange(1, len(trace) + 1)
            y = np.asarray([entry[2] for entry in trace], dtype=float)
            y = np.maximum(np.minimum.accumulate(y), 1e-300)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=name,
                    line={"color": colors[name], "width": 1.6},
                    showlegend=row_index == 1,
                    legendgroup=name,
                ),
                row=row_index,
                col=1,
            )
        fig.add_hline(
            y=item["tolerance"],
            line={"color": "#555555", "dash": "dot", "width": 1.0},
            row=row_index,
            col=1,
        )
    fig.update_yaxes(type="log", title_text="best objective so far (MSE in w)")
    fig.update_xaxes(title_text="objective evaluations")
    fig.update_layout(
        title="Q1B SVI cold-start convergence (short, medium and long tenor)",
        height=330 * len(selected) + 140,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)


def load_fits(args: argparse.Namespace, base: argparse.Namespace):
    options, shibor, metadata = A1.load_data(base.options, base.shibor, base.metadata)
    forwards, _ = A1.estimate_forwards(options, shibor, args.price_col)
    data, _ = A1.apply_filters(options, forwards, base)
    iv, _, _ = A1.compute_implied_vols(data[data.liquid])
    sample_dates, _ = q1b.select_sample_dates(options, metadata)
    otm, otm_info = q1b.build_otm_quotes(data, iv, sample_dates, base.min_strikes)
    bounds = A1.svi_bounds(base.eta_min, base.eta_max)
    reference_starts, _ = q1b.load_reference_starts(base.approach1_dir, otm, sample_dates, bounds)
    fits = []
    for date in sample_dates:
        group = q1b.date_quotes(otm, date)
        for expiry, slice_group in group.groupby("expiry", sort=True):
            key = (
                pd.Timestamp(date).strftime("%Y-%m-%d"),
                pd.Timestamp(expiry).strftime("%Y-%m-%d"),
            )
            if key not in reference_starts:
                continue
            fits.append(
                {
                    "date": key[0],
                    "expiry": key[1],
                    "T": float(slice_group["T"].iloc[0]),
                    "n_quotes": int(len(slice_group)),
                    "k": slice_group.k.to_numpy(dtype=float),
                    "w": slice_group.w.to_numpy(dtype=float),
                    "reference": np.asarray(reference_starts[key]["raw"], dtype=float),
                    "reference_source": str(reference_starts[key]["source"]),
                }
            )
    if args.max_fits is not None:
        fits = fits[: int(args.max_fits)]
    return fits, sample_dates, otm_info, bounds


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = q1b.parse_args([])
    base.seed = int(args.seed)
    base.price_col = args.price_col
    q1b.set_seed(args.seed)
    q1b.set_torch_threads(base.torch_threads)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    console = Console(width=220)
    fits, sample_dates, otm_info, bounds = load_fits(args, base)
    if not fits:
        raise RuntimeError("no SVI fits available for the cold-start comparison")
    if not args.quiet:
        console.print(
            f"Cold-start SVI: {len(fits)} fits | OTM quotes={otm_info['otm_quotes']:,} | "
            f"dates={len(sample_dates)}"
        )
    rows = []
    histories = []
    for index, fit in enumerate(fits, start=1):
        row, history = row_from_fit(
            fit["date"],
            fit["expiry"],
            fit["T"],
            fit["n_quotes"],
            fit["k"],
            fit["w"],
            bounds,
            fit["reference"],
            fit["reference_source"],
            args,
        )
        rows.append(row)
        histories.append(history)
        if not args.quiet:
            console.print(
                f"  [{index}/{len(fits)}] {row['date']} {row['expiry']} T={row['T']:.3f} "
                f"cold={row['cold_objective']:.3e} best={row['best_known_objective']:.3e} "
                f"adam={row['adam_final_objective']:.3e} lbfgs={row['lbfgs_final_objective']:.3e} "
                f"scipy={row['scipy_final_objective']:.3e}"
            )
    summary = summarise(rows, args)
    protocol = protocol_payload(args, base, rows, sample_dates, otm_info, bounds)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "solver_comparison_cold.csv", index=False)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        payload = {"params": {}, "metrics": {}, "fit_seconds": 0.0, "history": {}}
    payload["params"]["svi_solver_cold_start"] = protocol
    payload["metrics"]["svi_solver_cold_start"] = summary
    metrics_path.write_text(
        json.dumps(A1.to_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_convergence_html(histories, out_dir / "svi_convergence_cold.html")
    if not args.quiet:
        table = Table(title="Cold-start SVI solver comparison")
        for column in ("solver", "reached", "iterations to tol. median/p95", "evals to tol. median/p95", "wall to tol. median/p95", "total wall median/p95", "final objective median/p95"):
            table.add_column(column, justify="right" if column != "solver" else "left")
        for name in SOLVERS:
            item = summary["solvers"][name]
            table.add_row(
                name,
                f"{item['reached']}/{summary['fits']}",
                f"{item['iterations_to_tolerance']['median']:.1f}/{item['iterations_to_tolerance']['p95']:.1f}",
                f"{item['function_evals_to_tolerance']['median']:.1f}/{item['function_evals_to_tolerance']['p95']:.1f}",
                f"{item['wall_seconds_to_tolerance']['median']:.4f}/{item['wall_seconds_to_tolerance']['p95']:.4f}",
                f"{item['wall_seconds_total']['median']:.4f}/{item['wall_seconds_total']['p95']:.4f}",
                f"{item['final_objective']['median']:.3e}/{item['final_objective']['p95']:.3e}",
            )
        console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
