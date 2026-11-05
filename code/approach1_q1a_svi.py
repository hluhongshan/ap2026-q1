"""Q1 Option A: arbitrage-free SVI implied-volatility surface for CFFEX IO options.

The pipeline is deterministic. It loads CFFEX IO option quotes and SHIBOR,
estimates the forward from put-call parity, filters liquid quotes, inverts
Black-Scholes implied volatilities, fits raw SVI slices, audits butterfly and
calendar no-arbitrage, repairs violations with penalised refits, and writes
all metrics and figures under ``results/approach1/<out>``.

SVI raw parameterisation
------------------------
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + eta^2)),
with b >= 0, |rho| < 1, eta > 0. Analytic derivatives used in the audits are

    w'(k)  = b * (rho + d / s)
    w''(k) = b * eta^2 / s^3,  d = k - m, s = sqrt(d^2 + eta^2).

The butterfly function is the correct Gatheral form

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2.

The brief's printed form omits the w''/2 term; both are reported.

Documented SVI starting points (a is solved so that w(0) = observed w near 0)
-----------------------------------------------------------------------------
1. flat ................ b=0.05, rho= 0.00, m= 0.00, eta=0.10
2. mild negative skew .. b=0.10, rho=-0.50, m= 0.00, eta=0.10
3. strong negative skew  b=0.20, rho=-0.70, m=-0.05, eta=0.15
4. steep short wing .... b=0.05, rho=-0.90, m= 0.05, eta=0.05
5. high curvature ...... b=0.30, rho= 0.00, m= 0.00, eta=0.20
6. positive skew ....... b=0.15, rho= 0.30, m= 0.00, eta=0.08
7. wide shallow smile .. b=0.02, rho=-0.30, m=-0.10, eta=0.25
8. moderate skew ....... b=0.10, rho=-0.50, m=-0.02, eta=0.20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]


def rel_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return str(candidate)


DEFAULT_OPTIONS = ROOT / "data" / "clean" / "options_io.csv"
DEFAULT_SHIBOR = ROOT / "data" / "clean" / "shibor.csv"
DEFAULT_METADATA = ROOT / "data" / "metadata.json"
DEFAULT_RAW_CFFEX = ROOT / "data" / "raw" / "cffex"
DEFAULT_OUT = ROOT / "results" / "approach1" / "io"
DEFAULT_ETA_MIN = 0.01
DEFAULT_ETA_MAX = 1.0
LEGACY_ETA_MIN = 1e-6
LEGACY_ETA_MAX = 2.0

PILLAR_DAYS = {"ON": 1, "1W": 7, "2W": 14, "1M": 30, "3M": 90, "6M": 180, "9M": 270, "1Y": 365}
GRID_K = np.round(np.arange(-0.14, 0.14 + 0.005, 0.01), 10)
SURFACE_K = np.round(np.arange(-0.20, 0.20 + 0.01, 0.02), 10)
SVI_STARTS = (
    ("flat", 0.05, 0.00, 0.00, 0.10),
    ("mild_negative_skew", 0.10, -0.50, 0.00, 0.10),
    ("strong_negative_skew", 0.20, -0.70, -0.05, 0.15),
    ("steep_short_wing", 0.05, -0.90, 0.05, 0.05),
    ("high_curvature", 0.30, 0.00, 0.00, 0.20),
    ("positive_skew", 0.15, 0.30, 0.00, 0.08),
    ("wide_shallow_smile", 0.02, -0.30, -0.10, 0.25),
    ("moderate_skew", 0.10, -0.50, -0.02, 0.20),
)
SVI_LOWER = np.array([-2.0, 0.0, -0.999, -2.0, DEFAULT_ETA_MIN])
SVI_UPPER = np.array([2.0, 5.0, 0.999, 2.0, DEFAULT_ETA_MAX])
BUTTERFLY_TOL = 1e-8
CALENDAR_TOL = 1e-8


def svi_bounds(eta_min: float, eta_max: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([-2.0, 0.0, -0.999, -2.0, eta_min], dtype=float)
    upper = np.array([2.0, 5.0, 0.999, 2.0, eta_max], dtype=float)
    return lower, upper
IV_GAP_REPORT = 0.01
PRICE_TOL = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 Option A: SVI surface on CFFEX IO options")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--price-col", choices=["close", "settle"], default="settle")
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--kband", type=float, default=0.20)
    parser.add_argument("--min-days", type=int, default=7)
    parser.add_argument("--min-strikes", type=int, default=6)
    parser.add_argument("--eta-min", type=float, default=DEFAULT_ETA_MIN)
    parser.add_argument("--eta-max", type=float, default=DEFAULT_ETA_MAX)
    parser.add_argument("--legacy-svi-params", type=Path, default=None)
    parser.add_argument("--raw-cffex", type=Path, default=DEFAULT_RAW_CFFEX)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--shibor", type=Path, default=DEFAULT_SHIBOR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(val) for val in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_data(options_path: Path, shibor_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    options = pd.read_csv(options_path, parse_dates=["date", "expiry"])
    shibor = pd.read_csv(shibor_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    metadata = json.loads(metadata_path.read_text())

    required_options = {"date", "contract", "expiry", "cp", "K", "open", "high", "low", "close", "settle", "volume", "oi", "S"}
    required_shibor = {"date", *PILLAR_DAYS.keys()}
    missing_options = required_options - set(options.columns)
    missing_shibor = required_shibor - set(shibor.columns)
    if missing_options:
        raise ValueError(f"options file missing columns: {sorted(missing_options)}")
    if missing_shibor:
        raise ValueError(f"shibor file missing columns: {sorted(missing_shibor)}")

    checks = {
        "rows": len(options),
        "duplicate_date_contract": int(options.duplicated(["date", "contract"]).sum()),
        "nonpositive_K": int((options.K <= 0).sum()),
        "expiry_not_after_date": int((options.expiry <= options.date).sum()),
        "missing_quotes": int(options[["close", "settle", "volume", "S"]].isna().sum().sum()),
    }
    if any(checks[key] for key in ("duplicate_date_contract", "nonpositive_K", "expiry_not_after_date", "missing_quotes")):
        raise ValueError(f"data verification failed: {checks}")
    if options.groupby(["date", "expiry", "K"]).cp.nunique().max() != 2:
        raise ValueError("some date/expiry/strike groups do not contain both C and P")

    return options, shibor, metadata


def build_rate_curves(shibor: pd.DataFrame) -> tuple[pd.DataFrame, dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]]:
    taus = np.array([PILLAR_DAYS[name] / 365.0 for name in PILLAR_DAYS], dtype=float)
    curves: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    zero_rows = []
    for _, row in shibor.iterrows():
        date = pd.Timestamp(row["date"])
        levels = np.array([row[name] for name in PILLAR_DAYS], dtype=float)
        if not np.all(np.isfinite(levels)):
            continue
        zeros = np.log1p((levels / 100.0) * taus * (365.0 / 360.0)) / taus
        curves[date] = (taus, zeros)
        for name, tau, level, zero in zip(PILLAR_DAYS, taus, levels, zeros):
            zero_rows.append((date, name, tau, level, zero))
    return pd.DataFrame(zero_rows, columns=["date", "pillar", "tau", "level_pct", "zero_rate"]), curves


def zero_rate(curves: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]], sorted_dates: np.ndarray, date: pd.Timestamp, T: float) -> tuple[float, pd.Timestamp]:
    idx = int(np.searchsorted(sorted_dates, np.datetime64(date), side="right") - 1)
    if idx < 0:
        idx = 0
    curve_date = pd.Timestamp(sorted_dates[idx])
    taus, zeros = curves[curve_date]
    return float(np.interp(T, taus, zeros)), curve_date


def estimate_forward_pair(
    K: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    vol_c: np.ndarray,
    vol_p: np.ndarray,
    DF: float,
    S0: float,
    r: float,
    T: float,
    band: float = 0.05,
) -> dict:
    Fk = K + (C - P) / DF
    positive = np.isfinite(Fk) & (Fk > 0)
    if not np.any(positive):
        F = S0 / DF
        return {
            "F": float(F),
            "F_median": float(F),
            "Fk": Fk,
            "in_band": np.zeros_like(Fk, dtype=bool),
            "fallback": True,
            "resid_S0": C - P - (S0 - K * np.exp(-r * T)),
            "resid_F": C - P - DF * (F - K),
        }
    median_f = float(np.median(Fk[positive]))
    in_band = positive & (np.abs(np.log(Fk / median_f)) <= band)
    weights = np.maximum(vol_c + vol_p, 0.0)
    if np.sum(weights[in_band]) > 0:
        F = float(np.sum(weights[in_band] * Fk[in_band]) / np.sum(weights[in_band]))
    else:
        F = float(np.mean(Fk[in_band]))
    return {
        "F": F,
        "F_median": median_f,
        "Fk": Fk,
        "in_band": in_band,
        "fallback": False,
        "resid_S0": C - P - (S0 - K * np.exp(-r * T)),
        "resid_F": C - P - DF * (F - K),
    }


def estimate_forwards(options: pd.DataFrame, shibor: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    zero_df, curves = build_rate_curves(shibor)
    sorted_dates = zero_df.date.drop_duplicates().sort_values().to_numpy()
    forward_rows = []
    pair_rows = []
    for (date, expiry), group in options.groupby(["date", "expiry"], sort=True):
        T = (expiry - date).days / 365.0
        r, curve_date = zero_rate(curves, sorted_dates, date, T)
        DF = math.exp(-r * T)
        S0 = float(group.S.iloc[0])
        piv = group.pivot_table(index="K", columns="cp", values=price_col, aggfunc="first")
        vols = group.pivot_table(index="K", columns="cp", values="volume", aggfunc="first")
        if "C" not in piv.columns or "P" not in piv.columns:
            continue
        pair = piv[["C", "P"]].copy()
        pair["C_vol"] = vols["C"]
        pair["P_vol"] = vols["P"]
        pair = pair.dropna()
        if pair.empty:
            continue
        K = pair.index.to_numpy(dtype=float)
        C = pair["C"].to_numpy(dtype=float)
        P = pair["P"].to_numpy(dtype=float)
        vol_c = pair["C_vol"].to_numpy(dtype=float) if "C_vol" in pair.columns else pair["C"].to_numpy(dtype=float) * 0.0
        vol_p = pair["P_vol"].to_numpy(dtype=float) if "P_vol" in pair.columns else pair["P"].to_numpy(dtype=float) * 0.0
        est = estimate_forward_pair(K, C, P, vol_c, vol_p, DF, S0, r, T)
        q = r - math.log(est["F"] / S0) / T
        forward_rows.append({
            "date": date,
            "expiry": expiry,
            "T": T,
            "days": int((expiry - date).days),
            "r": r,
            "curve_date": curve_date,
            "DF": DF,
            "S0": S0,
            "F": est["F"],
            "F_median": est["F_median"],
            "q": q,
            "n_pairs": len(pair),
            "n_in_band": int(est["in_band"].sum()),
            "fallback_F": est["fallback"],
        })
        for i, k_value in enumerate(K):
            pair_rows.append({
                "date": date,
                "expiry": expiry,
                "K": k_value,
                "C": C[i],
                "P": P[i],
                "volume_C": vol_c[i],
                "volume_P": vol_p[i],
                "F_K": est["Fk"][i],
                "in_band": bool(est["in_band"][i]),
                "resid_S0": est["resid_S0"][i],
                "resid_F": est["resid_F"][i],
                "F": est["F"],
                "T": T,
                "r": r,
                "DF": DF,
                "S0": S0,
                "q": q,
            })
    return pd.DataFrame(forward_rows), pd.DataFrame(pair_rows)


def apply_filters(options: pd.DataFrame, forwards: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    data = options.merge(forwards[["date", "expiry", "T", "days", "r", "DF", "S0", "F", "q"]], on=["date", "expiry"], how="left")
    if data[["F", "T", "DF"]].isna().any().any():
        raise ValueError("forward estimate missing for some option rows")
    data["price"] = data[args.price_col]
    data["k"] = np.log(data.K / data.F)
    data["pass_volume"] = data.volume >= args.min_volume
    data["pass_price"] = data.price >= args.min_price
    data["pass_kband"] = np.abs(data.k) <= args.kband
    data["pass_days"] = data.days >= args.min_days
    flags = ["pass_volume", "pass_price", "pass_kband", "pass_days"]
    data["liquid"] = data[flags].all(axis=1)

    independent = {name: int((~data[name]).sum()) for name in flags}
    sequential: dict[str, int] = {}
    remaining = np.ones(len(data), dtype=bool)
    for name in flags:
        removed = int((~data[name] & remaining).sum())
        sequential[name] = removed
        remaining &= data[name].to_numpy()
    counts = {
        "quotes_total": int(len(data)),
        "quotes_liquid": int(data.liquid.sum()),
        "independent_removals": independent,
        "sequential_removals": sequential,
        "date_expiry_groups_total": int(data.groupby(["date", "expiry"]).ngroups),
        "date_expiry_groups_liquid": int(data[data.liquid].groupby(["date", "expiry"]).ngroups),
    }
    return data, counts


def bs_price(cp: str, F: float, K: float, T: float, DF: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0 or K <= 0 or F <= 0:
        return float("nan")
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if cp == "C":
        return float(DF * (F * ndtr(d1) - K * ndtr(d2)))
    return float(DF * (K * ndtr(-d2) - F * ndtr(-d1)))


def implied_vol(price: float, cp: str, F: float, K: float, T: float, DF: float) -> tuple[float, str]:
    if not np.isfinite(price) or price <= 0:
        return float("nan"), "below_intrinsic"
    if cp == "C":
        intrinsic = DF * max(F - K, 0.0)
        upper = DF * F
    else:
        intrinsic = DF * max(K - F, 0.0)
        upper = DF * K
    scale = max(1.0, abs(F), abs(K))
    if price <= intrinsic + PRICE_TOL * scale:
        return float("nan"), "below_intrinsic"
    if price >= upper - PRICE_TOL * scale:
        return float("nan"), "above_upper_bound"

    def objective(sigma: float) -> float:
        return bs_price(cp, F, K, T, DF, sigma) - price

    low, high = 1e-4, 5.0
    try:
        f_low = objective(low)
        f_high = objective(high)
        if f_low > 0:
            return float("nan"), "below_intrinsic"
        if f_high < 0:
            return float("nan"), "above_upper_bound"
        sigma = brentq(objective, low, high, xtol=1e-13, rtol=1e-13, maxiter=200)
        if not np.isfinite(sigma):
            return float("nan"), "brentq_failed"
        return float(sigma), "ok"
    except (ValueError, RuntimeError):
        return float("nan"), "brentq_failed"


def compute_implied_vols(liquid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for row in liquid.itertuples(index=False):
        sigma, status = implied_vol(row.price, row.cp, row.F, row.K, row.T, row.DF)
        rows.append({
            "date": row.date,
            "expiry": row.expiry,
            "K": row.K,
            "cp": row.cp,
            "price": row.price,
            "volume": row.volume,
            "F": row.F,
            "T": row.T,
            "DF": row.DF,
            "k": row.k,
            "iv": sigma,
            "status": status,
        })
    iv = pd.DataFrame(rows)
    calls = iv[iv.cp == "C"][["date", "expiry", "K", "iv", "status", "volume", "F", "T", "DF", "k"]].rename(
        columns={"iv": "iv_c", "status": "status_c", "volume": "volume_c", "k": "k_c"}
    )
    puts = iv[iv.cp == "P"][["date", "expiry", "K", "iv", "status", "volume", "k"]].rename(
        columns={"iv": "iv_p", "status": "status_p", "volume": "volume_p", "k": "k_p"}
    )
    pairs = calls.merge(puts, on=["date", "expiry", "K"], how="outer")
    pairs["iv_gap"] = pairs.iv_c - pairs.iv_p
    failure_states = {"below_intrinsic", "above_upper_bound", "brentq_failed"}
    failure_mask = pairs.status_c.isin(failure_states) | pairs.status_p.isin(failure_states)
    gap_mask = (pairs.status_c == "ok") & (pairs.status_p == "ok") & (pairs.iv_gap.abs() > IV_GAP_REPORT)
    divergence = pairs[failure_mask | gap_mask].copy()
    divergence_failure = divergence.status_c.isin(failure_states) | divergence.status_p.isin(failure_states)
    divergence_gap = (
        (divergence.status_c == "ok")
        & (divergence.status_p == "ok")
        & (divergence.iv_gap.abs() > IV_GAP_REPORT)
    )
    divergence["reason"] = np.where(
        divergence_failure,
        "iv_failure",
        np.where(divergence_gap, "call_put_gap", "other"),
    )
    status_counts = {}
    for cp in ("C", "P"):
        subset = iv[iv.cp == cp]
        status_counts[cp] = {str(key): int(val) for key, val in subset.status.value_counts().items()}
    both = pairs[(pairs.status_c == "ok") & (pairs.status_p == "ok")].copy()
    gap_stats = {
        "attempted_quotes": int(len(iv)),
        "ok_quotes": int((iv.status == "ok").sum()),
        "status_counts": status_counts,
        "paired_strikes": int(len(pairs)),
        "gap_available": int(len(both)),
        "gap_median": float(both.iv_gap.median()) if len(both) else float("nan"),
        "gap_mean": float(both.iv_gap.mean()) if len(both) else float("nan"),
        "gap_p95_abs": float(both.iv_gap.abs().quantile(0.95)) if len(both) else float("nan"),
        "gap_max_abs": float(both.iv_gap.abs().max()) if len(both) else float("nan"),
        "n_gap_gt_0_01": int((both.iv_gap.abs() > 0.01).sum()),
        "n_gap_gt_0_05": int((both.iv_gap.abs() > 0.05).sum()),
    }
    return iv, divergence, gap_stats


def svi_w(k: np.ndarray | float, params: np.ndarray) -> np.ndarray:
    a, b, rho, m, eta = params
    d = np.asarray(k) - m
    return a + b * (rho * d + np.sqrt(d * d + eta * eta))


def svi_g(k: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, rho, m, eta = params
    k = np.asarray(k, dtype=float)
    d = k - m
    s = np.sqrt(d * d + eta * eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = a + b * (rho * d + s)
        w1 = b * (rho + d / s)
        w2 = b * eta * eta / (s * s * s)
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 1.0 / 4.0) + w2 / 2.0
        g_printed = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 1.0 / 4.0)
    return w, w1, w2, g, g_printed


def _start_params(w0: float, b: float, rho: float, m: float, eta: float) -> np.ndarray:
    a = w0 - b * (rho * (0.0 - m) + math.sqrt(m * m + eta * eta))
    return np.array([a, b, rho, m, eta], dtype=float)


def build_starting_points(k: np.ndarray, w: np.ndarray) -> list[tuple[str, np.ndarray]]:
    w0 = float(w[int(np.argmin(np.abs(k)))])
    starts = []
    for label, b, rho, m, eta in SVI_STARTS:
        params = _start_params(w0, b, rho, m, eta)
        starts.append((label, params))
    return starts


def fit_svi(
    k: np.ndarray,
    w: np.ndarray,
    k_grid: np.ndarray | None = None,
    penalty=None,
    x0_params: np.ndarray | None = None,
    include_starts: bool = True,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict | None:
    if bounds is None:
        bounds = svi_bounds(DEFAULT_ETA_MIN, DEFAULT_ETA_MAX)
    lower, upper = bounds
    starts: list[tuple[str, np.ndarray]] = []
    if x0_params is not None:
        starts.append(("warm_start", np.asarray(x0_params, dtype=float)))
    if include_starts:
        starts.extend(build_starting_points(k, w))

    def objective(params: np.ndarray) -> np.ndarray:
        residual = svi_w(k, params) - w
        if penalty is None:
            return residual
        return np.concatenate([residual, penalty(params, k_grid)])

    best = None
    for label, x0 in starts:
        x0 = np.clip(x0, lower + 1e-10, upper - 1e-10)
        try:
            sol = least_squares(
                objective,
                x0,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                max_nfev=4000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        params = np.asarray(sol.x, dtype=float)
        rmse = float(np.sqrt(np.mean((svi_w(k, params) - w) ** 2)))
        candidate = {
            "params": params,
            "cost": float(sol.cost),
            "rmse": rmse,
            "nfev": int(sol.nfev),
            "success": bool(sol.success),
            "start": label,
            "message": str(sol.message),
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate
    return best


def audit_svi(params: np.ndarray, k_grid: np.ndarray, tol: float = BUTTERFLY_TOL) -> pd.DataFrame:
    w, w1, w2, g, g_printed = svi_g(k_grid, params)
    return pd.DataFrame({
        "k": k_grid,
        "w": w,
        "w1": w1,
        "w2": w2,
        "g": g,
        "g_printed": g_printed,
        "w_nonpositive": ~(w > 0),
        "violates_correct_strict": np.isfinite(g) & (g < 0.0),
        "violates_correct": np.isfinite(g) & (g < -tol),
        "violates_printed_strict": np.isfinite(g_printed) & (g_printed < 0.0),
        "violates_printed": np.isfinite(g_printed) & (g_printed < -tol),
    })


def audit_summary(audit: pd.DataFrame) -> dict:
    return {
        "butterfly_violations_correct_strict": int(audit.violates_correct_strict.sum()),
        "butterfly_violations_correct": int(audit.violates_correct.sum()),
        "butterfly_violations_printed_strict": int(audit.violates_printed_strict.sum()),
        "butterfly_violations_printed": int(audit.violates_printed.sum()),
        "w_nonpositive": int(audit.w_nonpositive.sum()),
        "min_g": float(audit.g.min()) if np.isfinite(audit.g).any() else float("nan"),
        "min_w": float(audit.w.min()) if np.isfinite(audit.w).any() else float("nan"),
    }


def penalty_residual(
    params: np.ndarray,
    k_grid: np.ndarray,
    butterfly_weight: float,
    w_weight: float = 1e4,
    prev_w: np.ndarray | None = None,
    calendar_weight: float = 0.0,
) -> np.ndarray:
    w, _, _, g, _ = svi_g(k_grid, params)
    g_safe = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    butterfly_penalty = np.sqrt(butterfly_weight) * np.maximum(0.0, -g_safe)
    w_penalty = np.sqrt(w_weight) * np.maximum(0.0, 1e-8 - np.nan_to_num(w, nan=-1.0, posinf=0.0, neginf=-1.0))
    pieces = [butterfly_penalty, w_penalty]
    if prev_w is not None and calendar_weight > 0:
        calendar_penalty = np.sqrt(calendar_weight) * np.maximum(0.0, prev_w - np.nan_to_num(w, nan=-1.0, posinf=0.0, neginf=-1.0))
        pieces.append(calendar_penalty)
    return np.concatenate(pieces)


def fit_constraint_score(params: np.ndarray, k: np.ndarray, w: np.ndarray, prev_w: np.ndarray | None = None) -> tuple:
    audit = audit_svi(params, GRID_K)
    n_bf = int(audit.violates_correct.sum())
    n_w = int(audit.w_nonpositive.sum())
    n_cal = int((prev_w - svi_w(GRID_K, params) > CALENDAR_TOL).sum()) if prev_w is not None else 0
    max_bf = float(max(0.0, -audit.g[audit.violates_correct].min())) if n_bf else 0.0
    max_cal = float(max(0.0, (prev_w - svi_w(GRID_K, params)).max())) if prev_w is not None else 0.0
    rmse = float(np.sqrt(np.mean((svi_w(k, params) - w) ** 2)))
    return (n_bf + n_w + n_cal, n_bf + n_w + n_cal, max_bf + max_cal, rmse)


def repair_svi(
    k: np.ndarray,
    w: np.ndarray,
    raw_params: np.ndarray,
    current_params: np.ndarray,
    prev_w: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    best_params = np.asarray(current_params, dtype=float)
    best_score = fit_constraint_score(best_params, k, w, prev_w)
    levels = (1e2, 1e4, 1e6)
    attempts = 0
    for level in levels:
        starts = [best_params, np.asarray(raw_params, dtype=float)]
        for start in starts:
            def penalty(params, grid, level=level, prev_w=prev_w):
                return penalty_residual(
                    params,
                    grid,
                    butterfly_weight=level,
                    w_weight=level,
                    prev_w=prev_w,
                    calendar_weight=level if prev_w is not None else 0.0,
                )
            candidate = fit_svi(
                k,
                w,
                k_grid=GRID_K,
                penalty=penalty,
                x0_params=start,
                include_starts=False,
                bounds=bounds,
            )
            attempts += 1
            if candidate is None:
                continue
            score = fit_constraint_score(candidate["params"], k, w, prev_w)
            if score < best_score:
                best_score = score
                best_params = candidate["params"]
        if best_score[0] == 0:
            break
    return best_params, {
        "attempts": attempts,
        "feasible": bool(best_score[0] == 0),
        "remaining_constraint_count": int(best_score[0]),
        "max_violation": float(best_score[2]),
        "rmse": float(np.sqrt(np.mean((svi_w(k, best_params) - w) ** 2))),
    }


def make_fits(
    forwards: pd.DataFrame,
    data: pd.DataFrame,
    iv: pd.DataFrame,
    min_strikes: int,
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[dict, pd.DataFrame, dict]:
    from collections import Counter

    fit_input_rows = []
    for row in data[data.liquid].itertuples(index=False):
        if row.cp == "C" and row.K >= row.F:
            fit_input_rows.append(row)
        elif row.cp == "P" and row.K < row.F:
            fit_input_rows.append(row)
    otm = pd.DataFrame([{
        "date": row.date,
        "expiry": row.expiry,
        "K": row.K,
        "cp": row.cp,
        "price": row.price,
        "volume": row.volume,
        "k": row.k,
        "F": row.F,
        "T": row.T,
    } for row in fit_input_rows])
    if otm.empty:
        raise ValueError("no OTM liquid quotes after filtering")
    otm = otm.merge(iv[["date", "expiry", "K", "cp", "iv", "status"]], on=["date", "expiry", "K", "cp"], how="left")
    otm = otm[(otm.status == "ok") & np.isfinite(otm.iv) & (otm.iv > 0)].copy()
    otm["w"] = otm.iv ** 2 * otm["T"]
    otm = otm.sort_values(["date", "expiry", "K", "cp"]).drop_duplicates(["date", "expiry", "K"], keep="first")

    fwd_map = forwards.set_index(["date", "expiry"])
    fits: dict[tuple[str, str], dict] = {}
    skipped = []
    for (date, expiry), group in otm.groupby(["date", "expiry"], sort=True):
        key = (date.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d"))
        fwd = fwd_map.loc[(date, expiry)]
        if len(group) < min_strikes:
            skipped.append({
                "date": key[0],
                "expiry": key[1],
                "n_strikes": int(len(group)),
                "reason": "fewer_than_min_strikes",
                "T": float(fwd["T"]),
                "F": float(fwd["F"]),
            })
            continue
        k = group.k.to_numpy(dtype=float)
        w = group.w.to_numpy(dtype=float)
        raw = fit_svi(k, w, bounds=bounds)
        if raw is None:
            skipped.append({
                "date": key[0],
                "expiry": key[1],
                "n_strikes": int(len(group)),
                "reason": "raw_fit_failed",
                "T": float(fwd["T"]),
                "F": float(fwd["F"]),
            })
            continue
        raw_audit = audit_svi(raw["params"], GRID_K)
        fits[key] = {
            "date": key[0],
            "expiry": key[1],
            "T": float(fwd["T"]),
            "F": float(fwd["F"]),
            "DF": float(fwd["DF"]),
            "S0": float(fwd["S0"]),
            "q": float(fwd["q"]),
            "k": k,
            "w": w,
            "iv": group.iv.to_numpy(dtype=float),
            "n_strikes": int(len(group)),
            "raw": raw,
            "raw_audit_frame": raw_audit,
            "raw_audit": audit_summary(raw_audit),
            "repaired": None,
            "repaired_audit_frame": None,
            "repaired_audit": None,
            "repair_info": None,
            "calendar_repair_info": None,
        }

    present_keys = {(date.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d")) for date, expiry in otm[["date", "expiry"]].drop_duplicates().itertuples(index=False)}
    for (date, expiry) in forwards[["date", "expiry"]].itertuples(index=False):
        key = (date.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d"))
        if key in present_keys:
            continue
        fwd = fwd_map.loc[(date, expiry)]
        skipped.append({
            "date": key[0],
            "expiry": key[1],
            "n_strikes": 0,
            "reason": "no_otm_liquid_quotes",
            "T": float(fwd["T"]),
            "F": float(fwd["F"]),
        })
    skip_reasons = dict(Counter(item["reason"] for item in skipped))

    repair_history = []
    for key, fit in fits.items():
        raw_params = fit["raw"]["params"]
        raw_audit = fit["raw_audit"]
        needs_repair = raw_audit["butterfly_violations_correct"] > 0 or raw_audit["w_nonpositive"] > 0
        if needs_repair:
            repaired_params, info = repair_svi(
                fit["k"], fit["w"], raw_params, raw_params, prev_w=None, bounds=bounds
            )
            fit["repair_info"] = info
        else:
            repaired_params = raw_params
            fit["repair_info"] = {
                "attempts": 0,
                "feasible": True,
                "remaining_constraint_count": 0,
                "max_violation": 0.0,
                "rmse": fit["raw"]["rmse"],
            }
        repaired_audit = audit_svi(repaired_params, GRID_K)
        fit["repaired"] = {
            "params": repaired_params,
            "rmse": float(np.sqrt(np.mean((svi_w(fit["k"], repaired_params) - fit["w"]) ** 2))),
            "nfev": fit["raw"]["nfev"] if not needs_repair else 0,
            "success": True,
            "start": "raw" if not needs_repair else "penalised_warm_start",
            "message": "unchanged" if not needs_repair else "penalised_refit",
        }
        fit["repaired_audit_frame"] = repaired_audit
        fit["repaired_audit"] = audit_summary(repaired_audit)
        repair_history.append({
            "date": fit["date"],
            "expiry": fit["expiry"],
            "needed_repair": bool(needs_repair),
            "raw_butterfly": raw_audit["butterfly_violations_correct"],
            "raw_w_nonpositive": raw_audit["w_nonpositive"],
            "repaired_butterfly": fit["repaired_audit"]["butterfly_violations_correct"],
            "repaired_w_nonpositive": fit["repaired_audit"]["w_nonpositive"],
            **fit["repair_info"],
        })

    by_date: dict[str, list[tuple[str, str]]] = {}
    for key in fits:
        by_date.setdefault(key[0], []).append(key)
    calendar_history = []
    for date, keys in sorted(by_date.items()):
        keys.sort(key=lambda item: fits[item]["T"])
        for _pass in range(2):
            changed = False
            prev_fit = fits[keys[0]]
            prev_w = svi_w(GRID_K, prev_fit["repaired"]["params"])
            for key in keys[1:]:
                fit = fits[key]
                current_w = svi_w(GRID_K, fit["repaired"]["params"])
                violations = int((prev_w - current_w > CALENDAR_TOL).sum())
                if violations > 0:
                    raw_params = fit["raw"]["params"]
                    current_params = fit["repaired"]["params"]
                    calibrated_params, info = repair_svi(
                        fit["k"], fit["w"], raw_params, current_params, prev_w=prev_w, bounds=bounds
                    )
                    old_rmse = fit["repaired"]["rmse"]
                    fit["repaired"] = {
                        "params": calibrated_params,
                        "rmse": float(np.sqrt(np.mean((svi_w(fit["k"], calibrated_params) - fit["w"]) ** 2))),
                        "nfev": 0,
                        "success": True,
                        "start": "calendar_penalised_warm_start",
                        "message": "calendar_penalised_refit",
                    }
                    fit["repaired_audit_frame"] = audit_svi(calibrated_params, GRID_K)
                    fit["repaired_audit"] = audit_summary(fit["repaired_audit_frame"])
                    fit["calendar_repair_info"] = {
                        "date": date,
                        "expiry": key[1],
                        "previous_expiry": prev_fit["expiry"],
                        "violations_before": violations,
                        "violations_after": int((prev_w - svi_w(GRID_K, calibrated_params) > CALENDAR_TOL).sum()),
                        "rmse_before": old_rmse,
                        "rmse_after": fit["repaired"]["rmse"],
                        **info,
                    }
                    calendar_history.append(fit["calendar_repair_info"])
                    changed = True
                prev_fit = fit
                prev_w = svi_w(GRID_K, fit["repaired"]["params"])
            if not changed:
                break

    raw_repaired_params = pd.DataFrame([{
        "date": f["date"],
        "expiry": f["expiry"],
        "T": f["T"],
        "F": f["F"],
        "n_strikes": f["n_strikes"],
        "a_raw": f["raw"]["params"][0],
        "b_raw": f["raw"]["params"][1],
        "rho_raw": f["raw"]["params"][2],
        "m_raw": f["raw"]["params"][3],
        "eta_raw": f["raw"]["params"][4],
        "rmse_raw": f["raw"]["rmse"],
        "start_raw": f["raw"]["start"],
        "nfev_raw": f["raw"]["nfev"],
        "a_repaired": f["repaired"]["params"][0],
        "b_repaired": f["repaired"]["params"][1],
        "rho_repaired": f["repaired"]["params"][2],
        "m_repaired": f["repaired"]["params"][3],
        "eta_repaired": f["repaired"]["params"][4],
        "rmse_repaired": f["repaired"]["rmse"],
        "repair_attempts": f["repair_info"]["attempts"],
        "repair_feasible": f["repair_info"]["feasible"],
        "calendar_refit": bool(f["calendar_repair_info"] is not None),
    } for f in fits.values()])
    raw_repaired_params = raw_repaired_params.sort_values(["date", "T"]).reset_index(drop=True)
    history = {
        "repair": repair_history,
        "calendar_repair": calendar_history,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "otm_counts": {
            "otm_quotes_used": int(len(otm)),
            "fits": int(len(fits)),
            "skipped": int(len(skipped)),
        },
    }
    return fits, raw_repaired_params, history


def butterfly_rows(fits: dict, stage: str) -> pd.DataFrame:
    rows = []
    for fit in fits.values():
        frame = fit[f"{stage}_audit_frame"].copy()
        frame.insert(0, "stage", stage)
        frame.insert(1, "date", fit["date"])
        frame.insert(2, "expiry", fit["expiry"])
        frame.insert(3, "T", fit["T"])
        frame["F"] = fit["F"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calendar_rows(fits: dict, stage: str) -> pd.DataFrame:
    rows = []
    by_date: dict[str, list[dict]] = {}
    for fit in fits.values():
        by_date.setdefault(fit["date"], []).append(fit)
    for date, items in sorted(by_date.items()):
        items.sort(key=lambda item: item["T"])
        for left, right in zip(items[:-1], items[1:]):
            w1 = svi_w(GRID_K, left[f"{stage}"]["params"])
            w2 = svi_w(GRID_K, right[f"{stage}"]["params"])
            forward_var = (w2 - w1) / (right["T"] - left["T"])
            rows.append(pd.DataFrame({
                "stage": stage,
                "date": date,
                "expiry_1": left["expiry"],
                "expiry_2": right["expiry"],
                "T_1": left["T"],
                "T_2": right["T"],
                "k": GRID_K,
                "w_1": w1,
                "w_2": w2,
                "forward_variance": forward_var,
                "violates_calendar_strict": w2 < w1,
                "violates_calendar": w2 < w1 - CALENDAR_TOL,
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calendar_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "checks": 0,
            "violations": 0,
            "violations_strict": 0,
            "min_forward_variance": float("nan"),
            "locations": 0,
            "locations_strict": 0,
        }
    violations = frame[frame.violates_calendar]
    violations_strict = frame[frame.violates_calendar_strict]
    return {
        "checks": int(len(frame)),
        "violations": int(frame.violates_calendar.sum()),
        "violations_strict": int(frame.violates_calendar_strict.sum()),
        "min_forward_variance": float(frame.forward_variance.min()),
        "locations": int(len(violations)),
        "locations_strict": int(len(violations_strict)),
    }


def repricing_for_stage(fits: dict, liquid: pd.DataFrame, stage: str, basis: str) -> tuple[pd.DataFrame, dict]:
    """Reprice liquid quotes from one fitted slice.

    Both bases are evaluated on the same strikes: those with a liquid OTM
    option and a liquid call. ``otm`` uses the traded OTM option (put below F,
    call at/above F); ``calls`` uses the traded call. No synthetic call is made.
    """
    rows = []
    for fit in fits.values():
        mask = (
            (liquid.date == pd.Timestamp(fit["date"]))
            & (liquid.expiry == pd.Timestamp(fit["expiry"]))
        )
        group = liquid[mask]
        if group.empty:
            rows.append({
                "basis": basis,
                "stage": stage,
                "date": fit["date"],
                "expiry": fit["expiry"],
                "n_strikes": 0,
                "n_otm_available": 0,
                "n_calls_available": 0,
                "n_common_strikes": 0,
                "max_rel_error": np.nan,
                "median_rel_error": np.nan,
                "worst_K": np.nan,
                "fit_rmse": fit[stage]["rmse"],
            })
            continue
        calls = (
            group[group.cp == "C"][["K", "price"]]
            .rename(columns={"price": "call_price"})
            .drop_duplicates("K")
        )
        otm = group[
            ((group.cp == "C") & (group.K >= fit["F"]))
            | ((group.cp == "P") & (group.K < fit["F"]))
        ][["K", "cp", "price"]].rename(columns={"cp": "otm_cp", "price": "otm_price"}).drop_duplicates("K")
        common = otm.merge(calls, on="K", how="inner")
        common = common[(common.otm_price > 0) & np.isfinite(common.otm_price)]
        if basis == "calls":
            common = common[(common.call_price > 0) & np.isfinite(common.call_price)]
        if common.empty:
            rows.append({
                "basis": basis,
                "stage": stage,
                "date": fit["date"],
                "expiry": fit["expiry"],
                "n_strikes": 0,
                "n_otm_available": int(len(otm)),
                "n_calls_available": int(len(calls)),
                "n_common_strikes": int(len(common)),
                "max_rel_error": np.nan,
                "median_rel_error": np.nan,
                "worst_K": np.nan,
                "fit_rmse": fit[stage]["rmse"],
            })
            continue
        k = np.log(common.K.to_numpy(dtype=float) / fit["F"])
        w = svi_w(k, fit[stage]["params"])
        with np.errstate(invalid="ignore", divide="ignore"):
            iv = np.sqrt(w / fit["T"])
        if basis == "otm":
            cp_values = common.otm_cp.to_numpy(dtype=str)
            market = common.otm_price.to_numpy(dtype=float)
        else:
            cp_values = np.full(len(common), "C", dtype=object)
            market = common.call_price.to_numpy(dtype=float)
        model = np.array([
            bs_price(cp, fit["F"], K, fit["T"], fit["DF"], sigma)
            for cp, K, sigma in zip(cp_values, common.K, iv)
        ])
        valid = np.isfinite(model) & np.isfinite(iv) & (market > 0)
        if not np.any(valid):
            rows.append({
                "basis": basis,
                "stage": stage,
                "date": fit["date"],
                "expiry": fit["expiry"],
                "n_strikes": 0,
                "n_otm_available": int(len(otm)),
                "n_calls_available": int(len(calls)),
                "n_common_strikes": int(len(common)),
                "max_rel_error": np.nan,
                "median_rel_error": np.nan,
                "worst_K": np.nan,
                "fit_rmse": fit[stage]["rmse"],
            })
            continue
        rel = np.abs(model[valid] - market[valid]) / market[valid]
        K_valid = common.K.to_numpy(dtype=float)[valid]
        rows.append({
            "basis": basis,
            "stage": stage,
            "date": fit["date"],
            "expiry": fit["expiry"],
            "n_strikes": int(valid.sum()),
            "n_otm_available": int(len(otm)),
            "n_calls_available": int(len(calls)),
            "n_common_strikes": int(len(common)),
            "max_rel_error": float(np.max(rel)),
            "median_rel_error": float(np.median(rel)),
            "worst_K": float(K_valid[int(np.argmax(rel))]),
            "fit_rmse": fit[stage]["rmse"],
        })
    frame = pd.DataFrame(rows)
    summary = {
        "fits": int(len(frame)),
        "n_strikes_total": int(frame.n_strikes.sum()) if len(frame) else 0,
        "n_strikes_median": float(frame.n_strikes.median()) if len(frame) else float("nan"),
        "per_fit_max_median": float(frame.max_rel_error.median()) if len(frame) else float("nan"),
        "per_fit_max_p95": float(frame.max_rel_error.quantile(0.95)) if len(frame) else float("nan"),
        "per_fit_median_median": float(frame.median_rel_error.median()) if len(frame) else float("nan"),
        "per_fit_median_p95": float(frame.median_rel_error.quantile(0.95)) if len(frame) else float("nan"),
        "worst_fit_max": float(frame.max_rel_error.max()) if len(frame) else float("nan"),
    }
    return frame, summary


def surface_frames(fits: dict, stage: str) -> pd.DataFrame:
    rows = []
    for fit in fits.values():
        params = fit[stage]["params"]
        w = svi_w(SURFACE_K, params)
        with np.errstate(invalid="ignore", divide="ignore"):
            iv = np.sqrt(w / fit["T"])
        rows.append(pd.DataFrame({
            "date": fit["date"],
            "expiry": fit["expiry"],
            "T": fit["T"],
            "F": fit["F"],
            "k": SURFACE_K,
            "w": w,
            "iv": iv,
            "stage": stage,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def term_structure_rows(fits: dict) -> pd.DataFrame:
    rows = []
    by_date: dict[str, list[dict]] = {}
    for fit in fits.values():
        by_date.setdefault(fit["date"], []).append(fit)
    k_grid = np.round(np.arange(-0.14, 0.14 + 0.005, 0.02), 10)
    for date, items in sorted(by_date.items()):
        items.sort(key=lambda item: item["T"])
        if len(items) < 2:
            continue
        Ts = np.array([item["T"] for item in items])
        query_T = (Ts[:-1] + Ts[1:]) / 2.0
        for k in k_grid:
            ws = np.array([float(svi_w(k, item["repaired"]["params"])) for item in items])
            interp = np.interp(query_T, Ts, ws)
            rows.append(pd.DataFrame({
                "date": date,
                "k": k,
                "T_query": query_T,
                "w_interp": interp,
                "method": "linear_in_total_variance_at_fixed_k",
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def tenor_bucket(days: float) -> str:
    if days <= 30:
        return "<=30d"
    if days <= 90:
        return "31-90d"
    if days <= 180:
        return "91-180d"
    if days <= 365:
        return "181-365d"
    return ">365d"


def strike_spacing_summary(data: pd.DataFrame) -> dict:
    liquid = data[data.liquid]
    abs_diffs = []
    rel_diffs = []
    for (_, _), group in liquid.groupby(["date", "expiry"], sort=False):
        strikes = np.sort(group.K.unique())
        if len(strikes) < 2:
            continue
        diffs = np.diff(strikes)
        abs_diffs.extend(diffs.tolist())
        rel_diffs.extend((diffs / np.maximum(strikes[:-1], 1e-12)).tolist())
    abs_diffs = np.asarray(abs_diffs, dtype=float)
    rel_diffs = np.asarray(rel_diffs, dtype=float)
    return {
        "abs_median": float(np.median(abs_diffs)) if abs_diffs.size else float("nan"),
        "abs_p05": float(np.quantile(abs_diffs, 0.05)) if abs_diffs.size else float("nan"),
        "abs_p95": float(np.quantile(abs_diffs, 0.95)) if abs_diffs.size else float("nan"),
        "rel_median": float(np.median(rel_diffs)) if rel_diffs.size else float("nan"),
        "rel_p05": float(np.quantile(rel_diffs, 0.05)) if rel_diffs.size else float("nan"),
        "rel_p95": float(np.quantile(rel_diffs, 0.95)) if rel_diffs.size else float("nan"),
        "pairs": int(abs_diffs.size),
    }


def load_cffex_futures(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    required = {"合约代码", "今收盘", "今结算"}
    rows = []
    errors = []
    files = sorted(raw_dir.glob("*.csv"))
    for path in files:
        date_text = path.stem.split("_")[0]
        try:
            file_date = pd.Timestamp(date_text)
        except (TypeError, ValueError):
            errors.append({"file": path.name, "error": "unparseable_date"})
            continue
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append({"file": path.name, "error": type(exc).__name__})
            continue
        if not required.issubset(frame.columns):
            errors.append({"file": path.name, "error": "missing_columns"})
            continue
        for code_raw, close_raw, settle_raw in frame[["合约代码", "今收盘", "今结算"]].itertuples(index=False, name=None):
            code = str(code_raw).strip()
            if len(code) != 6 or not code.startswith("IF") or not code[2:].isdigit():
                continue
            yy = int(code[2:4])
            mm = int(code[4:6])
            if not 1 <= mm <= 12:
                continue
            close = pd.to_numeric(close_raw, errors="coerce")
            settle = pd.to_numeric(settle_raw, errors="coerce")
            rows.append({
                "date": file_date,
                "contract": code,
                "fut_expiry_month": f"{2000 + yy:04d}-{mm:02d}",
                "if_close": float(close) if pd.notna(close) else np.nan,
                "if_settle": float(settle) if pd.notna(settle) else np.nan,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {"files": len(files), "parsed_files": 0, "if_rows": 0, "errors": errors}
    frame = frame.drop_duplicates(["date", "contract"]).sort_values(["date", "contract"]).reset_index(drop=True)
    parsed_files = frame.date.nunique()
    stats = {
        "files": int(len(files)),
        "parsed_files": int(parsed_files),
        "if_rows": int(len(frame)),
        "errors": errors,
    }
    return frame, stats


def matched_q_stats(group: pd.DataFrame, basis_col: str) -> dict:
    matched = group[np.isfinite(group[basis_col].to_numpy(dtype=float))]
    if matched.empty:
        return {
            "n": 0,
            "implied_q_mean": float("nan"),
            "implied_q_median": float("nan"),
            "if_basis_mean": float("nan"),
            "if_basis_median": float("nan"),
            "q_minus_if_basis_mean": float("nan"),
            "q_minus_if_basis_median": float("nan"),
        }
    q = matched.q.to_numpy(dtype=float)
    basis = matched[basis_col].to_numpy(dtype=float)
    diff = q - basis
    return {
        "n": int(len(matched)),
        "implied_q_mean": float(np.mean(q)),
        "implied_q_median": float(np.median(q)),
        "if_basis_mean": float(np.mean(basis)),
        "if_basis_median": float(np.median(basis)),
        "q_minus_if_basis_mean": float(np.mean(diff)),
        "q_minus_if_basis_median": float(np.median(diff)),
    }


def forward_check(forwards: pd.DataFrame, futures: pd.DataFrame, source_stats: dict) -> tuple[pd.DataFrame, dict]:
    obs = forwards[["date", "expiry", "T", "days", "r", "DF", "S0", "F", "q"]].copy()
    obs["month"] = obs.date.dt.strftime("%Y-%m")
    obs["expiry_month"] = obs.expiry.dt.strftime("%Y-%m")
    obs["tenor_bucket"] = obs.days.map(tenor_bucket)
    if futures.empty:
        obs["if_close"] = np.nan
        obs["if_settle"] = np.nan
        obs["contract"] = None
    else:
        obs = obs.merge(
            futures[["date", "fut_expiry_month", "contract", "if_close", "if_settle"]],
            left_on=["date", "expiry_month"],
            right_on=["date", "fut_expiry_month"],
            how="left",
        ).drop(columns=["fut_expiry_month"])
    obs["if_rel_settle"] = (obs.F / obs.if_settle - 1.0).abs()
    obs["if_rel_close"] = (obs.F / obs.if_close - 1.0).abs()
    obs["if_basis_settle"] = np.where(
        (obs.if_settle > 0) & np.isfinite(obs.if_settle),
        obs.r - np.log(obs.if_settle / obs.S0) / obs["T"],
        np.nan,
    )
    obs["if_basis_close"] = np.where(
        (obs.if_close > 0) & np.isfinite(obs.if_close),
        obs.r - np.log(obs.if_close / obs.S0) / obs["T"],
        np.nan,
    )
    obs["q_minus_if_basis_settle"] = obs.q - obs.if_basis_settle
    obs["q_minus_if_basis_close"] = obs.q - obs.if_basis_close

    def dist(values: pd.Series) -> dict:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
            "max": float(np.max(arr)) if arr.size else float("nan"),
        }

    by_month = []
    for month, group in obs.groupby("month", sort=True):
        settle_stats = matched_q_stats(group, "if_basis_settle")
        close_stats = matched_q_stats(group, "if_basis_close")
        by_month.append({
            "month": month,
            "forwards": int(len(group)),
            "matched_settle": settle_stats["n"],
            "matched_close": close_stats["n"],
            "implied_q_median": settle_stats["implied_q_median"],
            "implied_q_matched_median": settle_stats["implied_q_median"],
            "implied_q_all_median": float(group.q.median()),
            "if_basis_settle_median": settle_stats["if_basis_median"],
            "if_basis_close_median": close_stats["if_basis_median"],
            "q_minus_if_basis_settle_mean": settle_stats["q_minus_if_basis_mean"],
            "q_minus_if_basis_settle_median": settle_stats["q_minus_if_basis_median"],
            "q_minus_if_basis_close_mean": close_stats["q_minus_if_basis_mean"],
            "q_minus_if_basis_close_median": close_stats["q_minus_if_basis_median"],
            "if_dev_settle": dist(group.if_rel_settle),
            "if_dev_close": dist(group.if_rel_close),
        })
    by_month_bucket = []
    for (month, bucket), group in obs.groupby(["month", "tenor_bucket"], sort=True):
        settle_stats = matched_q_stats(group, "if_basis_settle")
        close_stats = matched_q_stats(group, "if_basis_close")
        by_month_bucket.append({
            "month": month,
            "tenor_bucket": bucket,
            "forwards": int(len(group)),
            "matched_settle": settle_stats["n"],
            "matched_close": close_stats["n"],
            "implied_q_median": settle_stats["implied_q_median"],
            "implied_q_matched_median": settle_stats["implied_q_median"],
            "implied_q_all_median": float(group.q.median()),
            "if_basis_settle_median": settle_stats["if_basis_median"],
            "if_basis_close_median": close_stats["if_basis_median"],
            "q_minus_if_basis_settle_mean": settle_stats["q_minus_if_basis_mean"],
            "q_minus_if_basis_settle_median": settle_stats["q_minus_if_basis_median"],
            "q_minus_if_basis_close_mean": close_stats["q_minus_if_basis_mean"],
            "q_minus_if_basis_close_median": close_stats["q_minus_if_basis_median"],
            "if_dev_settle_median": float(group.if_rel_settle.median()) if group.if_rel_settle.notna().any() else float("nan"),
            "if_dev_close_median": float(group.if_rel_close.median()) if group.if_rel_close.notna().any() else float("nan"),
        })
    by_bucket = []
    for bucket, group in obs.groupby("tenor_bucket", sort=True):
        settle_stats = matched_q_stats(group, "if_basis_settle")
        close_stats = matched_q_stats(group, "if_basis_close")
        by_bucket.append({
            "tenor_bucket": bucket,
            "forwards": int(len(group)),
            "matched_settle": settle_stats["n"],
            "matched_close": close_stats["n"],
            "implied_q_median": settle_stats["implied_q_median"],
            "implied_q_matched_median": settle_stats["implied_q_median"],
            "implied_q_all_median": float(group.q.median()),
            "if_basis_settle_median": settle_stats["if_basis_median"],
            "if_basis_close_median": close_stats["if_basis_median"],
            "q_minus_if_basis_settle_mean": settle_stats["q_minus_if_basis_mean"],
            "q_minus_if_basis_settle_median": settle_stats["q_minus_if_basis_median"],
            "q_minus_if_basis_close_mean": close_stats["q_minus_if_basis_mean"],
            "q_minus_if_basis_close_median": close_stats["q_minus_if_basis_median"],
        })
    overall_settle = matched_q_stats(obs, "if_basis_settle")
    overall_close = matched_q_stats(obs, "if_basis_close")
    summary = {
        "source": source_stats,
        "forwards": int(len(obs)),
        "matched_settle": overall_settle["n"],
        "matched_close": overall_close["n"],
        "unmatched": int(len(obs) - overall_settle["n"]),
        "if_dev_settle": dist(obs.if_rel_settle),
        "if_dev_close": dist(obs.if_rel_close),
        "implied_q_median": overall_settle["implied_q_median"],
        "implied_q_matched_median": overall_settle["implied_q_median"],
        "implied_q_all_median": float(obs.q.median()),
        "if_basis_settle_median": overall_settle["if_basis_median"],
        "if_basis_close_median": overall_close["if_basis_median"],
        "q_minus_if_basis_settle_mean": overall_settle["q_minus_if_basis_mean"],
        "q_minus_if_basis_settle_median": overall_settle["q_minus_if_basis_median"],
        "q_minus_if_basis_close_mean": overall_close["q_minus_if_basis_mean"],
        "q_minus_if_basis_close_median": overall_close["q_minus_if_basis_median"],
        "by_month": by_month,
        "by_month_bucket": by_month_bucket,
        "by_bucket": by_bucket,
        "note": "IF close/settle are the CFFEX fields 今收盘 and 今结算; matched by the IO expiry year-month. Basis is annualized r - ln(IF/S0)/T. implied_q_median/q_minus_if_basis_* use exactly the matched pairs; implied_q_all_median is shown for contrast.",
    }
    return obs, summary


def svi_bound_comparison(
    current_params: pd.DataFrame,
    legacy_path: Path | None,
    eta_min: float,
    eta_max: float,
) -> tuple[pd.DataFrame, dict]:
    current = current_params[[
        "date", "expiry", "T", "eta_raw", "rmse_raw", "rmse_repaired"
    ]].copy()
    if legacy_path is None or not legacy_path.exists():
        return pd.DataFrame(), {
            "available": False,
            "reason": "no_legacy_svi_params",
            "eta_min": float(eta_min),
            "eta_max": float(eta_max),
        }
    legacy = pd.read_csv(legacy_path)
    needed = {"date", "expiry", "eta_raw", "rmse_raw", "rmse_repaired"}
    if not needed.issubset(legacy.columns):
        return pd.DataFrame(), {
            "available": False,
            "reason": "legacy_svi_params_missing_columns",
            "eta_min": float(eta_min),
            "eta_max": float(eta_max),
        }
    legacy = legacy[["date", "expiry", "eta_raw", "rmse_raw", "rmse_repaired"]].rename(columns={
        "eta_raw": "eta_legacy",
        "rmse_raw": "rmse_raw_legacy",
        "rmse_repaired": "rmse_repaired_legacy",
    })
    current["date"] = current.date.astype(str)
    current["expiry"] = current.expiry.astype(str)
    legacy["date"] = legacy.date.astype(str)
    legacy["expiry"] = legacy.expiry.astype(str)
    merged = current.rename(columns={
        "eta_raw": "eta_new",
        "rmse_raw": "rmse_raw_new",
        "rmse_repaired": "rmse_repaired_new",
    }).merge(legacy, on=["date", "expiry"], how="inner")
    if merged.empty:
        return merged, {
            "available": False,
            "reason": "no_common_fits",
            "eta_min": float(eta_min),
            "eta_max": float(eta_max),
        }
    merged["days"] = merged["T"] * 365.0
    merged["tenor_bucket"] = merged.days.map(tenor_bucket)
    merged["legacy_at_lower"] = merged.eta_legacy <= LEGACY_ETA_MIN + 1e-6
    merged["legacy_at_upper"] = merged.eta_legacy >= LEGACY_ETA_MAX - 1e-6
    merged["legacy_near_lower_1e_4"] = merged.eta_legacy < 1e-4
    merged["legacy_upper_gt_1_99"] = merged.eta_legacy > 1.99
    merged["new_at_lower"] = merged.eta_new <= eta_min + 1e-6
    merged["new_at_upper"] = merged.eta_new >= eta_max - 1e-6
    merged["rmse_raw_delta"] = merged.rmse_raw_new - merged.rmse_raw_legacy
    merged["rmse_repaired_delta"] = merged.rmse_repaired_new - merged.rmse_repaired_legacy

    def bucket_rows(group: pd.DataFrame) -> dict:
        return {
            "n": int(len(group)),
            "legacy_at_lower": int(group.legacy_at_lower.sum()),
            "legacy_at_upper": int(group.legacy_at_upper.sum()),
            "new_at_lower": int(group.new_at_lower.sum()),
            "new_at_upper": int(group.new_at_upper.sum()),
            "legacy_rmse_raw_median": float(group.rmse_raw_legacy.median()),
            "new_rmse_raw_median": float(group.rmse_raw_new.median()),
            "delta_rmse_raw_median": float(group.rmse_raw_delta.median()),
            "legacy_rmse_repaired_median": float(group.rmse_repaired_legacy.median()),
            "new_rmse_repaired_median": float(group.rmse_repaired_new.median()),
            "delta_rmse_repaired_median": float(group.rmse_repaired_delta.median()),
        }

    by_bucket = []
    for bucket in ("<=30d", "31-90d", "91-180d", "181-365d", ">365d"):
        group = merged[merged.tenor_bucket == bucket]
        if group.empty:
            continue
        by_bucket.append({"tenor_bucket": bucket, **bucket_rows(group)})
    overall = {
        "available": True,
        "eta_min": float(eta_min),
        "eta_max": float(eta_max),
        "legacy_eta_min": float(LEGACY_ETA_MIN),
        "legacy_eta_max": float(LEGACY_ETA_MAX),
        "common_fits": int(len(merged)),
        **{key: value for key, value in bucket_rows(merged).items() if key != "n"},
        "rmse_raw_new_p95": float(merged.rmse_raw_new.quantile(0.95)),
        "rmse_raw_legacy_p95": float(merged.rmse_raw_legacy.quantile(0.95)),
        "rmse_repaired_new_p95": float(merged.rmse_repaired_new.quantile(0.95)),
        "rmse_repaired_legacy_p95": float(merged.rmse_repaired_legacy.quantile(0.95)),
        "improved_raw": int((merged.rmse_raw_delta < 0).sum()),
        "worsened_raw": int((merged.rmse_raw_delta > 0).sum()),
        "unchanged_raw": int((merged.rmse_raw_delta == 0).sum()),
        "by_tenor_bucket": by_bucket,
        "legacy_path": rel_path(legacy_path),
        "note": "Legacy bounds are [1e-6, 2.0]; new bounds are [eta_min, eta_max]. RMSE is in total variance w on each fit's liquid OTM strike set.",
    }
    return merged, overall


def repricing_date_stats(frame: pd.DataFrame, date_str: str) -> dict:
    subset = frame[frame.date == date_str]
    if subset.empty:
        return {
            "fitmax_median": np.nan,
            "fitmax_p95": np.nan,
            "strikemedian_median": np.nan,
            "strikemedian_p95": np.nan,
        }
    valid = subset[np.isfinite(subset.max_rel_error)]
    if valid.empty:
        return {
            "fitmax_median": np.nan,
            "fitmax_p95": np.nan,
            "strikemedian_median": np.nan,
            "strikemedian_p95": np.nan,
        }
    return {
        "fitmax_median": float(valid.max_rel_error.median()),
        "fitmax_p95": float(valid.max_rel_error.quantile(0.95)),
        "strikemedian_median": float(valid.median_rel_error.median()),
        "strikemedian_p95": float(valid.median_rel_error.quantile(0.95)),
    }


def calibration_by_date(
    fits: dict,
    data: pd.DataFrame,
    butterfly_raw: pd.DataFrame,
    butterfly_rep: pd.DataFrame,
    calendar_raw: pd.DataFrame,
    calendar_rep: pd.DataFrame,
    repricing_raw: dict[str, pd.DataFrame],
    repricing_rep: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for date, group in data.groupby("date", sort=True):
        date_str = date.strftime("%Y-%m-%d")
        date_fits = [fit for fit in fits.values() if fit["date"] == date_str]
        bf_raw = butterfly_raw[(butterfly_raw.date == date_str) & butterfly_raw.violates_correct]
        bf_rep = butterfly_rep[(butterfly_rep.date == date_str) & butterfly_rep.violates_correct]
        cal_raw = calendar_raw[(calendar_raw.date == date_str) & calendar_raw.violates_calendar]
        cal_rep = calendar_rep[(calendar_rep.date == date_str) & calendar_rep.violates_calendar]
        qs = [fit["q"] for fit in date_fits]
        row = {
            "date": date_str,
            "month": date.strftime("%Y-%m"),
            "n_expiries": int(group.groupby("expiry").ngroups),
            "n_fits": int(len(date_fits)),
            "n_skipped": int(group.groupby("expiry").ngroups - len(date_fits)),
            "quotes_total": int(len(group)),
            "quotes_liquid": int(group.liquid.sum()),
            "median_q": float(np.median(qs)) if qs else np.nan,
            "butterfly_raw": int(len(bf_raw)),
            "butterfly_raw_strict": int(((butterfly_raw.date == date_str) & butterfly_raw.violates_correct_strict).sum()),
            "butterfly_repaired": int(len(bf_rep)),
            "butterfly_repaired_strict": int(((butterfly_rep.date == date_str) & butterfly_rep.violates_correct_strict).sum()),
            "butterfly_printed_raw": int(((butterfly_raw.date == date_str) & butterfly_raw.violates_printed).sum()),
            "butterfly_printed_raw_strict": int(((butterfly_raw.date == date_str) & butterfly_raw.violates_printed_strict).sum()),
            "butterfly_printed_repaired": int(((butterfly_rep.date == date_str) & butterfly_rep.violates_printed).sum()),
            "butterfly_printed_repaired_strict": int(((butterfly_rep.date == date_str) & butterfly_rep.violates_printed_strict).sum()),
            "calendar_raw": int(len(cal_raw)),
            "calendar_raw_strict": int(((calendar_raw.date == date_str) & calendar_raw.violates_calendar_strict).sum()),
            "calendar_repaired": int(len(cal_rep)),
            "calendar_repaired_strict": int(((calendar_rep.date == date_str) & calendar_rep.violates_calendar_strict).sum()),
        }
        for basis in ("otm", "calls"):
            raw_stats = repricing_date_stats(repricing_raw[basis], date_str)
            rep_stats = repricing_date_stats(repricing_rep[basis], date_str)
            for key, value in raw_stats.items():
                row[f"repricing_{basis}_{key}_raw"] = value
            for key, value in rep_stats.items():
                row[f"repricing_{basis}_{key}_repaired"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_fit_slices(fits: dict, sample_dates: list[str], out_path: Path) -> None:
    dates = [date for date in sample_dates if any(fit["date"] == date for fit in fits.values())]
    if not dates:
        dates = sorted({fit["date"] for fit in fits.values()})[:6]
    max_expiries = max(len({fit["expiry"] for fit in fits.values() if fit["date"] == date}) for date in dates)
    fig = make_subplots(
        rows=len(dates),
        cols=max_expiries,
        subplot_titles=[f"{date} exp {i + 1}" for date in dates for i in range(max_expiries)],
        vertical_spacing=0.035,
        horizontal_spacing=0.03,
    )
    for row_idx, date in enumerate(dates, start=1):
        items = sorted([fit for fit in fits.values() if fit["date"] == date], key=lambda item: item["T"])
        for col_idx, fit in enumerate(items, start=1):
            line_k = np.round(np.arange(-0.20, 0.20 + 0.005, 0.01), 10)
            raw_w = svi_w(line_k, fit["raw"]["params"])
            rep_w = svi_w(line_k, fit["repaired"]["params"])
            with np.errstate(invalid="ignore", divide="ignore"):
                raw_iv = np.sqrt(raw_w / fit["T"])
                rep_iv = np.sqrt(rep_w / fit["T"])
            fig.add_trace(
                go.Scatter(x=fit["k"], y=fit["iv"], mode="markers", name="OTM quotes", marker=dict(size=4, color="#1f77b4"), showlegend=(row_idx == 1 and col_idx == 1)),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(x=line_k, y=raw_iv, mode="lines", name="raw SVI", line=dict(color="#ff7f0e", width=1.2), showlegend=(row_idx == 1 and col_idx == 1)),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(x=line_k, y=rep_iv, mode="lines", name="repaired SVI", line=dict(color="#2ca02c", width=1.2, dash="dot"), showlegend=(row_idx == 1 and col_idx == 1)),
                row=row_idx,
                col=col_idx,
            )
            fig.update_xaxes(title_text="k", row=row_idx, col=col_idx)
            fig.update_yaxes(title_text="IV", row=row_idx, col=col_idx)
    fig.update_layout(height=max(420, 300 * len(dates)), width=1800, title="Fitted SVI slices on sample dates")
    fig.write_html(out_path, include_plotlyjs=True)


def build_surface(fits: dict, out_path: Path) -> None:
    by_date: dict[str, list[dict]] = {}
    for fit in fits.values():
        by_date.setdefault(fit["date"], []).append(fit)
    dates = sorted(by_date)
    frames = []
    for date in dates:
        items = sorted(by_date[date], key=lambda item: item["T"])
        Ts = np.array([item["T"] for item in items])
        z = []
        for item in items:
            w = svi_w(SURFACE_K, item["repaired"]["params"])
            with np.errstate(invalid="ignore", divide="ignore"):
                z.append(np.sqrt(w / item["T"]))
        frames.append(go.Frame(
            name=date,
            data=[go.Surface(
                x=SURFACE_K,
                y=Ts,
                z=np.vstack(z),
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                name=date,
            )],
        ))
    first_items = sorted(by_date[dates[0]], key=lambda item: item["T"])
    first_T = np.array([item["T"] for item in first_items])
    first_z = []
    for item in first_items:
        w = svi_w(SURFACE_K, item["repaired"]["params"])
        with np.errstate(invalid="ignore", divide="ignore"):
            first_z.append(np.sqrt(w / item["T"]))
    fig = go.Figure(
        data=[go.Surface(x=SURFACE_K, y=first_T, z=np.vstack(first_z), colorscale="Viridis", colorbar=dict(title="IV"))],
        frames=frames,
    )
    steps = [dict(method="animate", args=[[date], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=date) for date in dates]
    fig.update_layout(
        title="SVI implied-volatility surface (repaired; use slider to change date)",
        scene=dict(xaxis_title="k = ln(K/F)", yaxis_title="T (years)", zaxis_title="IV"),
        sliders=[dict(active=0, currentvalue=dict(prefix="Date: "), steps=steps)],
        height=850,
    )
    fig.write_html(out_path, include_plotlyjs=True)


def distribution(values: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": float("nan"), "p95": float("nan"), "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def monthly_sample_table(calibration: pd.DataFrame, sample_months: list[str]) -> list[dict]:
    rows = []
    for month in sample_months:
        subset = calibration[calibration.month == month]
        if subset.empty:
            continue
        rows.append({
            "month": month,
            "dates": int(len(subset)),
            "fits": int(subset.n_fits.sum()),
            "skipped": int(subset.n_skipped.sum()),
            "quotes_liquid": int(subset.quotes_liquid.sum()),
            "median_q": float(subset.median_q.median()) if subset.median_q.notna().any() else float("nan"),
            "butterfly_raw": int(subset.butterfly_raw.sum()),
            "butterfly_raw_strict": int(subset.butterfly_raw_strict.sum()),
            "butterfly_repaired": int(subset.butterfly_repaired.sum()),
            "butterfly_repaired_strict": int(subset.butterfly_repaired_strict.sum()),
            "calendar_raw": int(subset.calendar_raw.sum()),
            "calendar_raw_strict": int(subset.calendar_raw_strict.sum()),
            "calendar_repaired": int(subset.calendar_repaired.sum()),
            "calendar_repaired_strict": int(subset.calendar_repaired_strict.sum()),
            "repricing_otm_fitmax_median_raw": float(subset.repricing_otm_fitmax_median_raw.median()) if subset.repricing_otm_fitmax_median_raw.notna().any() else float("nan"),
            "repricing_otm_fitmax_p95_raw": float(subset.repricing_otm_fitmax_p95_raw.median()) if subset.repricing_otm_fitmax_p95_raw.notna().any() else float("nan"),
            "repricing_otm_strikemedian_median_raw": float(subset.repricing_otm_strikemedian_median_raw.median()) if subset.repricing_otm_strikemedian_median_raw.notna().any() else float("nan"),
            "repricing_otm_fitmax_median_repaired": float(subset.repricing_otm_fitmax_median_repaired.median()) if subset.repricing_otm_fitmax_median_repaired.notna().any() else float("nan"),
            "repricing_otm_strikemedian_median_repaired": float(subset.repricing_otm_strikemedian_median_repaired.median()) if subset.repricing_otm_strikemedian_median_repaired.notna().any() else float("nan"),
        })
    return rows


def print_summary(console: Console, params: dict, metrics: dict, sample_months: list[dict], sample_dates: list[dict]) -> None:
    console.print(Panel.fit(
        f"Q1A SVI | dataset=io | price_col={params['price_col']} | seed={params['seed']} | "
        f"fits={metrics['fits']} | skipped={metrics['skipped']}",
        title="Q1 Option A",
    ))
    table = Table(title="Quote filtering")
    table.add_column("Item")
    table.add_column("Value", justify="right")
    for key in ("quotes_total", "quotes_liquid"):
        table.add_row(key, f"{metrics[key]:,}")
    for key, value in metrics["filter_removals"]["independent"].items():
        table.add_row(f"independent {key}", f"{value:,}")
    for key, value in metrics["filter_removals"]["sequential"].items():
        table.add_row(f"sequential {key}", f"{value:,}")
    console.print(table)

    table = Table(title="Implied-vol inversion")
    table.add_column("Status")
    table.add_column("Calls", justify="right")
    table.add_column("Puts", justify="right")
    statuses = sorted(set(metrics["iv"]["status_counts"]["C"]) | set(metrics["iv"]["status_counts"]["P"]))
    for status in statuses:
        table.add_row(status, f"{metrics['iv']['status_counts']['C'].get(status, 0):,}", f"{metrics['iv']['status_counts']['P'].get(status, 0):,}")
    console.print(table)

    table = Table(title="No-arbitrage audits")
    table.add_column("Audit")
    table.add_column("Raw", justify="right")
    table.add_column("Repaired", justify="right")
    table.add_row("butterfly correct strict", f"{metrics['butterfly']['raw']['violations_strict']:,}", f"{metrics['butterfly']['repaired']['violations_strict']:,}")
    table.add_row("butterfly correct tol 1e-8", f"{metrics['butterfly']['raw']['violations']:,}", f"{metrics['butterfly']['repaired']['violations']:,}")
    table.add_row("butterfly printed strict", f"{metrics['butterfly']['raw']['violations_printed_strict']:,}", f"{metrics['butterfly']['repaired']['violations_printed_strict']:,}")
    table.add_row("butterfly printed tol 1e-8", f"{metrics['butterfly']['raw']['violations_printed']:,}", f"{metrics['butterfly']['repaired']['violations_printed']:,}")
    table.add_row("w <= 0", f"{metrics['butterfly']['raw']['w_nonpositive']:,}", f"{metrics['butterfly']['repaired']['w_nonpositive']:,}")
    table.add_row("calendar strict", f"{metrics['calendar']['raw']['violations_strict']:,}", f"{metrics['calendar']['repaired']['violations_strict']:,}")
    table.add_row("calendar tol 1e-8", f"{metrics['calendar']['raw']['violations']:,}", f"{metrics['calendar']['repaired']['violations']:,}")
    console.print(table)

    for basis in ("otm", "calls"):
        table = Table(title=f"Repricing {basis} |model - market| / market across fits")
        table.add_column("Metric")
        table.add_column("Raw", justify="right")
        table.add_column("Repaired", justify="right")
        for key, label in (
            ("per_fit_max_median", "median of per-fit max"),
            ("per_fit_max_p95", "p95 of per-fit max"),
            ("per_fit_median_median", "median of per-fit median"),
            ("per_fit_median_p95", "p95 of per-fit median"),
            ("worst_fit_max", "worst per-fit max"),
        ):
            table.add_row(
                label,
                f"{metrics['repricing'][basis]['raw'][key]:.6f}",
                f"{metrics['repricing'][basis]['repaired'][key]:.6f}",
            )
        console.print(table)

    if sample_months:
        table = Table(title="Sample months")
        for col in ("month", "dates", "fits", "skipped", "quotes", "median q", "bf raw", "bf rep", "cal raw", "cal rep", "otm fitmax raw", "otm strikemed raw", "otm fitmax rep"):
            table.add_column(col, justify="right" if col != "month" else "left")
        for row in sample_months:
            table.add_row(
                row["month"],
                f"{row['dates']}",
                f"{row['fits']}",
                f"{row['skipped']}",
                f"{row['quotes_liquid']:,}",
                f"{row['median_q']:.4f}",
                f"{row['butterfly_raw']}",
                f"{row['butterfly_repaired']}",
                f"{row['calendar_raw']}",
                f"{row['calendar_repaired']}",
                f"{row['repricing_otm_fitmax_median_raw']:.4f}",
                f"{row['repricing_otm_strikemedian_median_raw']:.4f}",
                f"{row['repricing_otm_fitmax_median_repaired']:.4f}",
            )
        console.print(table)

    if sample_dates:
        table = Table(title="Sample dates")
        for col in ("date", "fits", "median q", "bf raw", "bf rep", "cal raw", "cal rep", "otm fitmax raw", "otm fitmax rep"):
            table.add_column(col, justify="right" if col != "date" else "left")
        for row in sample_dates:
            table.add_row(
                row["date"], f"{row['n_fits']}", f"{row['median_q']:.4f}", f"{row['butterfly_raw']}",
                f"{row['butterfly_repaired']}", f"{row['calendar_raw']}", f"{row['calendar_repaired']}",
                f"{row['repricing_otm_fitmax_median_raw']:.4f}", f"{row['repricing_otm_fitmax_median_repaired']:.4f}",
            )
        console.print(table)


def run(args: argparse.Namespace) -> dict:
    console = Console(width=220)
    start = time.perf_counter()
    np.random.seed(args.seed)
    if not (0.0 < args.eta_min < args.eta_max):
        raise ValueError("require 0 < eta_min < eta_max")
    bounds = svi_bounds(args.eta_min, args.eta_max)
    root = ROOT.resolve()
    out_resolved = args.out_dir.resolve()
    if out_resolved != root and root not in out_resolved.parents:
        raise ValueError(f"output directory must stay inside {root}: {out_resolved}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading and verifying data[/bold]")
    options, shibor, metadata = load_data(args.options, args.shibor, args.metadata)
    input_hashes = {
        "options_sha256": sha256_file(args.options),
        "shibor_sha256": sha256_file(args.shibor),
    }
    console.print(f"  options rows={len(options):,} dates={options.date.nunique()} expiries={options.expiry.nunique()}")
    console.print(f"  shibor rows={len(shibor):,} range={shibor.date.min().date()}..{shibor.date.max().date()}")

    console.print("[bold]Estimating forwards from put-call parity[/bold]")
    forwards, parity = estimate_forwards(options, shibor, args.price_col)
    data, filter_counts = apply_filters(options, forwards, args)
    console.print(f"  date-expiry pairs={len(forwards)} | quotes liquid={filter_counts['quotes_liquid']:,}/{filter_counts['quotes_total']:,}")

    console.print("[bold]Inverting implied volatilities[/bold]")
    iv, divergence, iv_stats = compute_implied_vols(data[data.liquid])
    console.print(f"  IV ok={iv_stats['ok_quotes']:,}/{iv_stats['attempted_quotes']:,} | C-P gap median={iv_stats['gap_median']:.3g} p95|gap|={iv_stats['gap_p95_abs']:.3g}")

    console.print("[bold]Fitting raw SVI and repairing violations[/bold]")
    fit_start = time.perf_counter()
    fits, svi_params, fit_history = make_fits(forwards, data, iv, args.min_strikes, bounds)
    fit_seconds = time.perf_counter() - fit_start
    console.print(f"  fits={len(fits)} skipped={len(fit_history['skipped'])} {fit_history['skip_reasons']} raw repair candidates={sum(1 for x in fit_history['repair'] if x['needed_repair'])}")

    butterfly_raw = butterfly_rows(fits, "raw")
    butterfly_rep = butterfly_rows(fits, "repaired")
    calendar_raw = calendar_rows(fits, "raw")
    calendar_rep = calendar_rows(fits, "repaired")
    repricing_raw = {}
    repricing_raw_summary = {}
    repricing_rep = {}
    repricing_rep_summary = {}
    for basis in ("otm", "calls"):
        repricing_raw[basis], repricing_raw_summary[basis] = repricing_for_stage(
            fits, data[data.liquid], "raw", basis
        )
        repricing_rep[basis], repricing_rep_summary[basis] = repricing_for_stage(
            fits, data[data.liquid], "repaired", basis
        )

    bf_raw_summary = {
        "violations_strict": int(butterfly_raw.violates_correct_strict.sum()),
        "violations": int(butterfly_raw.violates_correct.sum()),
        "violations_printed_strict": int(butterfly_raw.violates_printed_strict.sum()),
        "violations_printed": int(butterfly_raw.violates_printed.sum()),
        "w_nonpositive": int(butterfly_raw.w_nonpositive.sum()),
        "fits_with_violation_strict": int(butterfly_raw.groupby(["date", "expiry"]).violates_correct_strict.any().sum()),
        "fits_with_violation": int(butterfly_raw.groupby(["date", "expiry"]).violates_correct.any().sum()),
        "checks": int(len(butterfly_raw)),
        "min_g": float(butterfly_raw.g.min()) if np.isfinite(butterfly_raw.g).any() else float("nan"),
    }
    bf_rep_summary = {
        "violations_strict": int(butterfly_rep.violates_correct_strict.sum()),
        "violations": int(butterfly_rep.violates_correct.sum()),
        "violations_printed_strict": int(butterfly_rep.violates_printed_strict.sum()),
        "violations_printed": int(butterfly_rep.violates_printed.sum()),
        "w_nonpositive": int(butterfly_rep.w_nonpositive.sum()),
        "fits_with_violation_strict": int(butterfly_rep.groupby(["date", "expiry"]).violates_correct_strict.any().sum()),
        "fits_with_violation": int(butterfly_rep.groupby(["date", "expiry"]).violates_correct.any().sum()),
        "checks": int(len(butterfly_rep)),
        "min_g": float(butterfly_rep.g.min()) if np.isfinite(butterfly_rep.g).any() else float("nan"),
    }
    calendar_raw_summary = calendar_summary(calendar_raw)
    calendar_rep_summary = calendar_summary(calendar_rep)

    console.print("[bold]Checking forwards against CFFEX IF futures[/bold]")
    futures, futures_stats = load_cffex_futures(args.raw_cffex)
    forward_rows, forward_summary = forward_check(forwards, futures, futures_stats)
    console.print(
        f"  IF files={futures_stats['files']} rows={futures_stats['if_rows']:,} "
        f"matched settle={forward_summary['matched_settle']}/{forward_summary['forwards']}"
    )
    bound_rows, bound_summary = svi_bound_comparison(
        svi_params, args.legacy_svi_params, args.eta_min, args.eta_max
    )
    spacing = strike_spacing_summary(data)
    console.print(
        f"  eta bounds=[{args.eta_min}, {args.eta_max}] | "
        f"strike spacing abs median={spacing['abs_median']:.1f} rel median={spacing['rel_median']:.4f}"
    )

    surface = surface_frames(fits, "repaired")[["date", "expiry", "T", "F", "k", "w", "iv"]].merge(
        surface_frames(fits, "raw")[["date", "expiry", "k", "w", "iv"]].rename(columns={"w": "w_raw", "iv": "iv_raw"}),
        on=["date", "expiry", "k"],
        how="left",
    )
    term_structure = term_structure_rows(fits)
    calibration = calibration_by_date(fits, data, butterfly_raw, butterfly_rep, calendar_raw, calendar_rep, repricing_raw, repricing_rep)

    parity_stats = {
        "pairs": int(len(parity)),
        "abs_resid_S0_median": float(parity.resid_S0.abs().median()) if len(parity) else float("nan"),
        "abs_resid_S0_p95": float(parity.resid_S0.abs().quantile(0.95)) if len(parity) else float("nan"),
        "abs_resid_F_median": float(parity.resid_F.abs().median()) if len(parity) else float("nan"),
        "abs_resid_F_p95": float(parity.resid_F.abs().quantile(0.95)) if len(parity) else float("nan"),
        "q_median": float(forwards.q.median()) if len(forwards) else float("nan"),
        "q_p05": float(forwards.q.quantile(0.05)) if len(forwards) else float("nan"),
        "q_p95": float(forwards.q.quantile(0.95)) if len(forwards) else float("nan"),
        "in_band_fraction": float(parity.in_band.mean()) if len(parity) else float("nan"),
        "fallback_forwards": int(forwards.fallback_F.sum()) if len(forwards) else 0,
    }

    sample_dates = list(metadata.get("sample_dates", []))
    sample_months = list(dict.fromkeys(pd.to_datetime(sample_dates).strftime("%Y-%m"))) if sample_dates else []
    if not sample_months:
        sample_months = sorted(calibration.month.unique())
    sample_date_rows = []
    for date in sample_dates:
        subset = calibration[calibration.date == date]
        if len(subset):
            row = subset.iloc[0]
            sample_date_rows.append({key: (float(row[key]) if isinstance(row[key], (float, np.floating)) else row[key]) for key in (
                "date", "n_fits", "n_skipped", "median_q", "butterfly_raw", "butterfly_repaired",
                "calendar_raw", "calendar_repaired", "repricing_otm_fitmax_median_raw",
                "repricing_otm_fitmax_median_repaired",
            )})
    monthly_rows = monthly_sample_table(calibration, sample_months)
    forward_months = {row["month"]: row for row in forward_summary["by_month"]}
    for row in monthly_rows:
        forward_month = forward_months.get(row["month"], {})
        row["if_basis_settle_median"] = forward_month.get("if_basis_settle_median", float("nan"))
        row["if_basis_close_median"] = forward_month.get("if_basis_close_median", float("nan"))
        row["if_dev_settle_median"] = forward_month.get("if_dev_settle", {}).get("median", float("nan"))
        row["if_dev_settle_p95"] = forward_month.get("if_dev_settle", {}).get("p95", float("nan"))

    metrics = {
        "quotes_total": filter_counts["quotes_total"],
        "quotes_liquid": filter_counts["quotes_liquid"],
        "filter_removals": {
            "independent": filter_counts["independent_removals"],
            "sequential": filter_counts["sequential_removals"],
            "note": "No bid-ask snapshot is available; volume >= min_volume stands in for spread/liquidity.",
        },
        "date_expiry_groups_total": filter_counts["date_expiry_groups_total"],
        "date_expiry_groups_liquid": filter_counts["date_expiry_groups_liquid"],
        "fits": int(len(fits)),
        "skipped": int(len(fit_history["skipped"])),
        "skipped_by_reason": fit_history["skip_reasons"],
        "skipped_detail": fit_history["skipped"],
        "otm_quotes_used": int(fit_history["otm_counts"]["otm_quotes_used"]),
        "parity": parity_stats,
        "iv": iv_stats,
        "repricing": {
            "primary_basis": "otm",
            "otm": {"raw": repricing_raw_summary["otm"], "repaired": repricing_rep_summary["otm"]},
            "calls": {"raw": repricing_raw_summary["calls"], "repaired": repricing_rep_summary["calls"]},
            "note": "Both bases use the common strikes with a liquid OTM quote and a liquid call. otm uses the traded OTM option (put below F, call at/above F); calls uses the traded call only. No synthetic call is constructed. Per fit: max and median relative repricing error; across fits: median and p95 of those statistics.",
        },
        "butterfly": {
            "raw": bf_raw_summary,
            "repaired": bf_rep_summary,
            "tolerance": BUTTERFLY_TOL,
            "grid": {"start": -0.14, "stop": 0.14, "step": 0.01, "points": int(len(GRID_K))},
            "note": "Primary test uses correct Gatheral g incl. w''/2; printed formula omits w''/2.",
        },
        "calendar": {
            "raw": calendar_raw_summary,
            "repaired": calendar_rep_summary,
            "tolerance_w": CALENDAR_TOL,
            "grid": {"start": -0.14, "stop": 0.14, "step": 0.01, "points": int(len(GRID_K))},
            "term_structure": "linear interpolation in total variance w at fixed k",
        },
        "forward_check": forward_summary,
        "svi_bounds": {
            **bound_summary,
            "eta_min": float(args.eta_min),
            "eta_max": float(args.eta_max),
            "strike_spacing": spacing,
            "starting_points_retained": len(SVI_STARTS),
            "start_reduction": False,
            "other_bounds": {
                "a": [-2.0, 2.0],
                "b": [0.0, 5.0],
                "rho": [-0.999, 0.999],
                "m": [-2.0, 2.0],
            },
        },
        "sample_months": monthly_rows,
        "sample_dates": sample_date_rows,
        "input_hashes": input_hashes,
    }
    params = {
        "approach": "Q1A_SVI",
        "dataset": "io",
        "seed": args.seed,
        "price_col": args.price_col,
        "min_volume": args.min_volume,
        "min_price": args.min_price,
        "kband": args.kband,
        "min_days": args.min_days,
        "min_strikes": args.min_strikes,
        "inputs": {
            "options": rel_path(args.options),
            "shibor": rel_path(args.shibor),
            "metadata": rel_path(args.metadata),
            "raw_cffex": rel_path(args.raw_cffex),
            "legacy_svi_params": rel_path(args.legacy_svi_params),
        },
        "price_note": "CFFEX fields: settle=今结算, close=今收盘; no settlement algorithm is inferred.",
        "rate_convention": "r(tau)=ln(1+L/100*tau*365/360)/tau from SHIBOR ACT/360 simple; linear in T, flat outside pillars",
        "forward_estimator": "volume-weighted mean of F_K=K+(C-P)/DF within |ln(F_K/median F_K)|<=0.05",
        "iv_bounds": [1e-4, 5.0],
        "svi_bounds": {
            "eta_min": float(args.eta_min),
            "eta_max": float(args.eta_max),
            "other": {"a": [-2.0, 2.0], "b": [0.0, 5.0], "rho": [-0.999, 0.999], "m": [-2.0, 2.0]},
            "justification": "eta_min=0.01 lies below the observed strike spacing; eta_max=1.0 covers wide shallow smiles without the legacy 2.0 extreme.",
            "starting_points_retained": len(SVI_STARTS),
            "start_reduction": False,
        },
        "svi_form": "a+b*(rho*(k-m)+sqrt((k-m)^2+eta^2)); b>=0, |rho|<1, eta>0",
        "svi_starting_points": [{"label": label, "b": b, "rho": rho, "m": m, "eta": eta} for label, b, rho, m, eta in SVI_STARTS],
        "svi_weights": "equal weights on OTM liquid quotes",
        "butterfly_grid": {"start": -0.14, "stop": 0.14, "step": 0.01},
        "surface_grid": {"start": -0.20, "stop": 0.20, "step": 0.02},
        "out_dir": rel_path(args.out_dir),
    }
    history = {
        **fit_history,
        "rates": {
            "pillars": PILLAR_DAYS,
            "convention": "simple ACT/360 SHIBOR converted to continuous zero rate on ACT/365 T",
        },
        "timing": {"fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - start},
    }
    metrics_payload = {
        "params": params,
        "metrics": metrics,
        "fit_seconds": float(fit_seconds),
        "history": history,
    }

    console.print("[bold]Writing outputs[/bold]")
    output = args.out_dir
    butterfly_raw.to_csv(output / "butterfly_audit_raw.csv", index=False)
    butterfly_rep.to_csv(output / "butterfly_audit_repaired.csv", index=False)
    pd.concat([butterfly_raw, butterfly_rep], ignore_index=True).to_csv(output / "butterfly_audit.csv", index=False)
    calendar_raw.to_csv(output / "calendar_audit_raw.csv", index=False)
    calendar_rep.to_csv(output / "calendar_audit_repaired.csv", index=False)
    pd.concat([calendar_raw, calendar_rep], ignore_index=True).to_csv(output / "calendar_audit.csv", index=False)
    for basis in ("otm", "calls"):
        repricing_raw[basis].to_csv(output / f"repricing_{basis}_raw.csv", index=False)
        repricing_rep[basis].to_csv(output / f"repricing_{basis}_repaired.csv", index=False)
    repricing_raw["otm"].to_csv(output / "repricing_raw.csv", index=False)
    repricing_rep["otm"].to_csv(output / "repricing.csv", index=False)
    repricing_raw["calls"].to_csv(output / "repricing_calls_raw.csv", index=False)
    repricing_rep["calls"].to_csv(output / "repricing_calls.csv", index=False)
    svi_params.to_csv(output / "svi_params.csv", index=False)
    calibration.to_csv(output / "calibration_by_date.csv", index=False)
    forward_rows.to_csv(output / "forward_check.csv", index=False)
    pd.DataFrame(forward_summary["by_month"]).to_csv(output / "if_check_by_month.csv", index=False)
    pd.DataFrame(forward_summary["by_month_bucket"]).to_csv(output / "if_basis_by_month_bucket.csv", index=False)
    pd.DataFrame(forward_summary["by_bucket"]).to_csv(output / "if_basis_by_tenor.csv", index=False)
    if len(bound_rows):
        bound_rows.to_csv(output / "svi_bound_comparison.csv", index=False)
        pd.DataFrame(bound_summary["by_tenor_bucket"]).to_csv(output / "svi_boundary_counts.csv", index=False)
    surface.to_csv(output / "surface.csv", index=False)
    term_structure.to_csv(output / "term_structure.csv", index=False)
    divergence.to_csv(output / "iv_divergence.csv", index=False)
    parity.to_csv(output / "parity_residuals.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(to_jsonable(metrics_payload), indent=2, allow_nan=False))
    build_fit_slices(fits, sample_dates, output / "fit_slices.html")
    build_surface(fits, output / "surface.html")

    print_summary(console, params, metrics, monthly_rows, sample_date_rows)
    console.print(f"[green]Wrote {output}[/green]")
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
