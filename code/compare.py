#!/usr/bin/env python3
"""Cross-approach comparison tables, read only from results/**/metrics.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_HTML = RESULTS / "comparison.html"

console = Console()


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_pm(mean, std, digits: int = 4) -> str:
    if mean is None:
        return "n/a"
    if std is None or not math.isfinite(float(std)):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ± {fmt(std, digits)}"


def load_all() -> list[dict]:
    paths = sorted(RESULTS.glob("approach*/*/metrics.json"))
    if not paths:
        raise SystemExit(f"no metrics.json files under {RESULTS}")
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if list(payload) != ["params", "metrics", "fit_seconds", "history"]:
            raise ValueError(f"{path}: unexpected top-level keys {list(payload)}")
        payloads.append(
            {
                "path": path,
                "rel": str(path.relative_to(ROOT)),
                "approach": path.parts[-3],
                "dataset": path.parts[-2],
                "payload": payload,
            }
        )
    return payloads


def q1_rows(payloads: list[dict]) -> tuple[list[dict], dict[str, str]]:
    rows = []
    used = {}
    for item in payloads:
        approach = item["approach"]
        metrics = item["payload"]["metrics"]
        fit_seconds = item["payload"]["fit_seconds"]
        if approach == "approach1":
            repricing = metrics["repricing"]["otm"]["raw"]
            butterfly = metrics["butterfly"]["raw"]
            calendar = metrics["calendar"]["raw"]
            rows.append(
                {
                    "source": item["rel"],
                    "dataset": item["dataset"],
                    "model": "SVI",
                    "median": repricing["per_fit_median_median"],
                    "p95": repricing["per_fit_max_p95"],
                    "butterfly": butterfly["violations_strict"],
                    "calendar": calendar["violations"],
                    "n": repricing["fits"],
                    "fit_seconds": fit_seconds,
                }
            )
            used[item["rel"]] = "Q1 SVI repricing and violations"
        elif approach == "approach2":
            chosen = metrics["chosen_model"]
            comparison = metrics["comparison_approach1"]
            rows.append(
                {
                    "source": item["rel"],
                    "dataset": item["dataset"],
                    "model": "neural",
                    "median": chosen["in_sample_repricing"]["median"],
                    "p95": chosen["in_sample_repricing"]["p95"],
                    "butterfly": chosen["violations"]["butterfly_violations_strict"],
                    "calendar": chosen["violations"]["calendar_violations_strict"],
                    "n": comparison["common_quotes"],
                    "fit_seconds": fit_seconds,
                }
            )
            rows.append(
                {
                    "source": item["rel"],
                    "dataset": item["dataset"],
                    "model": "SVI re-priced",
                    "median": comparison["approach1"]["median"],
                    "p95": comparison["approach1"]["p95"],
                    "butterfly": None,
                    "calendar": None,
                    "n": comparison["common_quotes"],
                    "fit_seconds": fit_seconds,
                }
            )
            used[item["rel"]] = "Q1 neural vs SVI comparison"
    rows.sort(key=lambda row: float(row["median"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, used


def q2a_rows(payloads: list[dict]) -> tuple[list[dict], list[dict], dict[str, str]]:
    rows = []
    study = []
    used = {}
    for item in payloads:
        if item["approach"] != "approach3":
            continue
        metrics = item["payload"]["metrics"]
        fit_seconds = item["payload"]["fit_seconds"]
        if item["dataset"] == "IF_study":
            convergence = metrics["convergence"]["pricing"]["asian"]
            bias = metrics["bias_variance"]["table"]["delta"]["asian"]["0.01"]
            study.append(
                {
                    "source": item["rel"],
                    "naive_slope": convergence["naive"]["slope"],
                    "control_variate_slope": convergence["control_variate"]["slope"],
                    "speedup": metrics["performance"]["speedup"],
                    "bias": bias["bias"],
                    "variance": bias["variance"],
                    "fit_seconds": fit_seconds,
                }
            )
            used[item["rel"]] = "Q2A convergence and bias-variance study"
            continue
        asian = metrics["pricing"]["asian"]
        barrier = metrics["pricing"]["barrier"]
        rows.append(
            {
                "source": item["rel"],
                "dataset": item["dataset"],
                "validation_z": metrics["validation_european"]["z"],
                "geometric_z": metrics["closed_form_checks"]["geometric_call_z"],
                "engine_z": asian["engine_agreement_z"],
                "asian_antithetic": asian["variance_ratio_vs_naive"]["antithetic"],
                "asian_control_variate": asian["variance_ratio_vs_naive"]["control_variate"],
                "asian_cv_antithetic": asian["variance_ratio_vs_naive"]["cv_antithetic"],
                "barrier_cv_antithetic": barrier["variance_ratio_vs_naive"]["cv_antithetic"],
                "passes_5x": metrics["variance_reduction_benchmark_asian"]["passes_5x"],
                "fit_seconds": fit_seconds,
            }
        )
        used[item["rel"]] = "Q2A z-scores and variance reduction"
    rows.sort(key=lambda row: float(row["asian_control_variate"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, study, used


def q2b_rows(payloads: list[dict]) -> tuple[list[dict], dict[str, str]]:
    rows = []
    used = {}
    for item in payloads:
        if item["approach"] != "approach4":
            continue
        for cell in item["payload"]["metrics"]["table"]:
            rows.append(
                {
                    "source": item["rel"],
                    "method": cell["method"],
                    "rebalance": cell["rebalance_days"],
                    "policy_kappa": cell["policy_kappa_bp"],
                    "eval_kappa": cell["eval_kappa_bp"],
                    "pnl": cell["pnl"],
                    "cvar": cell["cvar_mean"],
                    "net_cvar": cell["net_cvar_mean"],
                    "net_cvar_std": cell["net_cvar_std"],
                    "variance": cell["variance_mean"],
                    "mean_pnl": cell["mean_pnl_mean"],
                    "mean_pnl_std": cell["mean_pnl_std"],
                    "turnover": cell["turnover_mean"],
                    "fit_seconds": item["payload"]["fit_seconds"],
                }
            )
        used[item["rel"]] = "Q2B CVaR, variance and expected P&L"
    rows.sort(key=lambda row: (0 if row["pnl"] == "brief" else 1, float(row["net_cvar"]), row["method"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, used


def summary_block(block: dict) -> tuple[float, float, float, float, float, float, int | None]:
    if "Brier_mean" in block:
        return (
            block["Brier_mean"],
            block["Brier_std"],
            block["AUC_mean"],
            block["AUC_std"],
            block["accuracy_mean"],
            block["accuracy_std"],
            block.get("n"),
        )
    return (
        block["Brier"],
        0.0,
        block["AUC"],
        0.0,
        block["accuracy"],
        0.0,
        block.get("n"),
    )


def q3_rows(payloads: list[dict]) -> tuple[list[dict], dict[str, str]]:
    rows = []
    used = {}
    for item in payloads:
        metrics = item["payload"]["metrics"]
        if item["approach"] == "approach5":
            if "probit" in metrics and "test" in metrics["probit"]:
                for arm, block in metrics["probit"]["test"].items():
                    brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                    rows.append(
                        {
                            "source": item["rel"],
                            "arm": arm,
                            "vintage": "original",
                            "brier": brier,
                            "brier_std": brier_std,
                            "auc": auc,
                            "auc_std": auc_std,
                            "accuracy": accuracy,
                            "accuracy_std": accuracy_std,
                            "n": n,
                        }
                    )
            post_hoc = metrics.get("post_hoc", {})
            for block in post_hoc.get("test_table", []):
                brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                rows.append(
                    {
                        "source": item["rel"],
                        "arm": block["model"],
                        "vintage": "post_hoc",
                        "brier": brier,
                        "brier_std": brier_std,
                        "auc": auc,
                        "auc_std": auc_std,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "n": n,
                    }
                )
            used[item["rel"]] = "Q3 original and post_hoc probit arms"
        elif item["approach"] == "approach6":
            for arm, block in metrics.get("comparison", {}).items():
                brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                rows.append(
                    {
                        "source": item["rel"],
                        "arm": arm,
                        "vintage": "original",
                        "brier": brier,
                        "brier_std": brier_std,
                        "auc": auc,
                        "auc_std": auc_std,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "n": n,
                    }
                )
            retrain = metrics.get("post_hoc", {}).get("retrain", {})
            for arm, block in retrain.get("comparison", {}).items():
                brier, brier_std, auc, auc_std, accuracy, accuracy_std, n = summary_block(block)
                rows.append(
                    {
                        "source": item["rel"],
                        "arm": arm,
                        "vintage": "post_hoc",
                        "brier": brier,
                        "brier_std": brier_std,
                        "auc": auc,
                        "auc_std": auc_std,
                        "accuracy": accuracy,
                        "accuracy_std": accuracy_std,
                        "n": n,
                    }
                )
            used[item["rel"]] = "Q3 original and post_hoc LSTM/probit arms"
    rows.sort(key=lambda row: float(row["brier"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, used


def coverage_rows(payloads: list[dict], used: dict[str, str]) -> list[list]:
    rows = []
    for item in payloads:
        rows.append(
            [
                item["rel"].removeprefix("results/").removesuffix("/metrics.json"),
                item["approach"],
                item["dataset"],
                fmt(item["payload"]["fit_seconds"], 2),
                used.get(item["rel"], "UNUSED"),
            ]
        )
    unused = [row for row in rows if row[-1] == "UNUSED"]
    if unused:
        raise ValueError(f"metrics.json files excluded silently: {unused}")
    return rows


def rich_table(title: str, columns: list[str], rows: list[list]) -> Table:
    table = Table(title=title, show_lines=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*[str(value) for value in row])
    return table


def make_plotly_table(section_rows: list[list[str]], header_color: str = "#1f4e79") -> go.Table:
    n_cols = len(section_rows[0])
    columns = [[row[i] for row in section_rows[1:]] for i in range(n_cols)]
    return go.Table(
        header=dict(
            values=section_rows[0],
            fill_color=header_color,
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=columns,
            fill_color=["#f7f9fc", "#eef2f7"],
            align="left",
            font=dict(size=11),
            height=24,
        ),
        columnwidth=[1.2] + [1.0] * (n_cols - 1),
    )


def build_html(
    coverage: list[list[str]],
    q1: list[list[str]],
    q2a: list[list[str]],
    q2a_study: list[list[str]],
    q2b: list[list[str]],
    q3: list[list[str]],
) -> go.Figure:
    sections = [
        ("Source coverage: every results/approach*/*/metrics.json", coverage),
        ("Q1: SVI vs neural repricing and violations (ranked by median repricing error)", q1),
        ("Q2A: z-scores and variance reduction (ranked by Asian control-variate ratio)", q2a),
        ("Q2A supplementary study: convergence and bias-variance", q2a_study),
        ("Q2B: CVaR, variance and expected P&L (ranked within P&L definition by net CVaR)", q2b),
        ("Q3: Brier, AUC and accuracy, original and post-hoc arms (ranked by Brier)", q3),
    ]
    counts = [max(1, len(section[1]) - 1) for section in sections]
    total = sum(counts)
    heights = [max(0.08, count / total) for count in counts]
    figure = make_subplots(
        rows=len(sections),
        cols=1,
        specs=[[{"type": "table"}] for _ in sections],
        subplot_titles=[section[0] for section in sections],
        vertical_spacing=0.035,
        row_heights=heights,
    )
    for index, section in enumerate(sections, start=1):
        figure.add_trace(make_plotly_table(section[1]), row=index, col=1)
    figure.update_layout(
        title="AP 2026 cross-approach comparison (all values read from metrics.json)",
        height=max(1800, 320 + 34 * sum(counts)),
        margin=dict(l=20, r=20, t=80, b=20),
        template="plotly_white",
    )
    figure.update_annotations(font_size=13)
    return figure


def main() -> int:
    payloads = load_all()
    used: dict[str, str] = {}
    q1, used_q1 = q1_rows(payloads)
    used.update(used_q1)
    q2a, q2a_study, used_q2a = q2a_rows(payloads)
    used.update(used_q2a)
    q2b, used_q2b = q2b_rows(payloads)
    used.update(used_q2b)
    q3, used_q3 = q3_rows(payloads)
    used.update(used_q3)
    coverage = coverage_rows(payloads, used)

    console.print(
        rich_table(
            "Source coverage: every results/approach*/*/metrics.json",
            ["metrics.json", "approach", "dataset", "fit_seconds", "used for"],
            coverage,
        )
    )
    q1_table = [
        [
            row["rank"],
            row["source"].removeprefix("results/").split("/")[0],
            row["dataset"],
            row["model"],
            fmt(row["median"]),
            fmt(row["p95"]),
            fmt(row["butterfly"], 0),
            fmt(row["calendar"], 0),
            row["n"],
            fmt(row["fit_seconds"], 2),
        ]
        for row in q1
    ]
    console.print(
        rich_table(
            "Q1: SVI vs neural repricing and violations (ranked by median repricing error)",
            ["rank", "source", "dataset", "model", "median", "p95", "butterfly", "calendar", "n", "fit_s"],
            q1_table,
        )
    )
    q2a_table = [
        [
            row["rank"],
            row["source"].removeprefix("results/").split("/")[0],
            row["dataset"],
            fmt(row["validation_z"], 3),
            fmt(row["geometric_z"], 3),
            fmt(row["engine_z"], 3),
            fmt(row["asian_antithetic"], 2),
            fmt(row["asian_control_variate"], 2),
            fmt(row["asian_cv_antithetic"], 2),
            fmt(row["barrier_cv_antithetic"], 2),
            str(row["passes_5x"]),
            fmt(row["fit_seconds"], 2),
        ]
        for row in q2a
    ]
    console.print(
        rich_table(
            "Q2A: z-scores and variance reduction (ranked by Asian control-variate ratio)",
            ["rank", "source", "dataset", "val z", "geom z", "engine z", "anti", "CV", "CV+anti", "barrier CV+anti", "5x", "fit_s"],
            q2a_table,
        )
    )
    q2a_study_table = [
        [
            row["source"].removeprefix("results/").split("/")[0],
            fmt(row["naive_slope"], 3),
            fmt(row["control_variate_slope"], 3),
            fmt(row["speedup"], 2),
            fmt(row["bias"], 5),
            fmt(row["variance"], 8),
            fmt(row["fit_seconds"], 2),
        ]
        for row in q2a_study
    ]
    console.print(
        rich_table(
            "Q2A supplementary study: convergence and bias-variance",
            ["source", "naive slope", "CV slope", "speedup", "bias@0.01", "variance@0.01", "fit_s"],
            q2a_study_table,
        )
    )
    q2b_table = [
        [
            row["rank"],
            row["source"].removeprefix("results/").rsplit("/", 1)[0],
            row["method"],
            "n/a" if row["rebalance"] is None else row["rebalance"],
            "n/a" if row["policy_kappa"] is None else row["policy_kappa"],
            row["eval_kappa"],
            row["pnl"],
            fmt_pm(row["net_cvar"], row["net_cvar_std"]),
            fmt(row["cvar"]),
            fmt(row["variance"]),
            fmt_pm(row["mean_pnl"], row["mean_pnl_std"]),
            fmt(row["turnover"], 3),
        ]
        for row in q2b
    ]
    console.print(
        rich_table(
            "Q2B: CVaR, variance and expected P&L (ranked within P&L definition by net CVaR)",
            ["rank", "source", "method", "rebal", "policy k", "eval k", "pnl", "net CVaR", "CVaR", "variance", "E[P&L]", "turnover"],
            q2b_table,
        )
    )
    q3_table = [
        [
            row["rank"],
            row["source"].removeprefix("results/").rsplit("/", 1)[0],
            row["arm"],
            row["vintage"],
            fmt_pm(row["brier"], row["brier_std"]),
            fmt_pm(row["auc"], row["auc_std"]),
            fmt_pm(row["accuracy"], row["accuracy_std"]),
            row["n"],
        ]
        for row in q3
    ]
    console.print(
        rich_table(
            "Q3: Brier, AUC and accuracy, original and post-hoc arms (ranked by Brier)",
            ["rank", "source", "arm", "vintage", "Brier", "AUC", "accuracy", "n"],
            q3_table,
        )
    )

    figure = build_html(coverage, q1_table, q2a_table, q2a_study_table, q2b_table, q3_table)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)
    console.print(f"wrote {OUT_HTML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
