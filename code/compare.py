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
                    "status": "reference",
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
            if item["dataset"] == "io_scaled":
                status = "primary"
                used_label = "Q1B primary: scaled neural vs SVI"
            elif item["dataset"] == "io":
                status = "superseded: optimization failure, kept for the record"
                used_label = "Q1B superseded record"
            else:
                status = "secondary"
                used_label = "Q1 neural vs SVI comparison"
            rows.append(
                {
                    "source": item["rel"],
                    "dataset": item["dataset"],
                    "status": status,
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
                    "status": status,
                    "model": "SVI re-priced",
                    "median": comparison["approach1"]["median"],
                    "p95": comparison["approach1"]["p95"],
                    "butterfly": None,
                    "calendar": None,
                    "n": comparison["common_quotes"],
                    "fit_seconds": fit_seconds,
                }
            )
            used[item["rel"]] = used_label
    rows.sort(key=lambda row: float(row["median"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, used


def build_html(coverage: list[list[str]], q1: list[list[str]]) -> go.Figure:
    sections = [
        ("Source coverage: every results/approach*/*/metrics.json", coverage),
        ("Q1: SVI vs neural repricing and violations (primary Q1B = io_scaled; io superseded)", q1),
    ]
    counts = [max(1, len(section[1]) - 1) for section in sections]
    total = sum(counts)
    heights = [max(0.08, count / total) for count in counts]
    figure = make_subplots(
        rows=len(sections), cols=1,
        specs=[[{"type": "table"}] for _ in sections],
        subplot_titles=[section[0] for section in sections],
        vertical_spacing=0.035, row_heights=heights,
    )
    for index, section in enumerate(sections, start=1):
        figure.add_trace(make_plotly_table(section[1]), row=index, col=1)
    figure.update_layout(
        title="Q1 cross-approach comparison (all values read from metrics.json)",
        height=max(900, 320 + 34 * sum(counts)),
        margin=dict(l=20, r=20, t=80, b=20), template="plotly_white",
    )
    figure.update_annotations(font_size=13)
    return figure


def main() -> int:
    payloads = load_all()
    used: dict[str, str] = {}
    q1, used_q1 = q1_rows(payloads)
    used.update(used_q1)
    coverage = coverage_rows(payloads, used)
    console.print(rich_table(
        "Source coverage: every results/approach*/*/metrics.json",
        ["metrics.json", "approach", "dataset", "fit_seconds", "used for"], coverage,
    ))
    q1_table = [[
        row["rank"], row["source"].removeprefix("results/").split("/")[0], row["dataset"],
        row["status"], row["model"], fmt(row["median"]), fmt(row["p95"]),
        fmt(row["butterfly"], 0), fmt(row["calendar"], 0), row["n"], fmt(row["fit_seconds"], 2),
    ] for row in q1]
    console.print(rich_table(
        "Q1: SVI vs neural repricing and violations (primary Q1B = io_scaled; io superseded)",
        ["rank", "source", "dataset", "status", "model", "median", "p95", "butterfly", "calendar", "n", "fit_s"],
        q1_table,
    ))
    figure = build_html(coverage, q1_table)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)
    console.print(f"wrote {OUT_HTML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
