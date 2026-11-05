#!/usr/bin/env python3
"""Compare numeric values in a reproduced results tree against committed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "results"
DEFAULT_CANDIDATE = ROOT / ".cache" / "clean_clone_full" / "results"
RELATIVE_TOLERANCE = 1e-6
TIMING_TOKENS = (
    "second",
    "time",
    "duration",
    "elapsed",
    "speedup",
    "per_second",
    "timestamp",
)


def is_timing_key(key: str) -> bool:
    lower = key.lower()
    return any(token in lower for token in TIMING_TOKENS)


def flatten(value, path: str = "", out: dict[str, float] | None = None) -> dict[str, float]:
    if out is None:
        out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if is_timing_key(str(key)):
                continue
            child = f"{path}.{key}" if path else str(key)
            flatten(item, child, out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten(item, f"{path}[{index}]", out)
    elif isinstance(value, bool):
        return out
    elif isinstance(value, (int, float)):
        out[path] = value
    return out


def relative_difference(reference: float, candidate: float) -> float:
    scale = max(abs(reference), abs(candidate))
    if scale == 0.0:
        return 0.0
    return abs(candidate - reference) / scale


def compare_file(reference_path: Path, candidate_path: Path) -> dict:
    reference = flatten(json.loads(reference_path.read_text(encoding="utf-8")))
    candidate = flatten(json.loads(candidate_path.read_text(encoding="utf-8")))
    shared = sorted(set(reference) & set(candidate))
    differences = []
    exact = 0
    max_abs = 0.0
    max_rel = 0.0
    max_abs_key = ""
    max_rel_key = ""
    for key in shared:
        left = float(reference[key])
        right = float(candidate[key])
        if left == right:
            exact += 1
            continue
        absolute = abs(right - left)
        relative = relative_difference(left, right)
        if absolute > max_abs:
            max_abs = absolute
            max_abs_key = key
        if relative > max_rel:
            max_rel = relative
            max_rel_key = key
        differences.append(
            {
                "key": key,
                "reference": left,
                "candidate": right,
                "abs": absolute,
                "rel": relative,
            }
        )
    differences.sort(key=lambda item: (item["abs"], item["rel"]), reverse=True)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "compared": len(shared),
        "exact": exact,
        "max_abs": max_abs,
        "max_abs_key": max_abs_key,
        "max_rel": max_rel,
        "max_rel_key": max_rel_key,
        "differences": differences,
        "only_candidate": sorted(set(candidate) - set(reference)),
        "only_reference": sorted(set(reference) - set(candidate)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare reproduced metrics.json against committed metrics.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    reference_root = args.reference.resolve()
    candidate_root = args.candidate.resolve()
    if not reference_root.is_dir():
        raise SystemExit(f"missing reference directory: {reference_root}")
    if not candidate_root.is_dir():
        raise SystemExit(f"missing candidate directory: {candidate_root}")

    reference_files = {path.relative_to(reference_root): path for path in reference_root.glob("approach*/*/metrics.json")}
    candidate_files = {path.relative_to(candidate_root): path for path in candidate_root.glob("approach*/*/metrics.json")}
    if not candidate_files:
        raise SystemExit(f"no metrics.json files under {candidate_root}")

    console = Console(width=220)
    table = Table(title="Reproduction numeric comparison")
    for column in ("file", "compared", "exact", "max_abs", "max_rel", "top differing keys"):
        table.add_column(column, justify="right" if column not in {"file", "top differing keys"} else "left")

    reports = []
    failures = []
    for relative in sorted(candidate_files):
        if relative not in reference_files:
            reports.append({"file": str(relative), "status": "missing_reference"})
            table.add_row(str(relative), "0", "0", "n/a", "n/a", "missing committed reference")
            failures.append(str(relative))
            continue
        report = compare_file(reference_files[relative], candidate_files[relative])
        report["file"] = str(relative)
        reports.append(report)
        top_keys = ", ".join(item["key"] for item in report["differences"][: args.top]) or "-"
        table.add_row(
            str(relative),
            f"{report['compared']}",
            f"{report['exact']}",
            f"{report['max_abs']:.3e}",
            f"{report['max_rel']:.3e}",
            top_keys,
        )
        if report["max_rel"] > args.tolerance or report["only_candidate"]:
            failures.append(str(relative))

    console.print(table)

    detail = Table(title=f"Top differing keys (relative tolerance {args.tolerance:g})")
    for column in ("file", "key", "committed", "reproduced", "abs_diff", "rel_diff"):
        detail.add_column(column, justify="left" if column in {"file", "key"} else "right")
    for report in reports:
        for item in report.get("differences", [])[: args.top]:
            detail.add_row(
                report["file"],
                item["key"],
                f"{item['reference']:.17g}",
                f"{item['candidate']:.17g}",
                f"{item['abs']:.3e}",
                f"{item['rel']:.3e}",
            )
    if any(report.get("differences") for report in reports):
        console.print(detail)
    else:
        console.print("[green]all compared numeric values are exact[/green]")

    for report in reports:
        if report.get("only_candidate"):
            console.print(f"[yellow]{report['file']}: {len(report['only_candidate'])} numeric keys only in reproduced run[/yellow]")
        if report.get("only_reference"):
            console.print(f"[yellow]{report['file']}: {len(report['only_reference'])} numeric keys only in committed run[/yellow]")

    missing = sorted(set(reference_files) - set(candidate_files))
    if missing:
        console.print("[yellow]committed scenarios not present in the reproduced tree (not rerun by design):[/yellow]")
        for relative in missing:
            console.print(f"  {relative}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"tolerance": args.tolerance, "reports": reports}, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    if failures:
        console.print(f"[red]FAIL: {len(failures)} file(s) exceed tolerance or have unmatched keys[/red]")
        return 1
    console.print("[green]PASS: every compared numeric value is within tolerance[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
