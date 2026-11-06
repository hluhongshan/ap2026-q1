# AGENTS.md

## Goal
Q1 of the Advanced Programming 2026 brief, options A and B: fit an arbitrage-free implied-volatility surface w(k,T) on China index/ETF options. Option A is SVI slices; Option B is a neural surface with SVI solver comparisons.

## Environment
Python 3.13, managed with uv, dependencies pinned.
uv sync --python 3.13 && source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, Q1 approaches, compare
python code/check_reproduction.py

## Approach map
- approach1_q1a_svi.py: Q1A SVI surface
- approach2_q1b_neural.py: Q1B neural surface
- approach2_svi_cold_start.py: Q1B cold-start SVI solver comparison

## Data
data/clean/options_io.csv, data/clean/shibor.csv and data/clean/sina_comparison.csv are committed snapshots. data/raw/cffex/ is the committed CFFEX snapshot used by the forward check. data/legacy/ holds the legacy SVI snapshots used only for the eta-bound comparison. data/metadata.json is filtered to Q1 artifacts; data/fetch.py --offline verifies every checksum.

## Conventions
Seed: --seed 42 by default, CLI-overridable; every stochastic step is seeded.
Each code/approach*.py: argparse CLI, load -> verify -> split -> fit -> evaluate -> write.
Output: results/approachN/<dataset>/metrics.json = {params, metrics, fit_seconds, history}.
Terminal output via rich; figures via plotly HTML with include_plotlyjs=True.
results/approach2/io is a historical run from the parent repository and is not rerun.

## Reproduction
`./reproduce.sh` is the full offline path; `./reproduce.sh --quick` runs tests and compare only.
`python code/check_reproduction.py` compares `.cache/clean_clone_full/results` against the committed `results`, ignores timing fields, and uses relative tolerance 1e-6.

## Provenance
This repository was split from the combined AP 2026 repository at parent tag `submission`, commit `4521da3`. See `SPLIT_HISTORY.md` for the old-to-new commit map and the split disclosure.

## Agent rules
Write nothing outside the repo root; tool caches stay in the repo and are gitignored.
Propose first and wait for approval before any destructive operation.
Commit after each milestone (data:, feat:, exp:, refactor:, docs:).
Never commit credentials. Never invent results: every reported number comes from a metrics.json.
