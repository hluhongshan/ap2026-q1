# AGENTS.md

## Goal
Complete Q1, Q2 and Q3 of the Advanced Programming 2026 brief, options A and B of each.
Q1: fit an arbitrage-free implied volatility surface w(k,T) on China index/ETF options.
Q2: price path-dependent payoffs by Monte Carlo under GBM; learn a hedge minimizing CVaR_0.95 of -V_T.
Q3: forecast the PMI contraction regime P(d_{t+1}=1) using only F_t = {x_{i,s}: s + L_i <= t}.

## Environment
Python 3.13, managed with uv, dependencies pinned.
uv sync --python 3.13 && source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, all approaches and post-hoc scripts, compare
python code/compare.py   # writes results/comparison.html with every metrics.json included

## Approach map
- approach1_q1a_svi.py: Q1A SVI surface
- approach2_q1b_neural.py: Q1B neural surface
- approach3_q2a_mc.py: Q2A Monte Carlo pricing
- approach4_q2b_deephedge.py: Q2B deep hedging
- approach5_q3a_classical.py (+ posthoc): Q3A classical probit
- approach6_q3b_sequence.py (+ posthoc): Q3B LSTM and leakage ablation

## Conventions
Seed: --seed 42 by default, CLI-overridable; every stochastic step is seeded.
Each code/approachN_*.py: argparse CLI, load -> verify -> split -> fit -> evaluate -> write.
Output: results/approachN/<dataset>/metrics.json = {params, metrics, fit_seconds, history}.
Terminal output via rich; figures via plotly HTML with include_plotlyjs=True.
Every external data source has a committed snapshot; grading runs offline.

## Agent rules
Write nothing outside the repo root; tool caches stay in the repo and are gitignored.
Propose first and wait for approval before any destructive operation.
Commit after each milestone (data:, feat:, exp:, report:).
Never commit credentials. Never invent results: every reported number comes from a metrics.json.
