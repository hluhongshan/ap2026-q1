# AP 2026

Advanced Programming 2026 coursework: Q1 arbitrage-free implied volatility surfaces, Q2 Monte Carlo pricing and CVaR hedging, Q3 PMI regime forecasting.

## Setup

```bash
uv sync --python 3.13
source .venv/bin/activate
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache"
```

`uv.toml` pins uv's cache to `.uv-cache/`; `.cache/` and `.uv-cache/` are gitignored so tool caches stay inside the repository.

## Layout

- `data/raw/` — downloaded source snapshots (gitignored)
- `data/clean/` — committed cleaned snapshots for offline grading
- `code/` — analysis and model scripts
- `results/` — metrics and figures, one directory per approach

## Reproducibility

All stochastic steps are seeded (`--seed 42` by default). Every reported number must come from a `results/approachN/<dataset>/metrics.json`.
