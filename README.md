# AP 2026

Advanced Programming 2026 coursework: Q1 implied-volatility surfaces, Q2 Monte Carlo and CVaR hedging, Q3 PMI regime forecasting.

## Run

`uv sync --python 3.13 && source .venv/bin/activate`

Full reproduction: `./reproduce.sh`
Quick checks (tests + comparison): `./reproduce.sh --quick`
Clean-clone comparison: `python code/check_reproduction.py`

## Results

- `results/approach2/io/` is a historical run from commit `664a716` whose code
  has since changed. It is kept for the record and is **not** rerun by
  `./reproduce.sh`; current Q1B output is `results/approach2/io_scaled/`.
