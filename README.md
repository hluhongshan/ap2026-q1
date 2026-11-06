# AP 2026 Q1: implied-volatility surfaces

Q1 of the Advanced Programming 2026 coursework: arbitrage-free implied-volatility surfaces on CFFEX IO options, with SVI slices (Q1A) and a neural surface (Q1B).

## Run

`uv sync --python 3.13 && source .venv/bin/activate`

Full reproduction: `./reproduce.sh`
Quick checks (tests + comparison): `./reproduce.sh --quick`
Clean-clone comparison: `python code/check_reproduction.py`

## Results

- `results/approach1/io/` and `results/approach1/io_close/` are the Q1A settle and close runs.
- `results/approach2/io_scaled/` is the primary Q1B result; `results/approach2/io/` is the historical pre-fix run, labelled historical and not rerun.

## Provenance

Split from the combined AP 2026 repository at parent tag `submission`, commit `4521da3`. See `SPLIT_HISTORY.md` for the commit map and disclosure. `SESSION.md` is the full transcript for all questions and uses the old file names and hashes.
