# Advanced Programming 2026 Project Report — Q1

Hao Lu

## Provenance and split disclosure

The work was done in one repository (parent tag `submission`, commit `4521da3`) and split afterwards into one repository per question following the instructor's guidance. Commit hashes changed and the old-to-new commit map is included in `SPLIT_HISTORY.md`. Dates in rewritten commits are the original ones; the split commits are dated when they were made. Shared-file commits appear in more than one repository. `SESSION.md` is the full transcript for all questions and refers to old file names and hashes. Agent work on the split ran with `ap2026-split` as its root.

Section and table numbers below follow the parent report; only this question's sections are included.

### Old-to-new mapping

| Parent | This repository |
|---|---|
| `code/approach1_q1a_svi.py` | `code/approach1_q1a_svi.py` (unchanged) |
| `code/approach2_q1b_neural.py` | `code/approach2_q1b_neural.py` (unchanged) |
| `code/approach2_svi_cold_start.py` | `code/approach2_svi_cold_start.py` (unchanged) |
| `results/approach1/` | `results/approach1/` (unchanged) |
| `results/approach2/` | `results/approach2/` (unchanged) |

## 1. Summary

The repository implements Q1 of the Advanced Programming 2026 brief, options A and B. Q1 fits an arbitrage-free implied-volatility surface to CFFEX IO options, with SVI slices (Q1A) and a neural surface (Q1B).

Q1A fits all 693 liquid date-expiry slices on the settle price and reaches zero butterfly and zero calendar violations at the 1e-8 tolerance on its audit grid, but 1 butterfly and 12 calendar checks remain strictly negative after repair. The median per-fit median repricing error is 0.77% on settle and 1.27% on close; the median per-fit maximum is 2.58% and 4.33%. Q1B is close to SVI in median repricing on the same 470 OTM quotes (0.87% vs 0.82%) but much worse in the tail (p95 12.34% vs 5.45%; maximum 98.30% vs 17.37%), and it violates arbitrage on a denser grid outside its training collocation.

What was not achieved: Q1B is not globally arbitrage-free and is not competitive with SVI in the wings. The data are daily, without bid-ask quotes.

## 2. Environment and one-command reproduction

The project uses Python 3.13 under `uv` with pinned dependencies. From the repository root:

```bash
uv sync --python 3.13
source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, Q1 approaches, compare
python code/check_reproduction.py
```

`./reproduce.sh` runs `data/fetch.py --offline`, the Q1 tests, Q1A settle and close, Q1B neural and cold-start SVI, and `compare.py`. The offline check verifies the three committed clean artifacts against `data/metadata.json` and reports that all checksums match. The test suite ran 11 tests and passed.

The clean-clone check compares `.cache/clean_clone_full/results` with the committed results. The full clean-clone run completed in 3112 s with HTTP_PROXY and HTTPS_PROXY pointed at `http://127.0.0.1:9`. The checker reports 19,184 compared numeric values, all exact, with maximum absolute and relative differences 0. It excludes `results/approach2/io/metrics.json` as historical, not rerun. There were no differing time-capped Q1B fits: every shared numeric value matched.

The sum of committed `fit_seconds` for the Q1 reruns is 3,010.1 s, about 50.2 minutes; the clean-clone wall time was 51.9 minutes.

## 3. Data

### 3.1 Sources, coverage and snapshots

Table 1: Committed clean artifacts in this repository. Source: `data/metadata.json`.

| artifact | source | rows | retrieved |
|---|---|---|---|
| `data/clean/options_io.csv` | CFFEX IO daily quotes; CSI 300 close from akshare.stock_zh_index_daily | 30068 | 2026-11-04 |
| `data/clean/shibor.csv` | China Shibor | 2398 | 2026-11-04 |
| `data/clean/sina_comparison.csv` | CFFEX close vs Sina option daily close | 30 | 2026-11-04 |

The raw CFFEX snapshot is committed through a `.gitignore` exception: 128 daily files, of which 126 parse. `20240917_1.csv` and `20260925_1.csv` are non-CSV 200 pages with missing columns. `data/legacy/io_settle_svi_params.csv` and `data/legacy/io_close_svi_params.csv` are committed snapshots used only for the eta-bound comparison; they have 693 fits each.

The Sina cross-check compared 30 rows in `data/clean/sina_comparison.csv`: 29 matched the CFFEX close with zero difference, and 1 failed (IO2603-P-4650). The comparison is a close-price spot check, not a full second price source.

The IF forward check uses raw CFFEX IF rows, not a main-contract series. Each of the 126 parsed files lists four IF contracts for the date. The check merges IO forwards to IF contracts by date and expiry year-month. Far IO expiries have no same-month IF contract listed, so the unmatched count is structural: 498 of 750 IO forwards match, 252 do not. This was verified in `load_cffex_futures`, which filters every IF contract from the raw file and merges on `expiry_month`, and in the raw files, which contain four IF contracts per date.

### 3.2 Q1 filtering, parity and implied volatility

Q1 uses the CFFEX `settle` field as the primary price and repeats the pipeline on `close`. There is no bid-ask snapshot, so liquidity is proxied by volume. Filters are volume >= 10, price >= 1, `|k| <= 0.20` and days-to-expiry >= 7. The volume floor stands in for a missing spread, the price floor removes near-zero quotes, the strike band keeps the fit local enough for a raw SVI slice, and the days floor avoids degenerate short tenors. On settle, sequential removals are 8,008 by volume, 1,303 by price, 1,380 by strike band and 606 by days, leaving 18,771 liquid quotes. There are 750 date-expiry groups, 728 with liquid quotes, 693 fits and 57 skipped groups (34 with fewer than six strikes, 23 with no liquid OTM quote). The OTM quote set has 11,146 quotes. Rate convention is `r(tau) = ln(1 + L/100 * tau * 365/360) / tau` from SHIBOR, linear in tenor and flat outside pillars. The forward is a volume-weighted mean of `F_K = K + (C-P)/DF` within `|ln(F_K / median(F_K))| <= 0.05`; `q = r - ln(F/S0)/T`.
Table 2: Put-call parity residuals and implied q, settle. Source: `results/approach1/io/metrics.json`.

| statistic | value |
|---|---|
| pairs | 15,034 |
| absolute residual vs S0, median | 37.17 |
| absolute residual vs S0, p95 | 193.5 |
| absolute residual vs F, median | 1.303 |
| absolute residual vs F, p95 | 13.8 |
| implied q, median | 0.04401 |
| implied q, p05 | 0.004882 |
| implied q, p95 | 0.09258 |
| in-band fraction | 0.9996 |
| fallback forwards | 0 |

Table 3: Implied-volatility failures. Source: `results/approach1/io/metrics.json` and `results/approach1/io_close/metrics.json`.

| run | attempted | ok | C ok | C below intrinsic | P ok | P below intrinsic | paired strikes |
|---|---|---|---|---|---|---|---|
| settle | 18,771 | 18,583 | 9,575 | 107 | 9,008 | 81 | 11,732 |
| close | 18,810 | 18,308 | 9,426 | 282 | 8,882 | 220 | 11,755 |

Table 4: Call-put implied-volatility gap on paired strikes. Source: `results/approach1/io/metrics.json` and `results/approach1/io_close/metrics.json`.

| run | gap available | median | mean | p95 of absolute gap | max absolute gap | n > 0.01 | n > 0.05 |
|---|---|---|---|---|---|---|---|
| settle | 6,956 | 0.0004639 | -0.0006521 | 0.04116 | 0.1763 | 1,163 | 272 |
| close | 6,699 | -5.406e-05 | -0.0009677 | 0.05035 | 0.3107 | 2,314 | 338 |

## 4. Q1 Option A: SVI surface

### 4.1 Model

For each date-expiry slice, with `k = ln(K/F)` and total variance `w = sigma^2 T`, the raw SVI form is `w(k) = a + b (rho (k-m) + sqrt((k-m)^2 + eta^2))`, with `b >= 0`, `|rho| < 1`, `eta > 0`, and bounds `a in [-2,2]`, `b in [0,5]`, `rho in [-0.999,0.999]`, `m in [-2,2]`, `eta in [0.01,1.0]`. Implied volatility is obtained by inverting the Black-76 price with `brentq` on `[1e-4,5]`. Fitting is bounded nonlinear least squares (`scipy.optimize.least_squares`, TRF, `x_scale="jac"`) with equal weights on liquid OTM quotes and eight documented starting points plus an optional warm start. Each raw fit is audited on `k in [-0.14,0.14]` step 0.01 with the correct Gatheral butterfly function `g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2`, and with the printed form that omits `w''/2`. Calendar arbitrage is checked by linear interpolation of total variance in `T` at fixed `k`, requiring nondecreasing `w`. Violations are repaired by penalized refits at levels `1e2`, `1e4`, `1e6`, followed by a forward calendar repair against the previous expiry. Strict counts use `g<0`; tolerance counts use `g<-1e-8`.

### 4.2 Results

Table 5: Q1A overall audits and OTM repricing. Source: `results/approach1/io/metrics.json` and `results/approach1/io_close/metrics.json`.

| statistic | settle | close |
|---|---|---|
| liquid quotes | 18,771 | 18,810 |
| OTM quotes used | 11,146 | 11,192 |
| fits / skipped | 693 / 57 | 693 / 57 |
| butterfly raw, correct strict | 8 | 23 |
| butterfly raw, printed strict | 382 | 418 |
| butterfly repaired, correct at 1e-8 | 0 | 0 |
| butterfly repaired, correct strict | 1 | 3 |
| butterfly repaired, printed strict | 390 | 423 |
| calendar raw / repaired at 1e-8 | 29 / 0 | 29 / 0 |
| calendar repaired strict | 12 | 11 |
| OTM median of per-fit max repricing | 0.02583 | 0.0433 |
| OTM p95 of per-fit max repricing | 0.1537 | 0.1598 |
| OTM median of per-fit median repricing | 0.007745 | 0.01269 |
| OTM worst per-fit max repricing | 0.3251 | 0.2624 |
| eta at lower / upper bound | 31 / 67 | 72 / 46 |

The settle run is primary. The legacy eta bounds `[1e-6,2.0]` put 27 fits at the lower bound and 8 at the upper bound; the current bounds put 31 at the lower and 67 at the upper. Median raw RMSE in total variance is 8.466e-05 under the legacy bounds and 8.468e-05 under the current bounds. On close the counts are 51/8 legacy and 72/46 current, with median RMSE 1.561e-04 in both cases.

Table 6: Q1A calibration by sample month, settle. Source: `results/approach1/io/metrics.json`.

| month | dates | fits | skipped | liquid | median q | butterfly raw / repaired | calendar raw / repaired | repricing max median | repricing max p95 | repricing strike median |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024-03 | 21 | 109 | 16 | 2,524 | 0.0356 | 0 / 0 | 2 / 0 | 0.02145 | 0.09777 | 0.006327 |
| 2024-09 | 19 | 109 | 4 | 2,515 | 0.02848 | 5 / 0 | 17 / 0 | 0.02728 | 0.101 | 0.009705 |
| 2025-03 | 21 | 116 | 9 | 3,079 | 0.03855 | 0 / 0 | 1 / 0 | 0.02486 | 0.1273 | 0.009425 |
| 2025-09 | 22 | 126 | 5 | 3,921 | 0.03696 | 0 / 0 | 0 / 0 | 0.01797 | 0.08802 | 0.006128 |
| 2026-03 | 22 | 127 | 4 | 3,823 | 0.07338 | 3 / 0 | 7 / 0 | 0.02987 | 0.1044 | 0.009897 |
| 2026-09 | 21 | 106 | 19 | 2,909 | 0.07986 | 0 / 0 | 2 / 0 | 0.02903 | 0.1076 | 0.007651 |

Table 7: Implied q versus IF basis by sample month, settle. Source: `results/approach1/io/metrics.json`.

| month | forwards | matched | implied q median | IF basis median | q - basis median | q - basis mean |
|---|---|---|---|---|---|---|
| 2024-03 | 125 | 83 | 0.04228 | 0.04746 | -0.004131 | -0.001071 |
| 2024-09 | 113 | 75 | 0.02969 | 0.03273 | -0.005982 | -0.01834 |
| 2025-03 | 125 | 83 | 0.04378 | 0.04703 | -0.008457 | -0.01246 |
| 2025-09 | 131 | 87 | 0.03866 | 0.04631 | -0.007625 | -0.01321 |
| 2026-03 | 131 | 87 | 0.07852 | 0.08526 | -0.008041 | -0.004022 |
| 2026-09 | 125 | 83 | 0.08066 | 0.08702 | -0.005981 | -0.006046 |

Implied q rises from 0.02969 in 2024-09 to 0.07852 in 2026-03 and 0.08066 in 2026-09, while the matched IF basis rises from 0.03273 to 0.08526 and 0.08702. The q-minus-basis median stays between -0.00846 and -0.00413, so the 2026 rise is common to both series rather than a widening of the gap.

Figures: `results/approach1/io/surface.html` and `results/approach1/io/fit_slices.html`; close versions under `results/approach1/io_close/`.

### 4.3 Discussion and limitations

The settle fit removes all tolerance-level butterfly and calendar violations on the audit grid, but the strict counts (1 butterfly, 12 calendar) show that arbitrage-free holds only at numerical tolerance, not exactly. The printed butterfly formula records 382 raw strict violations instead of 8 and 390 repaired strict violations instead of 1, so the omitted `w''/2` term is material. The close run is worse: 23 raw strict butterfly violations and 3 after repair. The median per-fit median repricing is 0.77% on settle and 1.27% on close, but the median per-fit maximum is 2.58% and 4.33%. The forward check shows implied q about 0.41 to 0.85 percentage points below the IF basis on matched pairs; the mean is more negative than the median, which is consistent with short-dated annualization amplifying small forward differences, but this was not tested as a separate cause.

SVI fits at the eta bounds are common: 98 of 693 settle fits and 118 of 693 close fits sit on a bound. Whether these are genuine optima or optimizer/ill-conditioning effects was not tested. The audit is only on `|k| <= 0.14`; no global arbitrage proof is claimed. The 252 unmatched forwards are explained by the four-contract IF listing and month matching, as stated in Section 3.1.

## 5. Q1 Option B: neural surface

### 5.1 Model

Q1B uses the same OTM quotes as Q1A on six sample dates (2024-03-14, 2024-09-13, 2025-03-14, 2025-09-15, 2026-03-16, 2026-09-15), giving 30 date-expiry fits and 470 quotes. The 2024-03 date is one day before the metadata date 2024-03-15 because the selection rule takes the nearest mid-month date with at least 80% of that month's maximum quote count. The surface is one MLP across all expiries: inputs `k/0.2` and `sqrt(T)/sqrt(T_max)`, four layers (three hidden layers of 24 units), `tanh` activation, and a softplus output plus `1e-6` for `sigma`. The loss is the mean squared undiscounted OTM price per unit forward plus `lambda1 * mean(max(0,-g)^2)` and `lambda2 * mean(max(0,-dw/dT)^2)` on a collocation grid `k in [-0.14,0.14]` crossed with expiry tenors; `g` includes `w''/2`. The whole objective is multiplied by `c = 1 / initial_price_loss`, which leaves the minimizer unchanged. Optimization is 3,000 Adam steps followed by bounded L-BFGS (1,000 iterations, strong Wolfe, 45 s cap). Seeds are derived from date, design and replication but not lambda, so lambda pairs share an initialization.

Selection is pre-declared: complete fit coverage, then zero butterfly and calendar violations at 1e-8 on the Q1A grid, then the lowest pooled in-sample OTM relative repricing median, with ties broken by the smallest `lambda1 + lambda2`. The chosen pair is `lambda1=0.01`, `lambda2=0.0`.

### 5.2 Results

Table 8: Q1B 3x3 lambda ablation. Source: `results/approach2/io_scaled/metrics.json`.

| lambda1 | lambda2 | median | p95 | butterfly | calendar | dense butterfly | dense calendar |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.009106 | 0.1316 | 3 | 0 | 37 | 4 |
| 0 | 0.01 | 0.009106 | 0.1316 | 3 | 0 | 37 | 4 |
| 0 | 1 | 0.009106 | 0.1316 | 3 | 0 | 37 | 4 |
| 0.01 | 0 | 0.008687 | 0.1234 | 0 | 0 | 30 | 4 |
| 0.01 | 0.01 | 0.008687 | 0.1234 | 0 | 0 | 30 | 4 |
| 0.01 | 1 | 0.008687 | 0.1234 | 0 | 0 | 30 | 4 |
| 1 | 0 | 0.009898 | 0.1351 | 2 | 0 | 32 | 8 |
| 1 | 0.01 | 0.009898 | 0.1351 | 2 | 0 | 32 | 8 |
| 1 | 1 | 0.009898 | 0.1351 | 2 | 0 | 32 | 8 |

For each `lambda1`, the three `lambda2` values give identical final medians, p95 values and violation counts; `lambda2` changes the maximum calendar penalty during optimization but not the final metrics. The chosen `(0.01,0)` is one of three feasible pairs.

Table 9: Q1B chosen model and SVI comparison on the same 470 OTM quotes. Source: `results/approach2/io_scaled/metrics.json`.

| model | median | p95 | max | butterfly | calendar | dense butterfly | dense calendar |
|---|---|---|---|---|---|---|---|
| chosen neural (0.01,0) | 0.008687 | 0.1234 | 0.983 | 0 | 0 | 30 | 4 |
| unpenalized (0,0) | 0.009106 | 0.1316 | 0.9857 | 3 | 0 | 37 | 4 |
| SVI re-priced | 0.008214 | 0.05453 | 0.1737 | n/a | n/a | n/a | n/a |

The chosen neural surface has 0 butterfly and 0 calendar violations on the Q1A grid (`min g = 0.002068`), but 30 butterfly and 4 calendar violations on the dense grid (`k` from -0.20 to 0.20 step 0.005, listed expiries plus midpoints), with dense minimum `g = -0.5603` and dense minimum forward variance -0.005021. All dense violations are outside the training collocation `|k| <= 0.14`; this is consistent with extrapolation failure, but the cause was not separately tested.

Table 10: Q1B cold-start SVI solver comparison (primary solver result). Source: `results/approach2/io_scaled/metrics.json`.

| solver | reached | final objective median | final objective p95 | relative gap median | iterations median | evals median | wall s median |
|---|---|---|---|---|---|---|---|
| adam | 0/30 | 1.838e-08 | 7.536e-08 | 0.5271 | 5000 | 5001 | 0.6051 |
| lbfgs | 14/30 | 8.003e-09 | 7.464e-08 | 0.006494 | 500 | 644.5 | 0.205 |
| scipy | 30/30 | 6.56e-09 | 7.463e-08 | 1.266e-14 | 46 | 55.5 | 0.007105 |

The cold-start comparison uses a documented cold point `a = 0.5*min(w)`, `b=0.1`, `rho=-0.3`, `m=0`, `eta=0.1`, the same for all solvers, with a strict 0.1% relative tolerance against the best known objective. Adam is one setting only (5,000 iterations, learning rate 0.03, cosine schedule); the result does not support a general claim that Adam is poor. The warm-started comparison in `svi_solver` starts all solvers at approach1's optimum: torch reaches the 0.1% tolerance on 29/30 fits and scipy on 30/30, with torch's median iterations to tolerance equal to 0 because the start is already at the optimum. That comparison is not a convergence test. The cold-start table above is the primary solver result.

Table 11: Q1B penalty activity by date. Source: `results/approach2/io_scaled/metrics.json`.

| date | max butterfly penalty | max calendar penalty | final metrics differ from (0,0) |
|---|---|---|---|
| 2024-03-14 | 0 | 0 | False |
| 2024-09-13 | 0.000427 | 2.243e-06 | False |
| 2025-03-14 | 2.444e-08 | 0 | False |
| 2025-09-15 | 0 | 0 | False |
| 2026-03-16 | 6.733e-05 | 0 | True |
| 2026-09-15 | 0.9899 | 4.553e-08 | True |

Penalties were exactly zero on 2024-03-14 and 2025-09-15, transient on 2024-09-13 and 2025-03-14 with no change in final metrics, and changed final metrics on 2026-03-16 and 2026-09-15. On 2026-03-16, `lambda1=0.01` removes 3 butterfly violations (unpenalized 3, chosen 0) and improves the median from 0.008160 to 0.006407. On 2026-09-15 it keeps zero violations (unpenalized 0, chosen 0) and improves the median from 0.009869 to 0.008687; `lambda1=1` gives 1 violation. `lambda2` does not change final metrics.

Table 12: Q1B random and wing robustness. Source: `results/approach2/io_scaled/metrics.json`.

| design | model | fits | n | median | p95 |
|---|---|---|---|---|---|
| random | svi | 18/18 | 407 | 0.0163 | 0.1261 |
| random | neural | 18/18 | 407 | 0.01836 | 0.2212 |
| wing | svi | 6/6 | 83 | 0.05034 | 0.5592 |
| wing | neural | 6/6 | 83 | 0.09326 | 0.7809 |

Table 13: Q1B per-tenor comparison with SVI. Source: `NOTES_OPEN_ISSUES.md` (diagnostic record, not a metrics file); this table is not regenerated by `reproduce.sh`.

| T bucket | n | neural median | SVI median | neural p95 | SVI p95 | neural max | SVI max |
|---|---|---|---|---|---|---|---|
| <= 0.10 | 147 | 0.01748 | 0.01920 | 0.4965 | 0.1027 | 0.9830 | 0.1737 |
| 0.10-0.30 | 201 | 0.00820 | 0.00698 | 0.05438 | 0.02877 | 0.1346 | 0.05622 |
| 0.30-0.60 | 75 | 0.00501 | 0.00523 | 0.01893 | 0.01616 | 0.03722 | 0.02422 |
| > 0.60 | 47 | 0.00439 | 0.00340 | 0.01452 | 0.02333 | 0.05528 | 0.03982 |

The committed Q1B metrics do not contain the per-tenor breakdown; the table above is the previously recorded diagnostic in `NOTES_OPEN_ISSUES.md`, included because the brief asks for it, and is not a `metrics.json` result.

### 5.3 Discussion and limitations

The neural surface is close to SVI in median repricing but worse in the tail and in the wing holdout. The dense-grid audit means the surface must not be called globally arbitrage-free. The cold-start comparison shows that scipy is the reliable solver in this setting; Adam reaches tolerance on 0/30 cold starts at the single setting tested, and L-BFGS on 14/30. `results/approach2/io/metrics.json` is the optimization failure that motivated the fix: median 4.43%, p95 67.84%, max 98.19%, with no penalties. The diagnosis recorded in `NOTES_OPEN_ISSUES.md` is that L-BFGS stopped at 100 iterations and the loss scale was tiny; the fix was whole-objective scaling by `c = 1/initial_price_loss`, shared initialization independent of lambda, 3,000 Adam steps, 1,000 L-BFGS iterations, and 24 units in three hidden layers. The diagnosis was not independently re-tested for this report.

## 10. Indicative benchmarks

The brief file is held outside the repository. The target column below is quoted from the project brief, not from `AGENTS.md` or a results file. Every number in the reported-result column is from the metrics files named in the caption.

Table 38: Indicative Q1 benchmarks. Source: `results/approach1/io/metrics.json`, `results/approach1/io_close/metrics.json`; target column from the brief/instructions.

| benchmark | target | reported result | met? |
|---|---|---|---|
| Q1 median liquid-strike repricing relative error | < 2% | settle 0.007745, close 0.01269 | met |
| Q1 zero butterfly violations on interior grid after fitting | zero | settle 0 at 1e-8, 1 strict; close 0 at 1e-8, 3 strict | met at 1e-8; not met strict |
| Q1 calibration table across all sample dates | present | Table 6, six sample months | met |

The Q1 median is 0.77% on settle and 1.27% on close, so the 2% target is met on both price fields. The benchmark table contains only the brief's indicative targets.

## 11. Differences from the brief

1. **Butterfly formula.** The printed `g(k)` omits `w''/2`; the code uses the correct Gatheral form for decisions and reports both. On settle, raw strict violations are 8 with `w''/2` and 382 without; repaired strict violations are 1 and 390. On close, raw strict counts are 23 and 418; repaired strict counts are 3 and 423. Source: `results/approach1/io/metrics.json`, `results/approach1/io_close/metrics.json`.
2. **No bid-ask data.** Daily CFFEX data have no bid-ask spread, so volume >= 10 stands in for liquidity/spread. On settle this filter removes 8,008 quotes; the full sequential filter removes 11,297. Source: `results/approach1/io/metrics.json`.
3. **SVI eta bounds.** The current bounds are `[0.01,1.0]`, narrower than the legacy `[1e-6,2.0]`. This changes bound hits but not median RMSE materially. Source: `results/approach1/io/metrics.json`.
4. **Q1B sample date.** Q1B uses 2024-03-14, while the metadata sample date is 2024-03-15. The selection rule chooses the nearest mid-month date with at least 80% of the month's maximum quote count. Source: `results/approach2/io_scaled/metrics.json`.
5. **Individual submission.** This is an individual submission rather than a group submission.
6. **AI-drafted report.** The report was drafted by AI rather than written by a group, with instructor approval as stated by the author in Section 14.
7. **Only IO options.** Only CFFEX IO options are used for the surface; SSE ETF options are not fitted. Source: `data/metadata.json` and `results/approach1/io/metrics.json`.
8. **Parity forward.** `q` is replaced by the put-call parity forward: `q = r - ln(F/S0)/T` with `F` estimated from parity, rather than a dividend model. Source: `results/approach1/io/metrics.json`.
9. **FastboxData not used.** FastboxData was not used as a data source.

## 12. Limitations

- SVI and neural fits are audited only on the reported grids. The neural surface has 30 butterfly and 4 calendar violations on the dense grid outside its training collocation, so it must not be called globally arbitrage-free. SVI is not proved arbitrage-free off the audit grid either.
- Both models are weak in the wings and at short tenors: Q1A settle worst per-fit max repricing is 0.3251 and close is 0.2624; Q1B wing p95 is 0.7809 and max 1.519, versus SVI 0.5592.
- SVI fits frequently sit on the eta bounds (31 lower and 67 upper on settle; 72 lower and 46 upper on close), and the cause is not established.
- The Q1B per-tenor table comes from `NOTES_OPEN_ISSUES.md`, not from a committed `metrics.json`.

## 13. Session integrity

Known writes outside the repository root, from the session record:

- `/tmp/q1_online.log`, `/tmp/q1_online_resume.log`, `/tmp/q1_online_resume2.log`, `/tmp/q1_expiry_fix.log`: logs from the early Q1 data downloads.
- `/tmp/q1a_close.log`: log from the early Q1A close run (still present; not deleted).
- `/tmp/offline_check.log`, `/tmp/offline_check2.log`, `/tmp/uv_sync_final.log`: reproduction and environment checks.
- `~/.cache/uv`: possible writes from early `uv run` calls before `UV_CACHE_DIR` was pinned to the repository.
- `~/.local/share/opencode/tool-output/`: automatic tool-output files written by the harness.

Tool setup and other session facts:

- Before the first working session, the desktop app and CLI conflicted over one database, so the file in `~/.local/share/opencode` was moved by hand to a backup folder. This was tool setup, not a project change.
- An accidental one-word prompt was sent to a free model (session 01); it ran read-only commands and was stopped.
- Free quota ran out on 2026-11-04, so the session continued on OpenCode Go with the same model family.
- During reproduction the agent copied the uv-managed Python into `.cache` with read access to `~/.local/share/uv`, and deleted its own temporary clone under `.cache` without a separate approval.
- Claude also renamed two misnamed session files and read repository files, on the user's computer, without changing any results.

The report step itself wrote only `report.md` and a verification script under `.cache/`; it did not change code, data or results. The working tree already contained a modified `results/comparison.html` on entry; it was not touched.

The split work itself ran with `ap2026-split` as its root; `../ap2026` was read-only, and the split repositories were created afterwards from parent tag `submission`.

## 14. Statement of work

This is an individual submission by Hao Lu. The instructor approved individual completion with AI assistance. opencode with DeepSeek V4.1 Flash wrote the code, ran the experiments and drafted this report. Claude, in a separate chat that is not part of SESSION.md, drafted most of the prompts, proposed many methodology decisions and corrections, and reviewed outputs. Hao Lu directed the work, approved each step, and owns the final submission after reviewing it. The author does not claim to have independently derived the mathematics; the derivations above are a record of the implemented methods and their sources. The author has read this report.

## 15. Causal claims

Table 39: Prose claims about causes and their status. Source: `results/approach1/io/metrics.json`, `results/approach2/io_scaled/metrics.json`, and the report text.

| claim | status |
|---|---|
| Q1B dense-grid violations are caused by extrapolation outside the training collocation | hypothesis; violations are located outside the range, but the cause was not tested |
| Q1A eta-bound fits are caused by a flat or ill-conditioned objective | hypothesis; not tested |
| The negative mean q-minus-IF-basis gap is driven by short-dated annualization | hypothesis; not tested |
| The 252 unmatched IO/IF forwards are caused by far IO expiries having no same-month IF contract | tested in code and raw data |
| The Q1 clean-clone comparison is exact | tested by the checker: 19,184 shared numeric values, zero differences; `results/approach2/io` is historical and excluded |

## 16. Result files

Every number in the tables is from one of the following result files unless the table caption says otherwise: `results/approach1/io/metrics.json`, `results/approach1/io_close/metrics.json`, `results/approach2/io_scaled/metrics.json`. `results/approach2/io/metrics.json` is the historical pre-fix run kept for the record and is not rerun. Metadata and row counts come from `data/metadata.json`. The Q1B per-tenor table is the one exception and is labeled in its caption; it comes from `NOTES_OPEN_ISSUES.md`. The clean-clone totals in Section 2 are from the `code/check_reproduction.py` output, and the session-integrity facts are from the session record.
