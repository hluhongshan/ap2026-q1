# Advanced Programming 2026 Project Report

Hao Lu

## 1. Summary

The repository implements Q1, Q2 and Q3 of the Advanced Programming 2026 brief, options A and B of each. Q1 fits an arbitrage-free implied-volatility surface to CFFEX IO options, with SVI slices (Q1A) and a neural surface (Q1B). Q2 prices path-dependent payoffs under GBM by Monte Carlo (Q2A) and learns a CVaR-optimal hedge (Q2B). Q3 forecasts the PMI contraction regime with a classical probit/PCA model (Q3A) and an LSTM with leakage ablations (Q3B).

The results are mixed and are reported plainly. Q1A fits all 693 liquid date-expiry slices on the settle price and reaches zero butterfly and zero calendar violations at the 1e-8 tolerance on its audit grid, but 1 butterfly and 12 calendar checks remain strictly negative after repair. The median per-fit median repricing error is 0.77% on settle and 1.27% on close; the median per-fit maximum is 2.58% and 4.33%. Q1B is close to SVI in median repricing on the same 470 OTM quotes (0.87% vs 0.82%) but much worse in the tail (p95 12.34% vs 5.45%; maximum 98.30% vs 17.37%), and it violates arbitrage on a denser grid outside its training collocation. Q2A passes the Black-Scholes and closed-form checks; the Asian control-variate variance ratio is 308x to 1974x across four futures, above the 5x benchmark. Q2B's deep policy has lower gross and net CVaR than daily Black-Scholes hedging on the brief P&L at zero and at 50 bp cost. Q3A beats the expanding prior on Brier score but not persistence (AUC 0.6928 vs 0.7390), and its nowcast beats a random-walk-with-drift benchmark in all periods, including ex-COVID. Q3B does not beat persistence either; the retrained LSTM never predicts a probability above 0.5 on the test block and its test AUC is 0.6422 versus 0.7390 for persistence.

What was not achieved: Q1B is not globally arbitrage-free and is not competitive with SVI in the wings; Q2B covers only IF and uses one training seed; Q3A and Q3B do not reach the indicative AUC target of 0.75 or beat persistence. The data are daily, without bid-ask quotes or real-time macro vintages.

## 2. Environment and one-command reproduction

The project uses Python 3.13 under `uv` with pinned dependencies. From the repository root:

```bash
uv sync --python 3.13
source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, all approaches, post-hoc scripts, compare
python code/check_reproduction.py
```

`./reproduce.sh` runs `data/fetch.py --offline`, all tests, approach1 through approach6 including post-hoc scripts, and `compare.py`. The offline check verifies the nine committed clean artifacts against `data/metadata.json` and reports that all checksums match. The test suite ran 33 tests and passed.

The clean-clone check (`python code/check_reproduction.py`) compares `.cache/clean_clone_full/results` with the committed results. The cached clone predates three post-run commits: the `io_close` refresh, the Q3B `origin_date` label fix, and the Q1B cold-start addition. Each was checked separately: the refreshed `io_close` results match 6,972 numeric values exactly; the label fix was checked by parsed-JSON before/after comparison and no metric changed; the cold-start block was checked by two temporary runs with identical numbers excluding timings. The current checker reports 36,982 compared numeric values, all 36,982 exact, with maximum absolute and relative differences 0. It excludes `results/approach2/io/metrics.json` as historical, not rerun. The checker exits `FAIL` only because of key sets: `results/approach2/io_scaled/metrics.json` has 74 numeric keys only in the committed run (the cold-start block), and `results/approach6/macro/metrics.json` has 216 numeric keys only in the reproduced run (`history.refits[*].min_epochs` config echoes). There are zero numeric differences among shared keys.

The sum of committed `fit_seconds` is 4,668.2 s; excluding the historical Q1B run it is 3,869.8 s, about 64.5 minutes.

## 3. Data

### 3.1 Sources, coverage and snapshots

Table 1: Committed clean artifacts. Source: `data/metadata.json`.

| artifact | source | rows | retrieved |
|---|---|---|---|
| `data/clean/options_io.csv` | CFFEX IO daily quotes; CSI 300 close from akshare.stock_zh_index_daily | 30068 | 2026-11-04 |
| `data/clean/shibor.csv` | China Shibor | 2398 | 2026-11-04 |
| `data/clean/futures_IF.csv` | Sina main continuous futures IF0 (IF); interface=akshare.futures_main_sina | 2376 | 2026-11-05 |
| `data/clean/futures_RB.csv` | Sina main continuous futures RB0 (RB); interface=akshare.futures_main_sina | 4274 | 2026-11-05 |
| `data/clean/futures_CU.csv` | Sina main continuous futures CU0 (CU); interface=akshare.futures_main_sina | 5310 | 2026-11-05 |
| `data/clean/futures_SC.csv` | Sina main continuous futures SC0 (SC); interface=akshare.futures_main_sina | 2088 | 2026-11-05 |
| `data/clean/macro_panel.csv` | Q3 monthly macro panel from akshare macro_china interfaces and committed SHIBOR | 202 | 2026-11-05 |
| `data/clean/macro_meta.csv` | Q3 series metadata and conservative availability lags | 34 | 2026-11-05 |
| `data/clean/sina_comparison.csv` | CFFEX close vs Sina option daily close | 30 | 2026-11-04 |

The raw CFFEX snapshot is committed through a `.gitignore` exception: 128 daily files, of which 126 parse. `20240917_1.csv` and `20260925_1.csv` are non-CSV 200 pages with missing columns. `data/raw/q3` raw interface responses are local, gitignored caches; the cleaned panel and metadata are committed. `data/legacy/io_settle_svi_params.csv` and `data/legacy/io_close_svi_params.csv` are committed snapshots used only for the eta-bound comparison; they have 693 fits each.

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

### 3.3 Q3 panel, lags and revisions

The macro panel has 202 month-end observations and 34 series. Publication lags follow `F_t = {x_{i,s}: s + L_i <= t}`. Thirteen series have `L_i = 0` (market rates, yields, PMI and market-based series); twenty-one have `L_i = 1` (CPI, PPI, money, credit, trade, activity, GDP and similar). Most lags are conservative assumptions because the raw interfaces carry no genuine release date; lag zero is definitional only for market rates and yields. Jin10 announcement-date interfaces were saved but mostly stop in 2025 and are not used for the panel, except `macro_china_gdp_yearly` as a cross-check. GDP is derived from cumulative real levels and the current revised vintage: 79 quarters derived, 59 validation quarters, maximum absolute difference 0.83 percentage points and mean absolute difference 0.17 points. The data are the current revised vintage, not real-time vintages.

January and February are missing for: `retail_yoy` (Jan/Feb missing share 0.8824), `industrial_production_yoy` (Jan 1.000, Feb 0.7059), `fai_ytd_yoy` (Jan 1.000, Feb 0.1765), `electricity_yoy` (Jan 0.9412, Feb 0.000) and `industrial_profit_yoy` (Jan 1.000, Feb 0.000). These values are left missing rather than imputed.

Largest missing shares are `gdp_yoy` 0.6683, `lpr_5y` 0.5693, `unemployment` 0.4802, `tsf_12m_yoy` 0.4406 and `shibor_3m` 0.4257. At threshold 50, the full-sample PMI-below share is 0.3069 and the two-month regime share is 0.2079 (42 months); at 49.5 the regime share is 0.0743 (15 months).

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

## 6. Q2 Option A: Monte Carlo pricing

### 6.1 Model and estimators

Under the brief's GBM, `dS_t = r S_t dt + sigma S_t dW_t`, with `S0` the last close, `K=S0`, `T=1`, `m=252`, barrier `B=1.3 S0`, and `r` the SHIBOR zero at `T`. A futures price has zero drift under Q; the brief specifies drift `r`, so the code follows the brief and flags this as a modelling simplification. `sigma` is the standard deviation of daily log returns times `sqrt(252)` on a five-year window. The payoffs are arithmetic Asian `max(mean(S_{t_i})-K,0)`, discretely monitored up-and-out `max(S_T-K,0) * 1{max S_{t_i}<B}`, cash-or-nothing digital `1{S_T>K}`, and European `max(S_T-K,0)`. All are discounted by `exp(-rT)`.

Estimators are naive Monte Carlo, antithetic `(Z,-Z)` at equal path count, a control variate using the discrete geometric Asian closed form with `ln G ~ N(ln S0 + (r-sigma^2/2)T(m+1)/(2m), sigma^2 T (m+1)(2m+1)/(6m^2))`, and control variate plus antithetic. The control-variate beta is estimated on an independent pilot of 100,000 paths so the estimator is unbiased. European and Asian delta/vega use pathwise estimators; digital and barrier use likelihood-ratio estimators because pathwise is invalid at discontinuities. Finite differences use common random numbers and relative bumps `h in {0.1, 0.01, 0.001}` for spot and sigma. European and digital references are closed form; Asian and barrier references are high-precision control-variate estimates (524,288 paths per dataset, 2,097,152 paths in the IF study). The convergence study uses `N = 2^10` to `2^16`, 30 replications and a `2^22` reference.

### 6.2 Results

Table 14: Q2A calibration. Source: `results/approach3/{IF,CU,RB,SC}/metrics.json`.

| dataset | S0 | sigma | sigma MAD | mu mean | mu log-reg | excess kurtosis | max abs log return | tail-return count |
|---|---|---|---|---|---|---|---|---|
| IF | 4071 | 0.1911 | 0.1426 | -0.03642 | 0.01402 | 10.02 | 0.1056 | 106 |
| CU | 1.062e+05 | 0.1705 | 0.133 | 0.08798 | 0.09923 | 5.109 | 0.06849 | 113 |
| RB | 3008 | 0.1986 | 0.1537 | -0.07184 | -0.09079 | 4.062 | 0.08565 | 131 |
| SC | 681.6 | 0.4197 | 0.3082 | 0.06102 | -0.02652 | 4.974 | 0.1492 | 119 |

Table 15: Q2A prices for every payoff and estimator. Cells are price (standard error), all paths 200,000. Source: `results/approach3/{IF,CU,RB,SC}/metrics.json`.

| dataset | payoff | naive | antithetic | control variate | CV + antithetic |
|---|---|---|---|---|---|
| IF | european | 337.3 (1.18) | 337 (0.9064) | 337.5 (0.6674) | 337.4 (0.6226) |
| IF | asian | 192.7 (0.6542) | 192.6 (0.4904) | 192.9 (0.01666) | 192.8 (0.01653) |
| IF | barrier | 141.6 (0.5798) | 141.6 (0.4844) | 141.6 (0.5754) | 141.5 (0.4587) |
| IF | digital | 0.4854 (0.001102) | 0.4849 (0.0001956) | 0.4855 (0.0008975) | 0.4849 (0.0001949) |
| CU | european | 7950 (27.21) | 7952 (20.63) | 7948 (15.34) | 7943 (14.3) |
| CU | asian | 4533 (15.14) | 4539 (11.25) | 4532 (0.3408) | 4532 (0.3424) |
| CU | barrier | 4160 (15.67) | 4158 (12.63) | 4160 (15.26) | 4160 (12.3) |
| CU | digital | 0.493 (0.001102) | 0.4926 (3.037e-05) | 0.493 (0.0008972) | 0.4926 (3.037e-05) |
| RB | european | 258.6 (0.9081) | 258.6 (0.7012) | 258.7 (0.5166) | 258.9 (0.484) |
| RB | asian | 147.6 (0.503) | 147.4 (0.3804) | 147.6 (0.01313) | 147.6 (0.01305) |
| RB | barrier | 100.3 (0.4224) | 99.93 (0.3582) | 100.3 (0.4206) | 99.87 (0.3357) |
| RB | digital | 0.4842 (0.001102) | 0.4826 (0.0002221) | 0.4842 (0.0008988) | 0.4826 (0.0002211) |
| SC | european | 117.7 (0.4879) | 117.6 (0.4104) | 117.5 (0.284) | 117.5 (0.2778) |
| SC | asian | 67.81 (0.2599) | 67.71 (0.2115) | 67.67 (0.01481) | 67.66 (0.01472) |
| SC | barrier | 5.409 (0.05163) | 5.355 (0.04986) | 5.412 (0.05128) | 5.358 (0.04789) |
| SC | digital | 0.4255 (0.001092) | 0.4244 (0.0005388) | 0.4251 (0.0008934) | 0.4244 (0.000522) |

Table 16: Q2A both-engine agreement z-scores. Source: `results/approach3/{IF,CU,RB,SC}/metrics.json`.

| dataset | European | Asian | barrier | digital |
|---|---|---|---|---|
| IF | 0.6486 | 1.099 | 0.7633 | 1.198 |
| CU | -1.477 | -1.619 | -1.344 | -1.545 |
| RB | -0.1439 | -0.1302 | 0.2969 | 0.932 |
| SC | -0.3891 | -0.2016 | 2.252 | 1.064 |

Table 17: Q2A Greeks versus reference. Source: `results/approach3/{IF,CU,RB,SC}/metrics.json`.

| dataset | payoff | delta | delta ref | delta z | vega | vega ref | vega z |
|---|---|---|---|---|---|---|---|
| IF | european | 0.5682 | 0.568 | 0.1834 | 1598 | 1601 | -0.4397 |
| IF | asian | 0.5397 | 0.5403 | -0.4352 | 926.3 | 927.2 | -0.2409 |
| IF | digital | 0.0005046 | 0.0005053 | -0.4079 | -0.3579 | -0.3523 | -0.5279 |
| IF | barrier | 0.09765 | 0.0803 | 1.182 | -927.9 | -879.3 | -0.5756 |
| CU | european | 0.5679 | 0.5675 | 0.2993 | 4.173e+04 | 4.175e+04 | -0.1115 |
| CU | asian | 0.5425 | 0.5409 | 1.246 | 2.416e+04 | 2.414e+04 | 0.2597 |
| CU | digital | 2.173e-05 | 2.172e-05 | 0.07736 | -0.3995 | -0.3922 | -0.6016 |
| CU | barrier | 0.14 | 0.1304 | 0.5533 | -2.342e+04 | -2.198e+04 | -0.5564 |
| RB | european | 0.5701 | 0.5683 | 1.37 | 1182 | 1182 | 0.01267 |
| RB | asian | 0.5411 | 0.5393 | 1.36 | 684.4 | 686.8 | -0.8341 |
| RB | digital | 0.0006589 | 0.0006581 | 0.3762 | -0.3517 | -0.3407 | -1.078 |
| RB | barrier | 0.06839 | 0.07094 | -0.1841 | -744.4 | -618 | -2.153 |
| SC | european | 0.5982 | 0.5965 | 1.016 | 264.2 | 263.9 | 0.1735 |
| SC | asian | 0.5482 | 0.5473 | 0.6066 | 154.2 | 153.7 | 0.6143 |
| SC | digital | 0.001358 | 0.001353 | 0.9338 | -0.228 | -0.2254 | -0.6084 |
| SC | barrier | -0.0139 | -0.01232 | -0.4994 | -29.23 | -29.93 | 0.2311 |

Table 18: Q2A full finite-difference bias and variance. Source: `results/approach3/IF_study/metrics.json`.

| payoff | greek | bump | mean | bias | variance | RMSE | z |
|---|---|---|---|---|---|---|---|
| european | delta | 0.1 | 0.5616 | -0.006336 | 2.938e-05 | 0.008279 | -6.402 |
| european | delta | 0.01 | 0.5679 | -0.0001023 | 4.043e-05 | 0.006252 | -0.08816 |
| european | delta | 0.001 | 0.5681 | 8.182e-05 | 3.974e-05 | 0.006199 | 0.07109 |
| asian | delta | 0.1 | 0.5308 | -0.008875 | 2.215e-05 | 0.01001 | -9.87 |
| asian | delta | 0.01 | 0.5391 | -0.0005191 | 3.302e-05 | 0.005674 | -0.4797 |
| asian | delta | 0.001 | 0.5391 | -0.0005174 | 3.534e-05 | 0.005868 | -0.4631 |
| barrier | delta | 0.1 | 0.0744 | -0.0104 | 2.998e-05 | 0.01171 | -3.354 |
| barrier | delta | 0.01 | 0.09012 | 0.005317 | 0.0004428 | 0.02136 | 1.1 |
| barrier | delta | 0.001 | 0.08003 | -0.00478 | 0.005104 | 0.0704 | -0.3575 |
| digital | delta | 0.1 | 0.0004832 | -2.211e-05 | 5.21e-11 | 2.322e-05 | -16.78 |
| digital | delta | 0.01 | 0.000506 | 6.842e-07 | 1.326e-09 | 3.58e-05 | 0.1029 |
| digital | delta | 0.001 | 0.000527 | 2.172e-05 | 6.284e-09 | 8.091e-05 | 1.5 |
| european | vega | 0.1 | 1605 | 4.046 | 934.2 | 30.32 | 0.725 |
| european | vega | 0.01 | 1605 | 4.053 | 931 | 30.27 | 0.7275 |
| european | vega | 0.001 | 1605 | 4.047 | 931.7 | 30.28 | 0.7262 |
| asian | vega | 0.1 | 928.4 | 2.246 | 291.3 | 16.93 | 0.6989 |
| asian | vega | 0.01 | 928.5 | 2.267 | 291.1 | 16.93 | 0.7057 |
| asian | vega | 0.001 | 928.5 | 2.277 | 291.2 | 16.93 | 0.7087 |
| barrier | vega | 0.1 | -875.9 | -4.254 | 3689 | 59.87 | -0.211 |
| barrier | vega | 0.01 | -882.5 | -10.9 | 2.978e+04 | 170 | -0.3051 |
| barrier | vega | 0.001 | -810.4 | 61.26 | 4.605e+05 | 670 | 0.4899 |
| digital | vega | 0.1 | -0.3626 | -0.01027 | 0.001128 | 0.03458 | -1.675 |
| digital | vega | 0.01 | -0.3736 | -0.02129 | 0.01481 | 0.1215 | -0.9579 |
| digital | vega | 0.001 | -0.3883 | -0.03598 | 0.1004 | 0.3135 | -0.622 |

At `h=0.1` all four delta estimators fail the z-test: European -6.40, Asian -9.87, barrier -3.35 and digital -16.78. At `h=0.01` and `h=0.001` all delta z-scores are within plus or minus two. Vega z-scores are noisier for the barrier and digital, as expected for discontinuous payoffs.

Table 19: Q2A convergence slopes. Source: `results/approach3/IF_study/metrics.json`.

| payoff | naive | antithetic | control variate | CV + antithetic |
|---|---|---|---|---|
| european | -0.4628 | -0.5224 | -0.4948 | -0.5249 |
| asian | -0.4745 | -0.5164 | -0.5079 | -0.4601 |
| barrier | -0.494 | -0.5198 | -0.4936 | -0.5064 |
| digital | -0.4839 | -0.5361 | -0.4718 | -0.5368 |

The complete pricing slope range is -0.4601 to -0.5368. Greek convergence slopes are reported separately in Table 19b; they range from -0.4427 to -0.4959.

Table 19b: Q2A Greek convergence slopes. Source: `results/approach3/IF_study/metrics.json`.

| payoff | delta slope | vega slope |
|---|---|---|
| european | -0.4762 | -0.4644 |
| asian | -0.4668 | -0.4908 |
| barrier | -0.4427 | -0.4959 |
| digital | -0.4643 | -0.4703 |

Table 20: Q2A timing. Source: `results/approach3/IF_study/metrics.json`.

| paths | monitoring dates | loop seconds | vectorized seconds | speedup | loop paths/s | vectorized paths/s |
|---|---|---|---|---|---|---|
| 2,000 | 252 | 0.09895 | 0.003752 | 26.37 | 2.021e+04 | 5.33e+05 |

The line profile excerpt (source: `results/approach3/line_profile.txt`) shows where the loop spends its time:

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
   414    126500   32700000.0    258.5     17.9          for j in range(m):
   415    126000   49282000.0    391.1     26.9              log_s += drift + vol * z[i, j]
   416    126000   33841000.0    268.6     18.5              value = math.exp(log_s)
   417    126000   32671000.0    259.3     17.8              sum_s += value
   418    126000   30840000.0    244.8     16.8              if value > max_s:
```

The digital antithetic ratio is strongly dataset-dependent. Table 21 reports the drift `r`, volatility `sigma`, the derived quantity `r - sigma^2/2` and the ratio. Source: `results/approach3/{IF,CU,RB,SC}/metrics.json`.

| dataset | r | sigma | r - sigma^2/2 (derived) | digital antithetic ratio |
|---|---|---|---|---|
| IF | 0.01446 | 0.1911 | -0.003795 | 31.73 |
| CU | 0.01446 | 0.1705 | -7.22e-05 | 1316 |
| RB | 0.01446 | 0.1986 | -0.005251 | 24.61 |
| SC | 0.01446 | 0.4197 | -0.07363 | 4.105 |

Hypothesis, supported by the ordering but not proved: for an ATM digital, when `r - sigma^2/2` is near zero, a `Z, -Z` pair almost always has exactly one in-the-money leg, so the pair mean is nearly constant. CU has the value closest to zero and the largest ratio; SC has the most negative value and the smallest ratio. This is an explanation of the ordering, not a tested theorem.

### 6.3 Discussion and limitations

The Asian control variate is very effective because the geometric Asian is strongly correlated with the arithmetic payoff. The barrier control variate is weak: control-variate ratios are 1.008 to 1.055, while CV+antithetic ratios are 1.162 to 1.624. Finite differences are noisy for discontinuous payoffs: barrier vega RMSE grows from 59.87 at `h=0.1` to 670.0 at `h=0.001`, so smaller bumps do not help. The roll-jump counts in Table 14 are tail-return counts above the reported threshold, not identified roll dates; the main-contract series carry no roll flags. The calibration uses a five-year window and is sensitive to the large returns retained in the sample. The GBM drift is `r` as specified, not the zero drift of a futures price under Q.

## 7. Q2 Option B: deep hedging

### 7.1 Model

The hedge is a European ATM call on IF with `S0=100`, `K=100`, `T=0.25`, `m=63`, `sigma=0.1911` (the Q2A five-year IF calibration) and `r=0.01417` (SHIBOR zero at `T`). The policy is a small MLP with two hidden layers of 32 units mapping `(S_t/S0, t/T)` to a delta in `[0,1]` through a sigmoid; it does not see the previous position. The objective is empirical CVaR at 0.95 of `-V_T`, the mean of the worst 5% of losses per batch. Training uses `2^17` paths, batches of `2^12`, 100 epochs, Adam at learning rate 0.005, and one policy per cost `kappa in {0,10,50}` bp. The same network initialization is used for every kappa. Validation uses three independent seeds with `2^17` paths each; training uses one seed (42).

The brief P&L is `V_T = sum_t exp(r(T-t)) D_t (S_{t+1}-S_t) - kappa sum_t |D_t-D_{t-1}| S_t - payoff`, with `D_{-1}=0` and no terminal unwind. The self-financing P&L is `V_T = sum_t exp(r(T-t-dt)) D_t (S_{t+1}-exp(r dt) S_t) - kappa sum_t exp(r(T-t)) |D_t-D_{t-1}| S_t - payoff`. The brief formula omits the financing cost of the stock position. Training uses the brief P&L; evaluation reports both. Net CVaR is CVaR minus the future value of the Black-Scholes premium. Benchmarks are Black-Scholes deltas rebalanced every 1, 5 and 21 days at the same kappas, on the same validation paths.

### 7.2 Results

Table 22: Q2B full results for every strategy, kappa and P&L definition. Source: `results/approach4/IF/metrics.json`.

| method | rebalance d | policy kappa bp | eval kappa bp | P&L | CVaR | net CVaR | variance | E[P&L] | turnover |
|---|---|---|---|---|---|---|---|---|---|
| BS | 1 | n/a | 0 | brief | 4.781 | 0.7838 | 0.1815 | -3.808 | 3.04 |
| BS | 1 | n/a | 0 | selffin | 4.961 | 0.9634 | 0.1733 | -3.998 | 3.04 |
| BS | 1 | n/a | 10 | brief | 5.198 | 1.201 | 0.199 | -4.112 | 3.04 |
| BS | 1 | n/a | 10 | selffin | 5.381 | 1.383 | 0.1904 | -4.302 | 3.04 |
| BS | 1 | n/a | 50 | brief | 6.971 | 2.974 | 0.4371 | -5.328 | 3.04 |
| BS | 1 | n/a | 50 | selffin | 7.16 | 3.163 | 0.4273 | -5.521 | 3.04 |
| BS | 5 | n/a | 0 | brief | 5.883 | 1.885 | 0.7867 | -3.807 | 1.645 |
| BS | 5 | n/a | 0 | selffin | 6.069 | 2.072 | 0.7814 | -3.997 | 1.645 |
| BS | 5 | n/a | 10 | brief | 6.101 | 2.104 | 0.8066 | -3.971 | 1.645 |
| BS | 5 | n/a | 10 | selffin | 6.289 | 2.291 | 0.8012 | -4.162 | 1.645 |
| BS | 5 | n/a | 50 | brief | 6.998 | 3.001 | 0.9166 | -4.629 | 1.645 |
| BS | 5 | n/a | 50 | selffin | 7.188 | 3.191 | 0.9111 | -4.821 | 1.645 |
| BS | 21 | n/a | 0 | brief | 8.101 | 4.104 | 3.112 | -3.805 | 0.9313 |
| BS | 21 | n/a | 0 | selffin | 8.296 | 4.299 | 3.114 | -3.995 | 0.9313 |
| BS | 21 | n/a | 10 | brief | 8.211 | 4.214 | 3.145 | -3.898 | 0.9313 |
| BS | 21 | n/a | 10 | selffin | 8.407 | 4.41 | 3.146 | -4.088 | 0.9313 |
| BS | 21 | n/a | 50 | brief | 8.655 | 4.658 | 3.28 | -4.271 | 0.9313 |
| BS | 21 | n/a | 50 | selffin | 8.853 | 4.856 | 3.282 | -4.462 | 0.9313 |
| deep | n/a | 0 | 0 | brief | 4.679 | 0.6817 | 0.2323 | -3.809 | 2.848 |
| deep | n/a | 0 | 0 | selffin | 4.888 | 0.891 | 0.237 | -3.997 | 2.848 |
| deep | n/a | 10 | 10 | brief | 4.987 | 0.9894 | 0.25 | -4.085 | 2.77 |
| deep | n/a | 10 | 10 | selffin | 5.176 | 1.179 | 0.2492 | -4.275 | 2.77 |
| deep | n/a | 50 | 50 | brief | 6.083 | 2.085 | 0.3018 | -5.072 | 2.523 |
| deep | n/a | 50 | 50 | selffin | 6.331 | 2.334 | 0.3178 | -5.261 | 2.523 |

Table 23: Q2B 3x3 cross-evaluation. Source: `results/approach4/IF/metrics.json`.

| policy kappa bp | eval kappa bp | CVaR brief | net CVaR brief | CVaR selffin | net CVaR selffin | turnover |
|---|---|---|---|---|---|---|
| 0 | 0 | 4.679 | 0.6817 | 4.888 | 0.891 | 2.848 |
| 0 | 10 | 5.008 | 1.011 | 5.209 | 1.211 | 2.848 |
| 0 | 50 | 6.533 | 2.536 | 6.724 | 2.726 | 2.848 |
| 10 | 0 | 4.717 | 0.7195 | 4.906 | 0.9084 | 2.77 |
| 10 | 10 | 4.987 | 0.9894 | 5.176 | 1.179 | 2.77 |
| 10 | 50 | 6.379 | 2.382 | 6.568 | 2.571 | 2.77 |
| 50 | 0 | 5.06 | 1.063 | 5.31 | 1.313 | 2.523 |
| 50 | 10 | 5.229 | 1.231 | 5.484 | 1.486 | 2.523 |
| 50 | 50 | 6.083 | 2.085 | 6.331 | 2.334 | 2.523 |

Table 24: Q2B deep policy versus daily Black-Scholes on the brief P&L. Source: `results/approach4/IF/metrics.json`.

| kappa bp | BS daily gross CVaR | deep gross CVaR | BS daily net CVaR | deep net CVaR |
|---|---|---|---|---|
| 0 | 4.781 | 4.679 | 0.7838 | 0.6817 |
| 50 | 6.971 | 6.083 | 2.974 | 2.085 |

On the brief P&L, the deep policy lowers gross CVaR by 0.10 at kappa 0 and by 0.89 at kappa 50; net CVaR falls by the same amounts because the premium is common. The deep policy has higher variance than daily BS at kappa 0 (0.2323 vs 0.1815 brief; 0.2370 vs 0.1733 self-financing) and lower variance at kappa 50 (0.3018 vs 0.4371 brief; 0.3178 vs 0.4273 self-financing). A plausible cause of the CVaR gain is that the policy optimizes CVaR directly rather than following a delta rule, but the experiment does not isolate optimization from the state-dependent delta; that is a hypothesis.

The brief-minus-self-financing CVaR gap is negative everywhere: -0.1796 for BS daily at kappa 0, -0.2094 for the deep policy at kappa 0, and -0.2486 for the deep policy at kappa 50; the mean across all cells is -0.1960. The negative sign means the brief formula reports lower CVaR than the self-financing formula. Training CVaR falls from 11.01 to 4.70 (kappa 0), 11.08 to 4.98 (kappa 10) and 11.32 to 6.13 (kappa 50).

Figures: `results/approach4/IF/training.html` and `results/approach4/IF/policy.html`.

### 7.3 Discussion and limitations

The deep hedge improves CVaR over daily BS on the brief basis at all three costs, with lower turnover at higher cost. The gap is systematic and is larger for the deep policy at high cost. A plausible explanation, not tested here, is that the omitted stock financing scales with the position and the position is held through the path. The training used one seed only; three validation seeds reduce evaluation noise but do not establish training robustness. The study covers only IF, uses no terminal unwind cost, and the policy cannot represent a no-trade region because the previous position is not an input.

## 8. Q3 Option A: classical probit

### 8.1 Model and protocol

The information set is implemented by shifting each series forward by its lag `L_i`: row `t` contains only values public at the end of month `t`. The label is `d_t = 1{PMI_t < 50 and PMI_{t-1} < 50}`. The model is `P(d_{t+1}=1) = Phi(beta' z_t)`, where `z_t` contains an intercept, the first `r` principal components of the standardized aligned factor panel (excluding `gdp_yoy`, `cgb10y`, `spread_10y_2y`, `shibor_3m`) and the controls `cgb10y` and `spread_10y_2y`. PCA and standardization use only rows public at the origin; missing values become zero after standardization; series with under 50% training-window coverage are dropped. The probit is fitted by maximum likelihood with a ridge of `1e-4` and a Newton/BFGS fallback. Validation is expanding walk-forward, refit monthly, minimum 60 training months, a 36-month tuning block and a test block of 105 origins from 2018-01-31 to 2026-09-30. The PCA rank is chosen on the tuning block by cross-validated nowcast RMSE at month 3 of the quarter.

The nowcast is `y_q = b0 + b' fbar_q`, where `fbar_q` averages the aligned factors over the months of the quarter elapsed by `t`; training quarters are those with GDP public by `t`. The benchmark is a random walk with drift estimated from the same information set. Clustering uses rolling 36-month correlations, `d_ij = sqrt(2(1-rho_ij))`, Ward linkage, `k` chosen by silhouette, adjacent-window adjusted Rand indices, and lead-lag correlations with GDP.

### 8.2 Results

Table 25: Q3A test-block classification (105 origins, 34 positives, class share 0.3238). Source: `results/approach5/macro/metrics.json`.

| model | Brier | AUC | accuracy |
|---|---|---|---|
| probit | 0.2396 | 0.6928 | 0.6381 |
| prior | 0.2482 | 0.604 | 0.6762 |
| persistence | 0.2286 | 0.739 | 0.7714 |

The chosen PCA rank is 6, with tuning CV RMSE 0.3565 over 12 quarterly observations; rank 5 is 0.3838, rank 4 is 0.4313, rank 3 is 0.6175, rank 2 is 0.8006, rank 1 is 0.6774, rank 7 is 0.4564 and rank 8 is 0.5271. The first principal component explains 0.3098 of variance. No slope coefficient is significant at 5% with Newey-West standard errors; the largest are PC1 (`z=-1.892`) and `spread_10y_2y` (`z=-1.678`). The only strongly significant term is the intercept (`z=-8.180`).

Table 26: Q3A label sensitivity, including the prior. Source: `results/approach5/macro/metrics.json`.

| specification | probit Brier | probit AUC | probit acc | prior Brier | prior AUC | prior acc | persistence Brier | persistence AUC | persistence acc |
|---|---|---|---|---|---|---|---|---|---|
| threshold 50, two-month (main) | 0.2396 | 0.6928 | 0.6381 | 0.2482 | 0.604 | 0.6762 | 0.2286 | 0.739 | 0.7714 |
| threshold 49.5, two-month | 0.1715 | 0.7159 | 0.7714 | 0.1235 | 0.6052 | 0.8667 | 0.1714 | 0.6291 | 0.8286 |
| threshold 50, one-month | 0.2468 | 0.6625 | 0.6571 | 0.3033 | 0.6441 | 0.5333 | 0.2762 | 0.7219 | 0.7238 |

The probit loses to the prior on accuracy at the main threshold (0.6381 vs 0.6762) and at 49.5 (0.7714 vs 0.8667), and on Brier at 49.5 (0.1715 vs 0.1235). It wins on Brier and AUC at the main threshold against the prior, and on all three one-month metrics.

Table 27: Q3A SHIBOR sensitivity. Source: `results/approach5/macro/metrics.json`.

| model | Brier | AUC | accuracy |
|---|---|---|---|
| probit | 0.2408 | 0.6912 | 0.6381 |
| prior | 0.2482 | 0.604 | 0.6762 |
| persistence | 0.2286 | 0.739 | 0.7714 |

Table 28: Q3A nowcast RMSE. Source: `results/approach5/macro/metrics.json`.

| sample | n | month 1 | month 2 | month 3 | RW drift |
|---|---|---|---|---|---|
| all quarters | 35 | 3.626 | 3.411 | 3.314 | 4.163 |
| ex-COVID | 27 | 0.7095 | 0.8065 | 0.8784 | 1.258 |

The bridge beats the random walk in all periods, including ex-COVID: 0.71 to 0.88 versus 1.26. Ex-COVID the month-3 RMSE is higher than month-1, but the paired difference 0.1689 has interval `[-0.1071, 0.4390]`, so the degradation is not statistically resolved. The advantage over the random walk is not concentrated in COVID.

Table 29: Q3A full-sample descriptive probit coefficients. Source: `results/approach5/macro/metrics.json`.

| term | estimate | information SE | information z | Newey-West SE | Newey-West z |
|---|---|---|---|---|---|
| intercept | -1.547 | 0.2216 | -6.98 | 0.1891 | -8.18 |
| PC1 | -0.3636 | 0.1897 | -1.917 | 0.1922 | -1.892 |
| PC2 | 0.3264 | 0.1761 | 1.853 | 0.2125 | 1.536 |
| PC3 | 0.1725 | 0.1178 | 1.464 | 0.1334 | 1.293 |
| PC4 | -0.33 | 0.16 | -2.063 | 0.2189 | -1.507 |
| PC5 | -0.2055 | 0.137 | -1.5 | 0.1325 | -1.551 |
| PC6 | -0.1271 | 0.1263 | -1.006 | 0.1349 | -0.9424 |
| cgb10y | -0.5927 | 0.597 | -0.9929 | 0.7123 | -0.8321 |
| spread_10y_2y | -0.3375 | 0.2208 | -1.529 | 0.2011 | -1.678 |

Table 30: Q3A latest cluster membership (2023-01-31 to 2025-12-31, k=9). Source: `results/approach5/macro/metrics.json`.

| cluster | members |
|---|---|
| 1 | cgb10y, shibor_3m, lpr_1y, lpr_5y, cgb2y, cgb5y, cgb30y |
| 2 | pmi_non_man, m2_yoy, retail_yoy, real_estate_climate, fiscal_revenue_yoy |
| 3 | pmi, consumer_confidence, unemployment |
| 4 | new_credit_yoy |
| 5 | m1_yoy, tsf_12m_yoy, stock_mktcap_yoy, margin_balance_yoy |
| 6 | ppi_yoy, commodity_price_yoy, commodity_spot_yoy, industrial_profit_yoy |
| 7 | industrial_production_yoy, fai_ytd_yoy, exports_yoy, imports_yoy |
| 8 | fx_reserves_yoy, central_bank_assets_yoy |
| 9 | spread_10y_2y, cpi_yoy, electricity_yoy |

Table 31: Q3A adjacent-window ARI. Source: `results/approach5/macro/metrics.json`.

| left window | right window | common series | ARI |
|---|---|---|---|
| 2012-12-31 | 2013-12-31 | 26 | 0.09801 |
| 2013-12-31 | 2014-12-31 | 26 | 0.2098 |
| 2014-12-31 | 2015-12-31 | 28 | 0.08733 |
| 2015-12-31 | 2016-12-31 | 29 | 0.3273 |
| 2016-12-31 | 2017-12-31 | 29 | 0.3217 |
| 2017-12-31 | 2018-12-31 | 29 | 0.2424 |
| 2018-12-31 | 2019-12-31 | 31 | 0.0965 |
| 2019-12-31 | 2020-12-31 | 32 | 0.3533 |
| 2020-12-31 | 2021-12-31 | 32 | 0.09256 |
| 2021-12-31 | 2022-12-31 | 33 | 0.1148 |
| 2022-12-31 | 2023-12-31 | 33 | 0.5456 |
| 2023-12-31 | 2024-12-31 | 33 | 0.03484 |
| 2024-12-31 | 2025-12-31 | 33 | 0.1648 |

Table 32: Q3A lead-lag summary with GDP, latest window. Source: `results/approach5/macro/metrics.json`.

The lag sign is defined as `corr(cluster_t, GDP_{t+lag})`; a positive lag means the cluster leads GDP. Source: `results/approach5/macro/metrics.json`.

| cluster | largest absolute correlation | lag | n |
|---|---|---|---|
| 1 | 0.4131 | -1 | 67 |
| 2 | 0.7515 | 0 | 67 |
| 3 | 0.3671 | 3 | 64 |
| 4 | 0.2827 | 3 | 64 |
| 5 | 0.4092 | 0 | 67 |
| 6 | 0.4143 | -1 | 67 |
| 7 | 0.7771 | 0 | 67 |
| 8 | 0.4056 | 0 | 67 |
| 9 | 0.5491 | -3 | 65 |

Clustering has 14 rolling windows; the latest window selects `k=9` with silhouette 0.2675. Adjacent-window ARI has minimum 0.03484, median 0.1648 and maximum 0.5456, so cluster membership is unstable across windows.

Table 33: Q3A post-hoc probit robustness (test block already seen; not pre-registered). Source: `results/approach5/macro/metrics.json`.

| model | Brier | AUC | accuracy | share outside [0.01,0.99] |
|---|---|---|---|---|
| original_probit | 0.2396 | 0.6928 | 0.6381 | 0.2762 |
| ridge_selected_probit | 0.2183 | 0.6881 | 0.6381 | 0.181 |
| ridge_selected_probit_with_d_t | 0.2001 | 0.739 | 0.7048 | 0.181 |
| prior | 0.2482 | 0.604 | 0.6762 | 0 |
| persistence | 0.2286 | 0.739 | 0.7714 | 1 |

The post-hoc `d_t` extension improves Brier to 0.2001 and matches persistence's AUC exactly (0.7390), but its paired AUC difference is 0.0 with interval `[-0.1316, 0.0713]`, so there is no evidence that it beats persistence. The original probit's AUC difference versus persistence is -0.04619 with interval `[-0.2375, 0.03518]`.

Figures: `results/approach5/macro/regime.html`, `nowcast_revision.html`, `dendrogram.html` and `stability.html`.

### 8.3 Discussion and limitations

The probit improves on the prior in Brier and AUC at the main threshold, but not in accuracy, and no significance test against the prior was run; persistence is stronger because the two-month label is highly persistent and PMI itself is in the information set. The AUC is 0.6928, below the 0.75 indicative target. The rank is chosen on only 12 quarterly nowcast observations, so the rank choice is noisy. The test block is 105 months but only 35 quarters are available for the nowcast, and the AUC bootstrap intervals are wide. The nowcast beats the random walk in all periods, including ex-COVID; the month-3 ex-COVID RMSE being higher than month-1 is unexplained and is not attributed to a cause. Clustering is descriptive and unstable, and it is not used for forecasting.

## 9. Q3 Option B: LSTM and leakage

### 9.1 Model and protocol

Q3B uses 12-month windows of the standardized aligned panel, excluding `gdp_yoy`, as input to an LSTM with at most two layers and a linear logit head. Training pairs are `(window ending s, d_{s+1})` with `s+1 <= origin t`. The loss is `BCEWithLogits` with `pos_weight = n_neg/n_pos`, and probabilities are corrected by `p_cal = sigmoid(logit - log(pos_weight))`. The walk-forward scheme is expanding, with the LSTM refit every six months while the Q3A probit refits monthly. Hyperparameters are selected on the 36-month tuning block only by mean Brier over three seeds.

The original run used 18-month validation blocks. The retrained model, which is the primary Q3B result, uses the chosen setting hidden 8, one layer, dropout 0.2 and weight decay `1e-2`, an expanding split with the last 36 training-pair months held out for validation, minimum 30 epochs, patience 30 and maximum 300 epochs, with three seeds. Leakage arms are the disciplined aligned panel, a zero-lag panel (`L_i=0`), a zero-lag panel with full-sample standardization, and a zero-lag probit. Leakage premiums use a circular 12-month block bootstrap with 5,000 replicates; positive means the leaked arm looks better (lower Brier or higher AUC).

### 9.2 Results

Table 34: Q3B primary comparison, retrained LSTM (test block, 105 origins, 34 positives). Source: `results/approach6/macro/metrics.json`.

| arm | Brier | AUC | accuracy |
|---|---|---|---|
| probit | 0.2396 | 0.6928 | 0.6381 |
| lstm_disciplined | 0.2561 | 0.6422 | 0.6762 |
| lstm_zero_lag | 0.2566 | 0.6315 | 0.6762 |
| lstm_zero_lag_full_sample | 0.2522 | 0.6138 | 0.6762 |
| prior | 0.2482 | 0.604 | 0.6762 |
| persistence | 0.2286 | 0.739 | 0.7714 |

The retrained disciplined LSTM has Brier 0.2561, AUC 0.6422 and accuracy 0.6762; it does not beat persistence (0.2286, 0.7390, 0.7714) or the probit (0.2396, 0.6928, 0.6381) on Brier or AUC. Its accuracy equals the all-negative rate because it never predicts a probability above 0.5 on the test block. In `results/approach6/macro/post_hoc_predictions.csv`, the disciplined arm has 0 of 105 predictions above 0.5 for each seed; the maxima are 0.1767 (seed 42), 0.3172 (seed 43) and 0.2234 (seed 44).

Table 35: Q3B retrained leakage premiums (positive means leaked arm better). Source: `results/approach6/macro/metrics.json`.

| comparison | Brier premium | Brier 95% CI | AUC premium | AUC 95% CI |
|---|---|---|---|---|
| LSTM zero lag vs disciplined | -0.0008503 | [-0.003906, 0.001532] | 0.004557 | [-0.06256, 0.0745] |
| LSTM full-sample standardization vs disciplined | 0.00368 | [0.0001998, 0.007341] | -0.009528 | [-0.07817, 0.06286] |
| LSTM standardization only | 0.00453 | [-0.0005247, 0.01053] | -0.01408 | [-0.09826, 0.06635] |
| probit zero lag vs probit | 0.006345 | [-0.01956, 0.03238] | 0.001864 | [-0.04367, 0.05129] |

After retraining, the Brier premium is -0.00085 for zero-lag alignment (CI includes zero), 0.00368 for full-sample standardization (CI excludes zero) and 0.00453 for standardization only (CI includes zero); the AUC premiums are 0.00456, -0.00953 and -0.01408, all with intervals including zero. The disciplined and leaked arms are therefore mostly indistinguishable. The larger original premiums are attributed to the original 18-month validation design, which was unreliable: 4 of 24 refits had a single-class validation block, and the tuning AUC was far below the test AUC. The original LSTM is shown as a failed run, not as the primary result.

Table 36: Q3B original failed run and tuning/test gap. Source: `results/approach6/macro/metrics.json`.

| arm | tuning AUC | test AUC | test Brier | test accuracy |
|---|---|---|---|---|
| original LSTM disciplined | 0.2677 | 0.4925 | 0.2604 | 0.6762 |
| retrained LSTM disciplined | 0.9146 | 0.6422 | 0.2561 | 0.6762 |

The original tuning AUC is the chosen setting's seed-mean from `history.tuning` (setting 3); the retrained tuning AUC is the chosen setting's seed-mean from `post_hoc.retrain.tuning_table` (setting 3). The original validation design produced 4 single-class blocks out of 24 refits; the retrained scheme uses 36 validation months and a minimum of 30 epochs.

Table 37: Q3B latent bridge and null. Source: `results/approach6/macro/metrics.json`.

| quantity | value |
|---|---|
| latest cluster window | 2023-01-31 to 2025-12-31 |
| k | 9 |
| indicators | 33 |
| ARI | 0.5219 |
| NMI | 0.7354 |
| null ARI mean / p95 | 0.001082 / 0.09255 |
| null ARI p-value | 0.000999 |
| null NMI mean / p95 | 0.4389 / 0.5093 |
| null NMI p-value | 0.000999 |

The bridge maps each indicator to a correlation vector with the 16 hidden dimensions over 2023-01-31 to 2025-12-31, clusters those vectors with `k=9`, and compares with Q3A's latest indicator labels. Because `h_t` is computed from those indicators, the ARI significance may be partly mechanical; the null permutes indicator labels against fixed bridge clusters, not the full construction. The latent analysis is descriptive.

Figures: `results/approach6/macro/probabilities.html` and `latent.html`.

### 9.3 Discussion and limitations

The original LSTM is a failed run: test AUC 0.4925 is essentially random, and its accuracy is the all-negative rate. The diagnostics show that the training loop can fit (shifted-label train AUC 0.9962; overfit test train AUC 0.9895), so the failure is not a broken loss. The retrained model improves test AUC to 0.6422 but still does not beat persistence or the probit on Brier score. The leakage premiums after retraining are small and mostly include zero, so the disciplined and leaked arms are mostly indistinguishable. Because NBS PMI has `L_i=0` in both arms and the label is built from PMI, the measured lag leakage premium is structurally limited. The final LSTM is trained through 2026-09, so it embeds test months in-sample; the latent analysis is descriptive only.

## 10. Indicative benchmarks

The brief file is held outside the repository. The target column below is quoted from the project brief, not from `AGENTS.md` or a results file. Every number in the reported-result column is from the metrics files named in the caption.

Table 38: Indicative benchmarks. Source: `results/approach1/io/metrics.json`, `results/approach1/io_close/metrics.json`, `results/approach3/{IF,CU,RB,SC}/metrics.json`, `results/approach3/IF_study/metrics.json`, `results/approach5/macro/metrics.json`, `results/approach6/macro/metrics.json`; target column from the brief/instructions.

| benchmark | target | reported result | met? |
|---|---|---|---|
| Q1 median liquid-strike repricing relative error | < 2% | settle 0.007745, close 0.01269 | met |
| Q1 zero butterfly violations on interior grid after fitting | zero | settle 0 at 1e-8, 1 strict; close 0 at 1e-8, 3 strict | met at 1e-8; not met strict |
| Q1 calibration table across all sample dates | present | Table 6, six sample months | met |
| Q2 Asian naive-vs-control-variate variance reduction | factor >= 5 | 308.3x to 1974x (CV), 312.0x to 1955x (CV+anti) | met |
| Q2 pathwise/FD delta agreement | within reported standard errors | h=0.01 and 0.001 pass; h=0.1 fails for European, Asian, barrier and digital (digital z=-16.78) | partly met |
| Q3 regime AUC, classical walk-forward | >= 0.75 | 0.6928 | not met |
| Q3 nowcast error vs random walk with drift | lower than benchmark | bridge 0.7095 to 0.8784 vs 1.258 ex-COVID | met |
| Q3 leakage ablation quantified | quantified | Brier and AUC premiums with 95% intervals in Table 35 | met |

The Q1 median is 0.77% on settle and 1.27% on close, so the 2% target is met on both price fields. The benchmark table contains only the brief's indicative targets. Other checks in this report, such as the kappa=0 deep-hedge comparison or the cluster ARI, are my additions and are not brief targets.

## 11. Differences from the brief

1. **Butterfly formula.** The printed `g(k)` omits `w''/2`; the code uses the correct Gatheral form for decisions and reports both. On settle, raw strict violations are 8 with `w''/2` and 382 without; repaired strict violations are 1 and 390. On close, raw strict counts are 23 and 418; repaired strict counts are 3 and 423. Source: `results/approach1/io/metrics.json`, `results/approach1/io_close/metrics.json`.
2. **Hedging P&L.** The printed brief P&L omits the financing cost of the stock position, so it is not exactly self-financing. The brief-minus-self-financing CVaR gap is -0.1796 for daily BS at kappa 0 and -0.2094 for the deep policy at kappa 0; the mean across all cells is -0.1960. The negative sign means the brief formula reports lower CVaR than the self-financing formula. Source: `results/approach4/IF/metrics.json`.
3. **GBM drift.** The brief specifies drift `r`; the code follows the brief. A futures price has zero drift under Q, so this is a modelling simplification, recorded in `params.drift_note`. Source: `results/approach3/IF/metrics.json`.
4. **No bid-ask data.** Daily CFFEX data have no bid-ask spread, so volume >= 10 stands in for liquidity/spread. On settle this filter removes 8,008 quotes; the full sequential filter removes 11,297. Source: `results/approach1/io/metrics.json`.
5. **LSTM refit frequency.** Q3B refits the LSTM every six months and reuses the last refit between refits; Q3A refits the probit monthly. Source: `results/approach6/macro/metrics.json`.
6. **Cluster bridge.** Q3A clusters partition indicators, while the LSTM's month-level `h_t` clusters partition months, so a direct ARI is undefined. The bridge maps each indicator to a correlation vector with the 16 hidden dimensions and clusters those vectors with `k=9`; the resulting ARI may be partly mechanical because `h_t` is computed from those indicators. Source: `results/approach6/macro/metrics.json`.
7. **SVI eta bounds.** The current bounds are `[0.01,1.0]`, narrower than the legacy `[1e-6,2.0]`. This changes bound hits but not median RMSE materially. Source: `results/approach1/io/metrics.json`.
8. **Q1B sample date.** Q1B uses 2024-03-14, while the metadata sample date is 2024-03-15. The selection rule chooses the nearest mid-month date with at least 80% of the month's maximum quote count. Source: `results/approach2/io_scaled/metrics.json`.
9. **Q3B inputs.** The LSTM excludes `gdp_yoy` from the 12-month input panel to avoid a mixed-frequency quarterly series; the probit also excludes it from the factors. Source: `results/approach6/macro/metrics.json`.
10. **Q2B scope.** Only IF is used, with one training seed and no terminal unwind; the brief permits this but it limits generalisation. Source: `results/approach4/IF/metrics.json`.
11. **Individual submission.** This is an individual submission rather than a group submission.
12. **AI-drafted report.** The report was drafted by AI rather than written by a group, with instructor approval as stated by the author in Section 14.
13. **Only IO options.** Only CFFEX IO options are used for the surface; SSE ETF options are not fitted. Source: `data/metadata.json` and `results/approach1/io/metrics.json`.
14. **Parity forward.** `q` is replaced by the put-call parity forward: `q = r - ln(F/S0)/T` with `F` estimated from parity, rather than a dividend model. Source: `results/approach1/io/metrics.json`.
15. **FastboxData not used.** FastboxData was not used as a data source.

## 12. Limitations

- SVI and neural fits are audited only on the reported grids. The neural surface has 30 butterfly and 4 calendar violations on the dense grid outside its training collocation, so it must not be called globally arbitrage-free. SVI is not proved arbitrage-free off the audit grid either.
- Both models are weak in the wings and at short tenors: Q1A settle worst per-fit max repricing is 0.3251 and close is 0.2624; Q1B wing p95 is 0.7809 and max 1.519, versus SVI 0.5592.
- SVI fits frequently sit on the eta bounds (31 lower and 67 upper on settle; 72 lower and 46 upper on close), and the cause is not established.
- Main-contract futures series have no roll flags. The fields named `roll_jump` in `metrics.json` are tail-return counts (IF 106, CU 113, RB 131, SC 119), not identified roll dates.
- Q2B used one training seed, only IF, no terminal unwind cost, and no previous position in the policy inputs; it cannot represent a no-trade region.
- Macro data are current revised vintages, not real-time vintages. Most publication lags are conservative assumptions, not observed release dates. GDP is derived from cumulative real levels (79 derived quarters; validation maximum absolute difference 0.83 points).
- The retrained LSTM never exceeds 0.5 on the test block (maximum 0.3172) and does not beat persistence (Brier 0.2561 vs 0.2286; AUC 0.6422 vs 0.7390). Its accuracy is the all-negative rate.
- The PCA rank is chosen on 12 quarterly nowcast observations; the test block is short for regime classification, and the AUC intervals are wide (for example, the original probit minus persistence interval is `[-0.2375, 0.03518]`).
- The Q3 nowcast month-3 ex-COVID error is higher than month-1; the paired interval includes zero, and no cause is asserted.
- The Q3 clustering is unstable across windows (adjacent ARI median 0.1648) and is descriptive only.
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

## 14. Statement of work

This is an individual submission by Hao Lu. The instructor approved individual completion with AI assistance. opencode with DeepSeek V4.1 Flash wrote the code, ran the experiments and drafted this report. Claude, in a separate chat that is not part of SESSION.md, drafted most of the prompts, proposed many methodology decisions and corrections, and reviewed outputs. Hao Lu directed the work, approved each step, and owns the final submission after reviewing it. The author does not claim to have independently derived the mathematics; the derivations above are a record of the implemented methods and their sources. The author has read this report.

## 15. Causal claims

Table 39: Prose claims about causes and their status. Source: `results/approach1/io/metrics.json`, `results/approach2/io_scaled/metrics.json`, `results/approach3/IF_study/metrics.json`, `results/approach4/IF/metrics.json`, `results/approach5/macro/metrics.json`, `results/approach6/macro/metrics.json`, and the report text.

| claim | status |
|---|---|
| Q1B dense-grid violations are caused by extrapolation outside the training collocation | hypothesis; violations are located outside the range, but the cause was not tested |
| Q1A eta-bound fits are caused by a flat or ill-conditioned objective | hypothesis; not tested |
| The negative mean q-minus-IF-basis gap is driven by short-dated annualization | hypothesis; not tested |
| The digital antithetic ratio ordering is explained by `r - sigma^2/2` near zero | hypothesis; supported by the ordering across datasets, not proved |
| The deep hedge CVaR gain is caused by direct CVaR optimization | hypothesis; optimization is not isolated from the state-dependent delta |
| The brief-minus-self-financing gap is caused by the omitted stock financing | tested by the two P&L formulas; the gap is a direct consequence of the formula difference |
| The brief-minus-self-financing gap is larger at high cost because omitted financing scales with the position | hypothesis; the pattern is visible in the table, but the scaling was not isolated |
| The original LSTM failure is caused by the 18-month validation design | evidence, not isolated: 4 of 24 validation blocks were single-class and tuning AUC was far below test AUC, but retraining changed hyperparameters and validation together |
| The larger original leakage premiums are caused by the unreliable 18-month validation design | hypothesis/evidence: the original design had single-class blocks and a large tuning-to-test gap, but the premium change is not isolated from the model change |
| The Q3A low AUC is caused by persistence and PMI being in the information set | hypothesis; not tested |
| The Q3B bridge ARI significance is partly mechanical because `h_t` is computed from the indicators | hypothesis; the null permutes labels but not the full construction |
| The 252 unmatched IO/IF forwards are caused by far IO expiries having no same-month IF contract | tested in code and raw data |
| The clean-clone FAIL is caused only by key-set differences | tested by the checker: zero numeric differences among shared keys |

## 16. Result files

Every number in the tables is from one of the following result files unless the table caption says otherwise: `results/approach1/io/metrics.json`, `results/approach1/io_close/metrics.json`, `results/approach2/io/metrics.json`, `results/approach2/io_scaled/metrics.json`, `results/approach3/IF/metrics.json`, `results/approach3/CU/metrics.json`, `results/approach3/RB/metrics.json`, `results/approach3/SC/metrics.json`, `results/approach3/IF_study/metrics.json`, `results/approach4/IF/metrics.json`, `results/approach5/macro/metrics.json`, `results/approach6/macro/metrics.json`. Metadata and row counts come from `data/metadata.json`. The Q1B per-tenor table is the one exception and is labeled in its caption. The clean-clone totals in Section 2 are from the `code/check_reproduction.py` output, and the session-integrity facts are from the session record.
