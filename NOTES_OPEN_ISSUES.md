# Open issues

## Q1B neural surface: optimization cause verified and fixed

The 2025-09-15 diagnostic verified that the original underperformance was an
optimization failure at the tiny price-loss scale, not a lambda-weighting effect
and not a capacity limit.

On that date (92 OTM quotes, 5 expiries, shared lambda-independent seed):

| run | fit MSE | median | p95 | max |
|---|---:|---:|---:|---:|
| approach1 SVI target | 2.278e-08 | 0.00685 | 0.04207 | 0.08194 |
| neural before, unscaled | 7.632e-07 | 0.05304 | 0.72232 | 0.96369 |
| neural after, scaled | 2.001e-08 | 0.00697 | 0.05418 | 0.14782 |

Before, L-BFGS stopped at `max_iter=100`; the final gradient norm was `5.34e-7`
(versus `tolerance_grad=1e-10`) and the loss moved only from `7.6324e-7` after
Adam to `7.6321e-7` after L-BFGS. Scaling the whole objective by
`c=1/initial_price_loss=162.65`, with 3000 Adam steps and 1000 L-BFGS steps,
reduced the fit MSE by 38x and let L-BFGS reduce the Adam-end loss by 6.1x.
An unscaled 3000/500 control reached only `1.271e-7`; a 64-unit capacity test
was worse (`2.330e-8` fit MSE), so capacity is not the bottleneck.

The lambda ablation was inactive on the diagnostic date: all 10 lambda pairs
gave bit-identical results and the penalty terms were exactly zero at
initialization, end of Adam and end of L-BFGS. The old apparent lambda effect
was seed confounding.

## Result after the full rerun (`results/approach2/io_scaled`)

Configuration: seeds independent of lambda; `c=1/initial_price_loss` applied to
the whole objective; lambda unchanged; 3000 Adam steps; L-BFGS `max_iter=1000`,
`max_evals=1500`, `max_seconds=45`, `tolerance_grad=1e-8`,
`tolerance_change=1e-12`; 24 units, 3 hidden layers; 60-minute guard.
Runtime: 1638 s. Coverage: 54/54 ablation fits, 18/18 random robustness fits,
6/6 wing robustness fits, 0 failed fits, determinism check passed.

Chosen lambdas: `lambda1=0.01`, `lambda2=0.0`, selected by complete coverage,
zero violations at 1e-8 on the approach1 audit grid, lowest pooled median, then
smallest `lambda1+lambda2`.

Overall comparison with approach1 on the same 470 OTM quotes:

| model | median | p95 | max |
|---|---:|---:|---:|
| approach1 repaired SVI | 0.008214 | 0.054525 | 0.173657 |
| neural chosen | 0.008687 | 0.123426 | 0.982950 |

Per tenor bucket (neural / SVI medians; p95 and max are pooled across dates):

| T bucket | n | neural median | SVI median | neural p95 | SVI p95 | neural max | SVI max |
|---|---:|---:|---:|---:|---:|---:|---:|
| <= 0.10 | 147 | 0.01627 | 0.02291 | 0.84913 | 0.13047 | 0.98295 | 0.17366 |
| 0.10-0.30 | 201 | 0.00801 | 0.00810 | 0.09268 | 0.04335 | 0.13464 | 0.05622 |
| 0.30-0.60 | 75 | 0.00484 | 0.00486 | 0.02839 | 0.01945 | 0.03722 | 0.02422 |
| > 0.60 | 47 | 0.00420 | 0.00406 | 0.03892 | 0.03027 | 0.05528 | 0.03982 |

Robustness versus SVI:

| design | model | fits | n | median | p95 |
|---|---|---:|---:|---:|---:|
| random 30% | SVI | 18/18 | 407 | 0.016297 | 0.126115 |
| random 30% | neural | 18/18 | 407 | 0.018359 | 0.221206 |
| wing 15% | SVI | 6/6 | 83 | 0.050337 | 0.559186 |
| wing 15% | neural | 6/6 | 83 | 0.093261 | 0.780917 |

The neural surface is now competitive in overall price MSE and in median
relative error, but it is still worse in the tails, on the shortest tenor and
on the wing holdout.

## Penalty activity on all dates

`penalty_max` is the largest unweighted penalty term seen during Adam/L-BFGS.
"Differs from (0,0)" means the lambda pair changed the final metrics or
violation counts relative to the unpenalized fit on that date.

| date | max butterfly penalty | max calendar penalty | differs from (0,0) | outcome |
|---|---:|---:|---|---|
| 2024-03-14 | 0 | 0 | no | all lambda pairs identical |
| 2024-09-13 | 4.270e-04 | 2.243e-06 | no | penalties transient; final metrics identical |
| 2025-03-14 | 2.444e-08 | 0 | no | final metrics identical |
| 2025-09-15 | 0 | 0 | no | all lambda pairs identical |
| 2026-03-16 | 6.733e-05 | 0 | yes | `lambda1=0.01` removes 3 approach1-grid butterfly violations and improves median 0.00816 -> 0.00641; `lambda1=1` is worse (median 0.01436, 1 violation) |
| 2026-09-15 | 9.899e-01 (lambda1=1); 4.357e-05 (lambda1=0.01) | 4.553e-08 | yes | `lambda1=0.01` improves median 0.00987 -> 0.00869 and keeps zero approach1-grid violations; `lambda1=1` gives median 0.01235 with 1 butterfly violation |

So the penalties were not inactive on every date. They were active in training
on four of six dates, but changed final results only on 2026-03-16 and
2026-09-15. `lambda1=1` is too strong when it binds: it dominates the O(1e-6)
fit term and produces a worse surface. `lambda1=0.01` is useful on those two
dates.

## Remaining issue: dense-grid arbitrage outside the training collocation

Chosen model on the approach1 audit grid (k in [-0.14,0.14]): 0 butterfly and
0 calendar violations at strict and 1e-8. On the denser grid (k from -0.20 to
0.20 step 0.005, listed expiries plus midpoints), the chosen model has
30 butterfly and 4 calendar violations, all outside the training collocation
range |k|<=0.14:

- dense butterfly: 2024-09-13 (15), 2025-03-14 (14), 2026-03-16 (1);
  minimum dense `g = -0.5603`.
- dense calendar: 2026-09-15 (4); minimum dense forward variance `-0.00502`.

The worst relative repricing error is also a wing quote outside the collocation
range: 2026-09-15, 2026-10-16 expiry, `k=0.181`, price 1.4, relative error
0.983. The price-space loss barely penalizes it because its unit-forward price
is tiny. The training and selection grids therefore overstate arbitrage
freeness; extending the collocation to the full `kband` (+/-0.20) and adding
wing collocation points is the next change to consider.

## Superseded result

`results/approach2/io` is kept for the record and labeled in `compare.py` as
"superseded: optimization failure, kept for the record". `io_scaled` is the
primary Q1B result.
