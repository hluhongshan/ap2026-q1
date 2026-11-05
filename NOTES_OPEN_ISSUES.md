# Open issues

## Q1B neural surface underperforms approach1 SVI

On the same 470 OTM quotes, the chosen neural surface (`lambda1=1.0`, `lambda2=0.01`) is clearly worse than approach1's repaired SVI:

| Model | median | p95 | max |
|---|---:|---:|---:|
| approach1 repaired SVI | 0.008 | 0.055 | 0.174 |
| neural chosen | 0.044 | 0.678 | 0.982 |
| neural unpenalized `(0, 0)` | 0.053 | 0.694 | 0.998 |

The unpenalized network is also worse, so the gap is not only a penalty effect.

## Suspected cause (NOT verified)

The network may be underfitting because the option-price loss is very small (about `1e-6` scale). Adam and L-BFGS may stop at tolerances that are too loose relative to this scale. This has not been diagnosed or verified.

## Ablation confounding

The ablation is confounded because each lambda pair gets a different derived seed. Within a date, all lambda pairs should share one initialization (and differ only in lambda) to isolate the penalty effect.

## Planned diagnostic (if time allows)

- Compare the price-space loss of SVI and the network on the same quotes.
- Inspect the L-BFGS stop reasons and evaluation counts.
- Rescale the objective by a constant and check whether optimisation improves.
