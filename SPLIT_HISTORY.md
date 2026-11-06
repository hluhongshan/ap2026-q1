# Split history for ap2026-q1

## Provenance and split disclosure

The work was done in one repository (parent tag `submission`, commit `4521da3`) and split afterwards into one repository per question following the instructor's guidance. Commit hashes changed and the old-to-new commit map is included below. Dates in rewritten commits are the original ones; the split commits are dated when they were made. Shared-file commits appear in more than one repository. `SESSION.md` is the full transcript for all questions and refers to old file names and hashes. Agent work on the split ran with `ap2026-split` as its root.

## Path mapping

| Old path or name | New path or name |
|---|---|
| `(none)` | `(none)` |

## Commit map

The map is the unedited `git filter-repo` output. `pruned` means the commit became empty for this path set. `same` means filter-repo produced the identical hash; `rewritten` means the hash changed. Shared commits are listed with every repository that retained them.

| Old commit | New commit | Subject | Status | Retained in |
|---|---|---|---|---|
| `7ee9a860ad8a` | `7ee9a860ad8a` | chore: add agent contract before first agent session | same | Q1, Q2, Q3 |
| `bc857ecc516a` | `bc857ecc516a` | chore: scaffold repository | same | Q1, Q2, Q3 |
| `867a243cd90d` | `867a243cd90d` | data: add Q1 fetch pipeline and snapshot | same | Q1, Q2, Q3 |
| `278562b16b99` | `278562b16b99` | data: roll holiday option expiries to next trading day | same | Q1, Q2, Q3 |
| `0b680be37ac1` | `0b680be37ac1` | feat: Q1A SVI pipeline | same | Q1, Q2, Q3 |
| `c8c840c4b3e4` | `c8c840c4b3e4` | exp: Q1A results on IO | same | Q1 |
| `317fe9c55a8a` | `317fe9c55a8a` | fix: Q1A repricing basis and SVI bounds | same | Q1 |
| `18b5cc2f237d` | `18b5cc2f237d` | exp: Q1A results, settle primary | same | Q1 |
| `0e0a6b91984a` | `0e0a6b91984a` | chore: remove stale Q1A run | same | Q1 |
| `7551ad29752f` | `7551ad29752f` | exp: like-for-like forward check | same | Q1 |
| `0af1dd09691f` | `0af1dd09691f` | feat: Q1B differentiable SVI and neural surface | same | Q1 |
| `ab3313adfc8f` | `ab3313adfc8f` | fix: Q1B robustness coverage reporting | same | Q1 |
| `664a71673bbd` | `664a71673bbd` | exp: Q1B results on IO | same | Q1 |
| `84a746a51ab8` | `84a746a51ab8` | docs: record Q1B open issues | same | Q1 |
| `5f007e65d2ca` | `48ca0a666af2` | data: add Q2 futures snapshot | rewritten | Q1, Q2, Q3 |
| `7950d6fe1852` | `d038f87a9c77` | feat: Q2A Monte Carlo engine and estimators | rewritten | Q1, Q2, Q3 |
| `4fd0ca365927` | pruned | exp: Q2A results | pruned | Q2 |
| `32bfbcdf9c77` | pruned | fix: Q2A study reference consistency | pruned | Q2 |
| `89a27c6db5ed` | pruned | exp: refresh Q2A study outputs | pruned | Q2 |
| `d67df3903f33` | pruned | feat: Q2B deep hedging | pruned | Q2 |
| `b2a71278ed7d` | pruned | fix: Q2B default output path | pruned | Q2 |
| `fcdaf1096e58` | pruned | exp: Q2B results | pruned | Q2 |
| `4fc1d1053860` | `83cac191214d` | data: add Q3 macro panel | rewritten | Q1, Q2, Q3 |
| `6255defa60cc` | pruned | feat: Q3A factors, nowcast, probit, clustering | pruned | Q3 |
| `1922099796cc` | pruned | exp: Q3A results | pruned | Q3 |
| `d2cf1893d830` | pruned | exp: Q3A post-hoc probit regularization check | pruned | Q3 |
| `a87b9f3dfb18` | pruned | feat: Q3B LSTM and leakage ablation | pruned | Q3 |
| `5131fe20204d` | pruned | exp: Q3B results | pruned | Q3 |
| `9790ce9e52a3` | pruned | fix: Q3B training loop diagnostics | pruned | Q3 |
| `17bbfd75a41e` | pruned | exp: Q3B post-hoc retraining | pruned | Q3 |
| `fb4781adbadd` | `c887af7ce700` | feat: compare layer and reproduction script | rewritten | Q1, Q2, Q3 |
| `b5c09a83baf1` | `c9257333194d` | docs: update agent contract and README | rewritten | Q1, Q2, Q3 |
| `7290853b7bbe` | `2b31375e54b4` | fix: Q1B objective scaling and shared initialization | rewritten | Q1, Q2, Q3 |
| `d4a7f215effb` | `1ca85a6acbe7` | exp: Q1B results after objective scaling | rewritten | Q1, Q2, Q3 |
| `80009968446d` | `b82376b5a61b` | docs: correct Q1B tenor-bucket aggregation | rewritten | Q1 |
| `076e9d43a48c` | `d73073e45f33` | fix: use repo-relative paths in outputs | rewritten | Q1, Q2, Q3 |
| `e6ae1a473870` | `e26803a8d4fd` | docs: note historical approach2/io run | rewritten | Q1, Q2, Q3 |
| `ab12e7080c73` | `81a095ea20a9` | data: commit CFFEX raw snapshot for offline Q1 cross-check | rewritten | Q1, Q2, Q3 |
| `4a6164be766d` | `4401c5c3843f` | fix: commit legacy SVI snapshots used by Q1 comparisons | rewritten | Q1, Q2, Q3 |
| `509ac9f68ea4` | `11cde5c96af0` | chore: add reproduction check | rewritten | Q1, Q2, Q3 |
| `311bade4fe68` | `c1b839fa3102` | docs: document reproduction caveats | rewritten | Q1, Q2, Q3 |
| `8105e5dfd0ec` | `c8c9f7666ffa` | exp: refresh Q1A close results from current code | rewritten | Q1, Q2, Q3 |
| `b226622810fd` | pruned | fix: Q3B post-hoc origin_date label | pruned | Q3 |
| `e27222bdeaf5` | `501668f58ad8` | feat: Q1B cold-start SVI solver comparison | rewritten | Q1, Q2, Q3 |
| `99210e96e740` | `603a9133e0fc` | exp: Q1B cold-start solver results | rewritten | Q1 |
| `3015c7822b0c` | `f661d3d549e6` | report: add project report | rewritten | Q1, Q2, Q3 |
| `dad4ca07163c` | `91b7a93e8d57` | report: author review confirmation | rewritten | Q1, Q2, Q3 |
| `4521da3ed674` | `862b4017ce04` | report: add complete session export | rewritten | Q1, Q2, Q3 |

## Commit counts

- Original parent history: 48 commits.
- Commits retained by this repository after filtering: 34.
- Later split commits are not part of the filter-repo map; they are ordinary commits on top of the filtered history and are dated when made.
- No `submission` tag was created by the split; the parent tag is recorded above.
