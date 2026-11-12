# apigraph-repro: tm-cyber-lab's APIGraph numbers, reproduced by this repository's pipeline

**Verdict: PASS** under the amended criterion; **FAIL** under the criterion set before the first run.
Detection numbers reproduce exactly. The first run's XGBoost efficiency check missed its tolerance
(3.91e-5 against 1e-5) because XGBoost's own margin is accumulated in float32. The criterion was then
amended, before the second run, to check XGBoost's attributions against a float64 sum of the same trees
within 1e-6; the second run gives 7.8e-8. Both checks are computed and reported by `run.py`.

## Set-up

APIGraph Drebin features (datasets/apigraph/DATASET_CARD.md), width 1,159. Train on the 2012 pool (30,533
rows, 10.0% malware); test on each year 2013-2018, F1 of the malware class pooled over the months of the
year. Reference: tm-cyber-lab `research/b5-apigraph` at commit e6f8fb8; TMCore from tm-lab at 43dba5f.

| arm | configuration | threshold rule |
|---|---|---|
| FPTM | 20 clauses per class, T=10, S=100, L=64, LF=10, 30 epochs, seeds 1-10 | malware class score > benign class score (ties benign) |
| XGBoost | 5000 trees, learning rate 0.02, 256 leaves, hist, seed 0 | probability > 0.5 |
| logistic regression | C=1, lbfgs | probability > 0.5 |

## Detection

| year | FPTM S=100 (10 seeds) | lab | XGBoost | lab | logistic regression |
|---|---|---|---|---|---|
| 2013 | 84.24 ± 2.01 | 84.24 ± 2.01 | 83.29 | 83.29 | 82.44 |
| 2014 | 66.76 ± 3.07 | 66.76 ± 3.07 | 69.88 | 69.88 | 68.64 |
| 2015 | 61.25 ± 2.24 | 61.25 ± 2.24 | 66.13 | 66.13 | 66.45 |
| 2016 | 59.65 ± 3.56 | 59.65 ± 3.56 | 68.39 | 68.39 | 61.24 |
| 2017 | 64.55 ± 4.56 | 64.55 ± 4.56 | 72.13 | 72.13 | 71.44 |
| 2018 | 65.28 ± 4.12 | 65.28 ± 4.12 | 72.30 | 72.30 | 72.28 |

FPTM values are the mean and sample standard deviation over seeds. All twelve lab values agree to two
decimals. FPTM seeds 1-3 were also trained in tm-cyber-lab's own Julia environment, from its own
matrices; all eighteen per-year F1 values were identical to this repository's. The 73 `.tmx` matrices
built here are byte-identical to the lab's.

XGBoost leads FPTM in 2014-2018 by 3.1 to 8.7 F1 and trails it in 2013 by 0.95. Logistic regression is
within 1.3 F1 of XGBoost in every year except 2016 (7.15 behind).

## Attribution efficiency

100 rows of 2013 explained against 100 background rows of the 2012 pool (fixed draw, `default_rng(0)`).
Interventional Shapley values of each model's margin should sum to margin(x) - mean margin(background).

| model | method | margin compared with | max gap | tolerance | result |
|---|---|---|---|---|---|
| FPTM (seed 1) | closed form | model score difference | 4.3e-14 | 1e-9 | pass |
| logistic regression | w (x - mean background) | decision function | 3.6e-15 | 1e-9 | pass |
| XGBoost | interventional TreeSHAP (shap 0.54.1) | float64 sum of the trees (amended) | 7.8e-8 | 1e-6 | pass |
| XGBoost | interventional TreeSHAP (shap 0.54.1) | XGBoost's own float32 margin (original) | 3.9e-5 | 1e-5 | fail |

Criterion history: the original XGBoost check (last row) was set before the first run and failed. The
diagnosis below came next; the amended check (third row) was recorded in the header of `run.py` on
2026-11-12 before the second run. Both runs gave the same numbers in every row.

`xgb_precision.py` refits the same model and separates the two sources:

| quantity | value |
|---|---|
| XGBoost float32 margin minus float64 sum of the same trees, max | 3.3e-5 |
| gap against the float32 margin | 3.9e-5 |
| gap against the float64 tree sum | 7.8e-8 |
| largest margin magnitude | 14.0 |

The tolerance of 1e-5 did not allow for float32 accumulation over 5000 trees.

## Files

- `run.py`: the demonstration; `results.txt` its output, `results.json` every number including per-seed F1.
- `xgb_precision.py`: the float32 diagnosis; `xgb_precision.json` its output.

Run time on a laptop (Windows, 16 threads), second run: XGBoost fit and predict 123 s, its attribution
13 s for 100 x 100; each FPTM seed 22-29 s including Julia start-up (30 epochs train in 2.6-3.4 s), FPTM
attribution 30 s for 100 x 100. The first run took about twice as long on a busier machine.
