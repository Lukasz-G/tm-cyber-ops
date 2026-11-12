# apigraph-drift-floor: does a fixed model's attribution ranking move more than its own noise floor?

**Verdict: PASS** under the criterion set before the run: the month-shuffled control alerts in at most
10% of transitions for every model. The gate is one-sided, and the control alerted in 0-0.2% of
transitions where a floor matched to the series' noise would give about 5%. This floor sits above that
noise, so the monitor is close to silent. At k=100 it raises 4 alerts in 71 transitions for logistic
regression, 1, 0 and 1 for FPTM seeds 1-3 (2013-10 for seeds 1 and 3), and none for XGBoost.

## Set-up

APIGraph Drebin features (datasets/apigraph/DATASET_CARD.md), width 1,159. Every model is trained once
on the 2012 pool (30,533 rows) and kept fixed for the 72 test months 2013-01..2018-12.

| | |
|---|---|
| models | FPTM S=100 (20 clauses per class, T=10, L=64, LF=10, 30 epochs), seeds 1-3; XGBoost (5000 trees, learning rate 0.02, 256 leaves); logistic regression |
| attributions | exact closed form (FPTM), interventional TreeSHAP (XGBoost), exact linear formula (logistic regression), all of the margin, against a background set |
| series | 100 explained rows per month, background 1 (100 rows of the pool) fixed for every month |
| importance | mean \|phi\| over the explained rows; top-k among features with nonzero importance, ties by index (tm-cyber-lab `research/b2-drift`, commit e6f8fb8) |
| drift | Jaccard distance 1 - \|A ∩ B\| / \|A ∪ B\| between the top-k sets of consecutive months, 71 transitions |
| floor | 12 months (January and July, 2013-2018), 200 rows each, explained against background 1 and a disjoint background 2; 50 random splits into two halves; distance between half 1 under background 1 and half 2 under background 2; 600 values |
| alert | drift above the floor's 95th percentile, per model and seed, at k=100 |
| control | the 7,200 explained rows of the series shuffled across months, 20 shuffles; no temporal signal remains |
| gate | control alert rate (mean over shuffles) at most 10% for every model and seed |

Monthly F1, printed beside each alert, uses each model's own threshold rule: FPTM predicts malware when
the malware class score exceeds the benign class score (ties benign); XGBoost and logistic regression
when the predicted probability exceeds 0.5. F1 is the malware class within one month.

## Result at k=100

| model | features with nonzero importance | drift mean | floor median | floor p95 | control | alerts |
|---|---|---|---|---|---|---|
| logistic regression | 504 | 0.204 | 0.230 | 0.291 | 0.002 | 4 |
| FPTM seed 1 | 411 | 0.170 | 0.230 | 0.276 | 0.000 | 1 |
| FPTM seed 2 | 391 | 0.150 | 0.198 | 0.276 | 0.000 | 0 |
| FPTM seed 3 | 391 | 0.130 | 0.182 | 0.246 | 0.000 | 1 |
| XGBoost | 258 | 0.156 | 0.182 | 0.276 | 0.000 | 0 |

| model | month | drift / floor p95 | F1 previous month -> this month |
|---|---|---|---|
| logistic regression | 2013-07 | 0.319 / 0.291 | 77.4 -> 86.0 |
| logistic regression | 2013-09 | 0.319 / 0.291 | 82.1 -> 80.1 |
| logistic regression | 2013-12 | 0.305 / 0.291 | 78.7 -> 74.1 |
| logistic regression | 2014-11 | 0.305 / 0.291 | 65.2 -> 71.1 |
| FPTM seed 1 | 2013-10 | 0.291 / 0.276 | 87.5 -> 83.4 |
| FPTM seed 3 | 2013-10 | 0.261 / 0.246 | 84.0 -> 81.2 |

## Context, not gated

| model | drift mean | row-only floor median | row-only floor p95 | background-only mean | transitions above the row-only p95 |
|---|---|---|---|---|---|
| logistic regression | 0.204 | 0.198 | 0.276 | 0.128 | 7 |
| FPTM seed 1 | 0.170 | 0.165 | 0.214 | 0.159 | 10 |
| FPTM seed 2 | 0.150 | 0.131 | 0.214 | 0.135 | 5 |
| FPTM seed 3 | 0.130 | 0.113 | 0.182 | 0.134 | 6 |
| XGBoost | 0.156 | 0.148 | 0.230 | 0.109 | 0 |

The row-only floor takes both halves under background 1, as the series does; the background-only value
compares the same half under the two backgrounds.

1. **Month-to-month drift sits at the row-sampling level.** For every model the mean distance between
   consecutive months is within 0.02 of the median distance between two halves of one month under one
   background. At 100 explained rows, adjacent months differ little more than two samples of the same
   month do.
2. **The background redraw adds noise of the same size as the row sample** (0.109-0.159 at k=100). The
   series keeps its background fixed, so the gated floor counts noise that the series does not carry,
   which is why the control stays at 0-0.2%.
3. Above the row-only p95 (no control was computed at that threshold), transitions cluster in
   2013-10..2014-02 for logistic regression and all three FPTM seeds; 2013-10 and 2014-02 appear for
   all three seeds. XGBoost has none.
4. At k=1000 the gated floor is dominated by the background (background-only mean 0.16-0.41): which
   features have nonzero attribution depends on which features the background rows contain. No alerts
   at k=1000 under the gated floor; 0-7 above the row-only p95.
5. Different FPTM seeds on the same month, rows and background differ by 0.196 at k=100 (sd 0.028, 216
   pairs), more than any FPTM seed drifts from month to month (0.130-0.170). XGBoost and logistic
   regression are deterministic at these configurations.

Monthly F1 ranges from about 28 to 96 for every model; per-month values are in `results.json`, and each
transition with its drift, threshold and F1 is in `series.csv`.

## Procedure

Before this run, a smoke run with reduced sizes (FPTM seeds 1-2, 4 splits, 2 shuffles, a 20-tree
XGBoost) tested the script end to end. It showed logistic regression and FPTM numbers close to the ones
above. The design was not changed after it.

## Files

- `run.py`: the demonstration (header: question, scope, arms, pass criterion); `results.txt` its
  output; `results.json` every number, including monthly F1 and the drift series; `series.csv` one row
  per model, k and transition.

Run time on a laptop (Windows, 16 threads): logistic regression 4 s, each FPTM seed 71-76 s, XGBoost
980 s (TreeSHAP for 10,800 rows dominates), 1,205 s in total.
