# apigraph-drift-reference: is a month-to-month drift alarm calibrated at a fixed class mix, and how far does each month sit from the training year?

**Verdict: FAIL** under the criterion set before the run. The criterion asks that, for every model and
FPTM seed, a month-shuffled control alerts in between half and twice the floor's own exceedance rate e.
Five of twelve models meet it: logistic regression, XGBoost and FPTM seeds 1, 3 and 6. For FPTM seeds 2,
5, 8 and 10 the control alerts less than e/2; for seeds 4, 7 and 9 it alerts more than 2e. The
month-to-month alert at k=100 therefore has no stable false-alarm rate in this set-up: across FPTM
seeds the control alerts in 0.0% to 7.5% of transitions.

The distance from a held-out sample of the training year, which is reported but not gated, grows from
2013 to 2018 for every model. From 2014 on, every month of every model lies above the reference floor.

## Set-up

APIGraph Drebin features (datasets/apigraph/DATASET_CARD.md), width 1,159. The 2012 pool (30,533 rows,
10.0% malware) is split by a label-stratified random draw. Every model trains once on 27,480 rows at
the native share, with no class weights and no resampling, and stays fixed for the 72 test months
2013-01..2018-12 (5.8-10.0% malware per month). The other 3,053 rows (10% of each class) are held out.

| | |
|---|---|
| models | FPTM S=100 (20 clauses per class, T=10, L=64, LF=10, 30 epochs), seeds 1-10; XGBoost (5000 trees, learning rate 0.02, 256 leaves); logistic regression |
| attributions | exact closed form (FPTM), interventional TreeSHAP (XGBoost), exact linear formula (logistic regression), all of the margin, against one 100-row background (90 clean, 10 malware) drawn from the training rows |
| explained rows | every set has the pool's class mix: 450 clean and 50 malware rows per 500 (each month, each floor half, the reference, each reference-floor draw) |
| importance | mean \|phi\| over the explained rows; top-k among features with nonzero importance, ties by index (tm-cyber-lab `research/b2-drift`; class mix as in `research/b2-corpora`; commit 277531a) |
| drift | Jaccard distance 1 - \|A ∩ B\| / \|A ∪ B\| between top-k sets; 71 month-to-month transitions |
| floor | 24 months (January, April, July, October of 2013-2018), 1,000 rows each (900 clean, 100 malware), 50 random stratified splits into two halves of 500; distance between the halves under the same background as the series; 1,200 values |
| alert | a transition whose k=100 distance exceeds the floor's 95th percentile |
| e | the share of floor values strictly above that percentile; below 5% where floor values tie on it |
| control | the 36,000 explained rows of the series shuffled across months within each class, so each shuffled month keeps 450 clean and 50 malware rows; 20 shuffles |
| gate | for every model and FPTM seed, e > 0 and the control alert rate (mean over shuffles) in [e/2, 2e] |

Detection is measured per month and on the held-out rows, as precision, recall and F1 of the malware
class at two cut-offs on each model's margin. The default rule is margin > 0: FPTM predicts malware when
the malware class score exceeds the benign class score (ties benign), XGBoost and logistic regression
when the predicted probability exceeds 0.5. The held-out rule is the cut-off that maximises F1 on the
3,053 held-out 2012 rows (10.0% malware), applied as margin >= cut-off, as in tm-cyber-lab
`research/threshold-corpora` (commit 277531a). Under the held-out rule, the held-out figures are
measured on the same rows that chose the cut-off.

## Gate at k=100

| model | e | control | allowed range | in range | month-to-month alerts (of 71) | floor median / p95 | control median |
|---|---|---|---|---|---|---|---|
| logistic regression | 0.022 | 0.031 | 0.011-0.045 | yes | 4 | 0.077 / 0.131 | 0.095 |
| XGBoost | 0.015 | 0.010 | 0.007-0.030 | yes | 13 | 0.058 / 0.095 | 0.058 |
| FPTM seed 1 | 0.022 | 0.015 | 0.011-0.043 | yes | 4 | 0.095 / 0.131 | 0.077 |
| FPTM seed 2 | 0.007 | 0.000 | 0.004-0.015 | no | 4 | 0.058 / 0.113 | 0.058 |
| FPTM seed 3 | 0.015 | 0.016 | 0.007-0.030 | yes | 12 | 0.077 / 0.113 | 0.077 |
| FPTM seed 4 | 0.022 | 0.071 | 0.011-0.045 | no | 8 | 0.077 / 0.113 | 0.095 |
| FPTM seed 5 | 0.042 | 0.008 | 0.021-0.083 | no | 8 | 0.058 / 0.095 | 0.058 |
| FPTM seed 6 | 0.047 | 0.028 | 0.023-0.093 | yes | 12 | 0.058 / 0.095 | 0.058 |
| FPTM seed 7 | 0.010 | 0.042 | 0.005-0.020 | no | 9 | 0.077 / 0.113 | 0.077 |
| FPTM seed 8 | 0.032 | 0.005 | 0.016-0.063 | no | 13 | 0.058 / 0.095 | 0.058 |
| FPTM seed 9 | 0.031 | 0.075 | 0.015-0.062 | no | 16 | 0.058 / 0.095 | 0.077 |
| FPTM seed 10 | 0.041 | 0.004 | 0.020-0.082 | no | 10 | 0.077 / 0.113 | 0.058 |

At k=100 the distance takes discrete values: 5, 6, 7 and 8 swapped features give 0.095, 0.113, 0.131
and 0.148. Every floor p95 is one of the first three values, so a transition alerts when it swaps at least
one feature more than the floor's p95 does. The failures go in both directions and show no pattern by model.
For FPTM seeds 4 and 9 the control median lies one step above the floor median, and for seed 10 one step
below, in the same direction as the failure. For the other failing seeds the medians agree. The control
pools rows from all 72 months, while the floor uses 24 months; this run does not separate that
difference from the step size as a cause.

Month-to-month alerts (all in `series.csv`) are more frequent in 2013: 33 of the 113 alerts fall in
its 11 transitions, which are 15% of all transitions. 2013-02 alerts for XGBoost and FPTM seeds
5-10; 2013-08 for logistic regression, XGBoost and FPTM seeds 5, 6, 8, 9 and 10. Since the gate failed,
these alerts carry no calibrated false-alarm rate.

## Distance from the training year (reported, not gated)

Each month's 500 rows against the 500 held-out 2012 rows of the reference, at k=100. The reference
floor is the same 500 rows against 1,200 draws of 500 other held-out rows; its p95 is 0.165 for logistic
regression, 0.113 for XGBoost and 0.113-0.148 across FPTM seeds. Mean distance per year, with the
number of months above the reference floor p95 in brackets:

| model | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|
| logistic regression | 0.187 (7) | 0.256 (12) | 0.282 (12) | 0.310 (12) | 0.320 (12) | 0.347 (12) |
| XGBoost | 0.116 (5) | 0.169 (12) | 0.195 (12) | 0.235 (12) | 0.221 (12) | 0.246 (12) |
| FPTM, mean of 10 seeds (range of months) | 0.174 (6-11) | 0.245 (12) | 0.270 (12) | 0.300 (12) | 0.297 (12) | 0.309 (12) |

A month could sit far from the reference simply because 500 rows of one month differ from 500 rows
drawn across a year. To check this, each month is also compared with 500 rows from the other 11 months
of its own year, against the same reference floor. That distance averages 0.045-0.133 per model, seed
and year. Of 72 months, 2 lie above the reference floor for logistic regression and 8 for XGBoost. Across
FPTM seeds the count is 0-17 (seeds 1-10: 3, 4, 11, 2, 5, 8, 4, 1, 0, 17). Seed 10 has 5 of its 17 in
2017, and seed 6 has 8 of 12 in 2013.

## Detection

Mean of the 12 monthly values per year (no pooling of rows across months), precision / recall / F1 in
percent for logistic regression and XGBoost, F1 mean ± sd over the 10 seeds for FPTM. The held-out
column is the 3,053 held-out 2012 rows.

| model | rule (cut-off) | held-out | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|---|
| logistic regression | default (0) | 95.6/92.2/93.8 | 94.0/73.6/82.3 | 81.9/59.1/68.5 | 76.4/56.8/64.5 | 71.6/55.6/62.3 | 70.9/67.2/68.7 | 74.2/70.0/71.7 |
| logistic regression | held-out (-1.35) | 92.2/96.1/94.1 | 88.0/85.0/86.4 | 72.2/74.9/73.4 | 65.6/64.8/64.7 | 58.8/66.1/62.1 | 60.9/74.5/66.9 | 62.9/74.8/68.0 |
| XGBoost | default (0) | 96.7/94.8/95.7 | 95.2/76.7/84.8 | 90.1/59.3/71.4 | 87.0/53.8/65.3 | 78.7/52.4/62.5 | 75.6/63.6/68.6 | 81.7/63.3/70.9 |
| XGBoost | held-out (1.478) | 98.6/94.1/96.3 | 97.7/68.6/80.3 | 93.6/45.2/60.9 | 92.8/43.7/58.4 | 79.5/36.5/49.5 | 75.2/45.7/56.0 | 85.3/54.5/65.3 |
| FPTM F1 | default (0) | 94.1-94.8 | 83.8 ± 1.6 | 66.5 ± 2.4 | 60.8 ± 2.3 | 58.8 ± 3.0 | 64.6 ± 5.1 | 68.3 ± 4.9 |
| FPTM F1 | held-out (-5 to 3) | 94.8-95.3 | 85.9 ± 1.8 | 69.9 ± 3.0 | 64.0 ± 2.0 | 63.2 ± 2.9 | 68.9 ± 4.4 | 70.0 ± 4.0 |

On the held-out rows all models reach F1 93.8-96.3. Under the default rule XGBoost has a higher F1
than the FPTM seed mean in every year, by 1.0 to 4.9 points. Under the held-out rule the FPTM seed mean
is higher than XGBoost in every year, by 4.7 to 13.7 points. XGBoost's held-out cut-off (log-odds
1.478, probability 0.81) lowers recall in every year, by 8.1 to 17.9 points, and raises precision in
five of six. FPTM margins are integer vote
differences, so a cut-off of 1 equals the default rule; this is the case for seed 8. Per-seed and
per-month values are in `detection.csv`.

## Context, not gated

1. At k=1000, where fewer floor values tie, e is 0.045-0.050 for every model. The control lies in
   [e/2, 2e] for all ten FPTM seeds and outside it for logistic regression (0.131, above 2e = 0.100)
   and XGBoost (0.006, below e/2 of about 0.025). The same criterion at k=1000 would also have failed.
2. Different FPTM seeds on the same month, rows and background differ by 0.224 at k=100 and 0.275 at
   k=1000. Month-to-month drift averages 0.077-0.101 at k=100 for every model.
3. The same rows under a second background differ by 0.150-0.191 at k=100.

## Procedure

The run script, including its header, was fixed before the full run. Three smoke runs with reduced
sizes tested it (the third: FPTM seeds 1-2, 4 splits, 2 shuffles, 50 reference-floor draws, a 20-tree
XGBoost). In the second, the earlier gate (control alert rate in [1%, 10%]) failed for FPTM, whose
control alerted in 0-0.7% of transitions because distances tie at k=100. The design then changed,
before any full-run number existed: stratified 450 + 50 explained rows everywhere, the native training
share, the gate relative to e, and the second cut-off. A risk was recorded before
the full run: in the third smoke run FPTM seed 2 had e = 0 at 96 floor values, because the top 5% tied
at the maximum. With 1,200 floor values e > 0 for every model.

A first full run was stopped after logistic regression had finished (4 s) to add the own-year
comparison; none of its results were read. The full run then ran once without changes.

The header gives the monthly malware share as 5.8-10.1%; the data give 5.75-10.05%. In `results.txt`
one line of TreeSHAP progress bars was removed; nothing else was changed.

## Files

- `run.py`: the demonstration (header: question, scope, arms, pass criterion); `results.txt` its
  output; `results.json` every number; `series.csv` one row per model, k, series and transition;
  `detection.csv` precision, recall and F1 per model, rule and period.
- `uv run python demos/apigraph-drift-reference/run.py --reuse` reproduces the numbers from the saved
  attributions and margins in `work/phi/` (not committed).

Run time on a laptop (Windows, 16 threads): logistic regression 3 s, XGBoost 3,028 s (TreeSHAP for
63,053 rows), FPTM 308-1,117 s per seed, 8,448 s in total.
