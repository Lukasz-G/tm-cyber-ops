# Demonstrations

One row per demonstration, its scope and its verdict. Each directory holds the run script (its header
states the question, scope, arms and pass criterion), a README and the results file.

| demo | scope | verdict | result |
|---|---|---|---|
| [apigraph-repro](apigraph-repro/) | APIGraph, train 2012 pool (30,533 rows), test 2013-2018 per year; FPTM 10 seeds, XGBoost, logistic regression | PASS (amended criterion; FAIL under the original) | Per-year F1 equals tm-cyber-lab b5-apigraph to two decimals for FPTM and XGBoost (2013: 84.24 ± 2.01 and 83.29). Efficiency: FPTM 4.3e-14, logistic regression 3.6e-15, XGBoost 7.8e-8 against a float64 sum of the same trees (amended tolerance 1e-6); 3.9e-5 against XGBoost's float32 margin (original tolerance 1e-5). |
| [apigraph-drift-floor](apigraph-drift-floor/) | APIGraph, models fixed after training on the 2012 pool, 72 months 2013-01..2018-12, 100 explained rows per month, top-100 features; FPTM seeds 1-3, XGBoost, logistic regression | PASS (one-sided gate; the floor sits above the series' noise) | Month-shuffled control alerts in 0-0.2% of 71 transitions (gate: at most 10%). Alerts above the same-month row + background floor: logistic regression 4, FPTM 1 / 0 / 1 (2013-10 for seeds 1 and 3), XGBoost 0. Mean month-to-month drift is within 0.02 of the same-month row-sampling median for every model. |
