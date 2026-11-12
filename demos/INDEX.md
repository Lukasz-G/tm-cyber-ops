# Demonstrations

One row per demonstration, its scope and its verdict. Each directory holds the run script (its header
states the question, scope, arms and pass criterion), a README and the results file.

| demo | scope | verdict | result |
|---|---|---|---|
| [apigraph-repro](apigraph-repro/) | APIGraph, train 2012 pool (30,533 rows), test 2013-2018 per year; FPTM 10 seeds, XGBoost, logistic regression | PASS (amended criterion; FAIL under the original) | Per-year F1 equals tm-cyber-lab b5-apigraph to two decimals for FPTM and XGBoost (2013: 84.24 ± 2.01 and 83.29). Efficiency: FPTM 4.3e-14, logistic regression 3.6e-15, XGBoost 7.8e-8 against a float64 sum of the same trees (amended tolerance 1e-6); 3.9e-5 against XGBoost's float32 margin (original tolerance 1e-5). |
