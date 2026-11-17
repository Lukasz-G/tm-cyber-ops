# tm-cyber-ops

A monitoring platform for malware detectors, built on two methods from
[tm-cyber-lab](https://github.com/Lukasz-G/tm-cyber-lab):

- exact Shapley attributions for Fuzzy-Pattern Tsetlin machines, computed in closed form;
- attribution drift that raises an alert only when it exceeds a noise floor, the distance expected
  between random samples of the same data.

Android malware data (APIGraph, 2012-2018) is replayed month by month. Three detectors (a Tsetlin
machine, XGBoost and logistic regression) score each month, are evaluated when the month's labels
arrive, and learn in three ways side by side. A dashboard shows the replay as it runs; retraining waits
for a human approval in n8n; language models write a short summary of each month.

**Status:** work in progress. The APIGraph replay runs end to end. Four further datasets are planned.
The methods and their evidence are in tm-cyber-lab; every result here is a scoped demonstration
(dataset, period, size).

![The dashboard replaying APIGraph month by month](docs/media/dashboard.gif)

The dashboard in playback mode over APIGraph, January 2013 to April 2016 (the replay was still
running). It shows F1 per detector and version at the max-F1 cut-off chosen on held-out 2012 apps,
the gain from updating, attribution drift against its noise floor, and the month report.

## What one month does

```mermaid
flowchart LR
    data[("APIGraph month<br/>Drebin features")] --> scores
    subgraph dagster["Dagster: one run per month"]
        scores["scores<br/>3 detectors x 3 arms"] --> detection["detection<br/>precision, recall, F1"]
        detection --> drift["drift<br/>top-100 attributions<br/>vs noise floor"]
        drift --> alert["alert"]
        alert --> learn["learn<br/>incremental and retrain arms"]
        learn --> summary["summary<br/>template + LLM"]
    end
    summary --> state[("DuckDB state")]
    state --> dashboard["Streamlit dashboard"]
    alert -- "retrain proposed" --> n8n["n8n approval form"]
    n8n -- "approve / reject" --> decision["Dagster job: decision"]
    decision --> state
```

| step | what it does |
|---|---|
| scores | applies pending approval decisions, then scores the month's rows with every detector and arm |
| detection | the labels arrive: precision, recall and F1 at the default rule (margin > 0) and at the max-F1 cut-off chosen on held-out training rows |
| drift | attributions of 500 rows of the month; the set of the 100 features with the largest mean absolute attribution is compared with the same set on held-out training rows (Jaccard distance). The noise floor is the 95th percentile of that distance between 1,200 random draws of held-out rows |
| alert | raised when the distance stays above its floor for two months, or when F1 falls 15 points below its held-out value |
| learn | the incremental arm learns the month; the retrain arm proposes a candidate while its alert is active |
| summary | a template text per detector and arm, and a summary of the month from each configured language model; every number in a model's draft must occur in the facts it was given |

## Detectors and learning arms

| | frozen | retrain | incremental |
|---|---|---|---|
| what happens after a month | nothing: the model trained on the 2012 pool | on an alert, a candidate is trained on the last 12 labelled months (with the 2012 pool while fewer exist) and proposed; it replaces the model once approved in n8n, or after a 5-minute timeout | the model learns the month's labelled rows |
| Tsetlin machine (FPTM) | | retrained from scratch | 5 more epochs from the saved automata |
| XGBoost | | retrained from scratch | 100 more boosting rounds |
| logistic regression | | retrained from scratch | refit with the L2 penalty centred on the previous weights |

The Tsetlin machine and XGBoost are trained on the same splits, with the same threshold rule and the
same noise-floor protocol, and both are reported. Attributions are exact for the Tsetlin machine
(closed form) and for logistic regression; XGBoost uses interventional TreeSHAP against a stated
background of 100 training rows.

## Running it

Requires [uv](https://docs.astral.sh/uv/), Julia 1.11, and Docker for n8n (optional).

```
uv sync
julia --project=julia -e "using Pkg; Pkg.instantiate()"
uv run tmco-up            # Dagster, the dashboard and n8n; continues a stopped replay
uv run tmco-up --reset    # starts a new replay
```

| service | address |
|---|---|
| dashboard | http://localhost:8501 |
| Dagster | http://localhost:3000 |
| n8n | http://localhost:5678 |

The APIGraph features are fetched and prepared by
`uv run python -m tmcyberops.datasets.apigraph download` and `... prepare` (see
[the dataset card](datasets/apigraph/DATASET_CARD.md)). Language-model summaries are optional: list the models in
`ops/platform.yaml` and put their keys in an untracked `.env`. The layer is provider-agnostic
(LangGraph with LiteLLM). Details: [docs/platform.md](docs/platform.md).

## Demonstrations

Each demonstration has a run script whose header states the question, scope, arms and pass criterion,
written before the run. Failures stay in the table. Full rows: [demos/INDEX.md](demos/INDEX.md).

| demonstration | question | verdict |
|---|---|---|
| [apigraph-repro](demos/apigraph-repro/) | do the detectors reproduce tm-cyber-lab's per-year F1 on APIGraph, and do the attributions sum to the margin? | PASS under an amended XGBoost tolerance; FAIL under the original |
| [apigraph-drift-floor](demos/apigraph-drift-floor/) | does month-to-month attribution drift of fixed models stay inside a same-month noise floor? | PASS |
| [apigraph-drift-reference](demos/apigraph-drift-reference/) | is the drift alarm calibrated at a fixed class mix, and how far does each month move from the training year? | FAIL: the control alerts at the expected rate for 5 of 12 models only |

## Datasets

Features only. Each dataset is fetched from its publisher by a download script and is never
redistributed here. Each has a card in `datasets/<name>/DATASET_CARD.md`.

| dataset | task | status |
|---|---|---|
| APIGraph (Drebin features, Chen et al. 2023) | Android malware detection, 2012-2018 | on the platform |
| BODMAS | Windows PE malware detection, 2019-2020 | next |
| AnoShift (Kyoto-2006+) | network intrusion detection, 2006-2015 | planned |
| PhreshPhish | phishing web pages | planned |
| BETH | host process-event anomalies | planned |

## Roadmap

1. More datasets on the platform: BODMAS, then AnoShift.
2. Agents with tools: alert triage, and a review of each retraining proposal before the approval form.
3. A comparison of language models (Claude, Kimi, Qwen) on the same month summaries.
4. The Tsetlin machine served to agents as an MCP tool.

## Implementation notes

The Tsetlin machine is `TMCore` from [tm-lab](https://github.com/Lukasz-G/tm-lab), pinned to a commit in
`julia/Project.toml`. Python writes binarised feature matrices in the `.tmx` format
([docs/matrix-format.md](docs/matrix-format.md)) and Julia reads them. Tests: `uv run pytest`.

## Licence

MIT. Dataset licences are given on each dataset card.
