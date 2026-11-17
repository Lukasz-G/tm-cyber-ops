# The monitoring platform

APIGraph replayed month by month, 2013-01 to 2018-12. FPTM, XGBoost and logistic regression run in
three learning arms each. Dagster runs the months, a Streamlit dashboard shows the results as they
arrive, n8n carries alerts and approvals, and language models can summarise each month.

```
uv run tmco-up            # start everything, or continue a stopped replay
uv run tmco-up --reset    # start a new replay
```

| service | address | what it shows |
|---|---|---|
| dashboard | http://localhost:8501 | F1 per month, drift from the reference, features entering and leaving the top 100, summaries, alerts, approvals, model versions |
| Dagster | http://localhost:3000 | the set-up run, one run per month, the replay clock (sensor `replay_clock`) |
| n8n | http://localhost:5678 | the alert flow and its approval forms (Docker; create the local owner account on first visit to edit the flow) |

Set `TMCO_DATA` when the APIGraph files are not under `data/` (see datasets/apigraph/DATASET_CARD.md).
Configuration lives in `ops/platform.yaml`; state, models and Dagster's run storage in `work/platform/`.

## One month

| step | what it does |
|---|---|
| scores | applies approval decisions, then computes the margin of every model and arm on the month's rows |
| detection | the month's labels arrive: precision, recall and F1 at the default rule (margin > 0) and at the max-F1 cut-off chosen on the model's held-out rows |
| drift | attributions of 500 rows of the month (450 clean, 50 malware) against the model's background; distance of their top-100 feature set from the model's reference rows and from the previous month |
| alert | an alert is raised when the distance from the reference stays above its floor for two months, or when F1 at the held-out cut-off falls 15 points below its held-out value; raised alerts go to n8n |
| learn | the incremental arm learns the month; the retrain arm proposes a candidate while its alert is active |
| summary | one template text per model and arm, and one summary of the month per configured language model |

The replay clock starts the next month once the previous one has finished (at most every
`replay.interval_seconds`). Pause the sensor in Dagster to stop the replay. A failed month stops the
replay until it is re-executed from the Dagster UI.

## Learning arms

| arm | after month t |
|---|---|
| frozen | nothing: the model trained on the 2012 pool |
| retrain | while the arm's alert is active, a candidate is trained from scratch on the last 12 labelled months (with the 2012 pool while fewer than 12 exist) and proposed; n8n sends an approval form, and the candidate replaces the current model when approved, or by itself after `retrain.approval_timeout_minutes` |
| incremental | FPTM trains 5 more epochs from its saved automata on the month's rows; XGBoost adds 100 boosting rounds on them; logistic regression is refitted on them with the L2 penalty centred on its previous weights |

A retrained model gets its own held-out rows (10% of its training rows, stratified), cut-off, background
and reference with its own floor. An incremental model keeps the 2012 held-out rows: its cut-off is
chosen again on them and its 500 reference rows are explained again, while the reference floor stays
the 2012 model's.

## Drift: what the lines mean

The reference distance compares the month's top-100 set with that of 500 held-out rows from the
model's training data. Its floor is the 95th percentile of the same distance for 1,200 draws of other
held-out rows, so a month above the floor differs from the training data more than 500-row samples of
the training data differ among themselves. The distance from the previous month has no calibrated
floor: in [demos/apigraph-drift-reference](../demos/apigraph-drift-reference/) the month-to-month
alarm at k=100 did not reach a stable false-alarm rate.

## Language models

Summaries come from a LangGraph graph with one node per configured model. A model given as
`<provider>/<model>` is called through LiteLLM, so any provider LiteLLM supports can be used; a model
given as `claude-cli/<model>` runs the Claude Code CLI headless, so a Claude subscription can stand in
until API keys are set. List the models in `llm.summarise_with` and put keys (or `CLAUDE_CLI`, the path
to the CLI) in `tm-cyber-ops/.env` (ignored by git), for example `ANTHROPIC_API_KEY=...`. Each draft is checked before it is shown: at most `llm.max_lines`
lines, and every number in it must occur in the facts the model received. Without keys the dashboard
shows the template summaries only.

## Approvals

The n8n flow (`ops/n8n/tmco-alerts.json`, installed by `tmco-up`) acknowledges every alert. For a
retrain proposal it answers with the link of an approval form, which the dashboard shows next to the
candidate. Submitting the form launches the Dagster job `decision`, which records the answer; the
next month applies it.
