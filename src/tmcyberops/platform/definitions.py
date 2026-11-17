"""Dagster definitions: the 2012 set-up, the monthly pipeline and the replay clock.

    dagster dev -m tmcyberops.platform.definitions        (started by `tmco-up`)

Each month is one partition. Its run executes six assets in order: scores -> detection and drift ->
alert -> learn -> summary. The replay clock (a sensor) starts the next month once the previous one has
finished; pause it in the Dagster UI to stop the replay. A failed month stops the replay until it is
re-executed.
"""

from __future__ import annotations

import dagster as dg

from tmcyberops.platform import config, pipeline, state

CFG = config.load()
MONTHS = config.months(CFG)
months = dg.StaticPartitionsDefinition(MONTHS)


@dg.asset(
    group_name="setup",
    description="Train every model on the 2012 pool; held-out cut-off and drift reference.",
)
def models_2012() -> dg.MaterializeResult:
    state.init()
    rows = pipeline.bootstrap(CFG)
    meta = {
        f"{r['model']} held-out F1": round(r["heldout_f1"], 1) for r in rows if r["arm"] == CFG["arms"][0]
    }
    return dg.MaterializeResult(metadata=meta)


def _month_asset(name: str, deps: list, fn, description: str):
    @dg.asset(name=name, partitions_def=months, deps=deps, group_name="month", description=description)
    def _asset(context) -> dg.MaterializeResult:
        out = fn(CFG, context.partition_key)
        return dg.MaterializeResult(metadata={k: v for k, v in out.items() if v is not None})

    return _asset


scores = _month_asset(
    "scores",
    [models_2012],
    pipeline.step_scores,
    "Apply approval decisions, then margins of every model and arm on the month's rows.",
)
detection = _month_asset(
    "detection",
    [scores],
    pipeline.step_detection,
    "Precision, recall and F1 at the default rule and the held-out cut-off.",
)
drift = _month_asset(
    "drift",
    [scores],
    pipeline.step_drift,
    "Attributions of 500 rows; top-100 distance from the model's reference rows and from the previous month.",
)
alert = _month_asset(
    "alert", [detection, drift], pipeline.step_alert, "Raise or clear alerts; send raised ones to n8n."
)
learn = _month_asset(
    "learn",
    [alert],
    pipeline.step_learn,
    "Incremental arm learns the month; retrain arm proposes a candidate while its alert is active.",
)
summary = _month_asset(
    "summary", [learn], pipeline.step_summary, "A short text per model and arm; marks the month done."
)


@dg.op(
    config_schema={
        "model": str,
        "arm": dg.Field(str, default_value="retrain"),
        "version": int,
        "decision": str,
        "decided_by": dg.Field(str, default_value="n8n form"),
    }
)
def record_decision(context) -> None:
    pipeline.record_decision(**context.op_config)


@dg.job(description="Record an approval decision on a retrain candidate (launched by n8n).")
def decision():
    record_decision()


setup_job = dg.define_asset_job("setup", selection=[models_2012], executor_def=dg.in_process_executor)
month_job = dg.define_asset_job(
    "month",
    selection=dg.AssetSelection.groups("month"),
    partitions_def=months,
    executor_def=dg.in_process_executor,
)


@dg.sensor(
    jobs=[setup_job, month_job],
    minimum_interval_seconds=int(CFG["replay"]["interval_seconds"]),
    default_status=dg.DefaultSensorStatus.RUNNING,
    description="Replay clock: one month at a time, in order.",
)
def replay_clock(context):
    active = context.instance.get_runs(
        filters=dg.RunsFilter(
            statuses=[
                dg.DagsterRunStatus.QUEUED,
                dg.DagsterRunStatus.NOT_STARTED,
                dg.DagsterRunStatus.STARTING,
                dg.DagsterRunStatus.STARTED,
            ]
        )
    )
    if active:
        return dg.SkipReason(f"run {active[0].run_id[:8]} in progress")
    rid = state.init()
    if not state.models_ready():
        return dg.RunRequest(run_key=f"{rid}-setup", job_name="setup")
    done = state.done_months()
    todo = [m for m in MONTHS if m not in done]
    if not todo:
        return dg.SkipReason("replay finished")
    return dg.RunRequest(run_key=f"{rid}-{todo[0]}", job_name="month", partition_key=todo[0])


defs = dg.Definitions(
    assets=[models_2012, scores, detection, drift, alert, learn, summary],
    jobs=[setup_job, month_job, decision],
    sensors=[replay_clock],
)
