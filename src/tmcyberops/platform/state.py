"""The platform's shared state: one DuckDB file that every Dagster step writes and the dashboard reads.

DuckDB lets one process write at a time, so every access opens a short-lived connection and retries
while another process holds the lock.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from tmcyberops.platform import config

SCHEMA = """
create table if not exists meta (key varchar primary key, value varchar);
create table if not exists models (
    model varchar, arm varchar, version integer, trained_on varchar, rows integer, cutoff double,
    heldout_precision double, heldout_recall double, heldout_f1 double, heldout_f1_cutoff double,
    reference_floor_p95 double, status varchar, proposed_month varchar, decided_by varchar,
    decided_ts timestamp, ts timestamp, primary key (model, arm, version));
create table if not exists decisions (
    ts timestamp, model varchar, arm varchar, version integer, decision varchar, decided_by varchar);
create table if not exists metrics (
    month varchar, model varchar, arm varchar, version integer, rule varchar, cutoff double, rows integer,
    malware integer, flagged_pct double, precision double, recall double, f1 double,
    primary key (month, model, arm, rule));
create table if not exists drift (
    month varchar, model varchar, arm varchar, version integer, versus varchar, k integer,
    distance double, floor_p95 double, above boolean, primary key (month, model, arm, versus, k));
create table if not exists features (
    month varchar, model varchar, arm varchar, kind varchar, position integer, feature varchar,
    importance double, rank integer, reference_rank integer, primary key (month, model, arm, kind, position));
create table if not exists summaries (
    month varchar, model varchar, arm varchar, author varchar, text varchar, ts timestamp,
    primary key (month, model, arm, author));
create table if not exists events (
    ts timestamp, month varchar, model varchar, arm varchar, kind varchar, message varchar, delivery varchar,
    link varchar);
create table if not exists alert_state (
    model varchar, arm varchar, kind varchar, active boolean, primary key (model, arm, kind));
create table if not exists months (month varchar primary key, ts timestamp);
"""


def db_path() -> Path:
    return config.home() / "state.duckdb"


@contextmanager
def connect(read_only: bool = False, tries: int = 300) -> Iterator[duckdb.DuckDBPyConnection]:
    for attempt in range(tries):
        try:
            con = duckdb.connect(str(db_path()), read_only=read_only)
            break
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or attempt == tries - 1:
                raise
            time.sleep(0.1)
    try:
        yield con
    finally:
        con.close()


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def init() -> str:
    """Create the tables if needed; return the replay id (new for a fresh database)."""
    with connect() as con:
        con.execute(SCHEMA)
        row = con.execute("select value from meta where key = 'replay_id'").fetchone()
        if row:
            return row[0]
        rid = uuid.uuid4().hex[:8]
        con.execute("insert into meta values ('replay_id', ?), ('created', ?)", [rid, now().isoformat()])
        return rid


def reset() -> None:
    p = db_path()
    for q in (p, p.with_name(p.name + ".wal")):
        if q.exists():
            q.unlink()


def replay_id() -> str | None:
    if not db_path().exists():
        return None
    with connect(read_only=True) as con:
        row = con.execute("select value from meta where key = 'replay_id'").fetchone()
    return row[0] if row else None


def upsert(table: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    sql = f"insert or replace into {table} ({', '.join(cols)}) values ({', '.join('?' * len(cols))})"
    with connect() as con:
        con.executemany(sql, [[r[c] for c in cols] for r in rows])


def append(table: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    sql = f"insert into {table} ({', '.join(cols)}) values ({', '.join('?' * len(cols))})"
    with connect() as con:
        con.executemany(sql, [[r[c] for c in cols] for r in rows])


def execute(sql: str, params: Sequence | None = None) -> None:
    with connect() as con:
        con.execute(sql, params or [])


def query(sql: str, params: Sequence | None = None) -> pd.DataFrame:
    if not db_path().exists():
        return pd.DataFrame()
    with connect(read_only=True) as con:
        return con.execute(sql, params or []).df()


def done_months() -> set[str]:
    df = query("select month from months")
    return set(df["month"]) if len(df) else set()


def models_ready() -> bool:
    df = query("select count(*) as n from models")
    return bool(len(df)) and int(df["n"][0]) > 0
