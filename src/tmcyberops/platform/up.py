"""Start the whole platform: n8n (Docker), Dagster (UI, daemon and replay clock) and the dashboard.

    uv run tmco-up              start, or continue a replay that was stopped
    uv run tmco-up --reset      start a new replay (clears the state; trained models are retrained)
    uv run tmco-up --no-n8n     without Docker

Ctrl+C stops Dagster and the dashboard; n8n keeps running (`docker compose -f ops/docker-compose.yml down`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tmcyberops.paths import REPO
from tmcyberops.platform import config, state

COMPOSE = REPO / "ops" / "docker-compose.yml"
FLOW = REPO / "ops" / "n8n" / "tmco-alerts.json"
FLOW_ID = "tmcoAlertsFlow01"
N8N_CONTAINER = "tmco-n8n-1"


def start_n8n() -> str:
    docker = shutil.which("docker")
    if not docker:
        return "n8n not started: docker not found"
    if subprocess.run([docker, "info"], capture_output=True).returncode != 0:
        return "n8n not started: Docker is not running (start Docker Desktop, then rerun)"
    r = subprocess.run([docker, "compose", "-f", str(COMPOSE), "up", "-d"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"n8n failed: {r.stderr.strip()[-400:]}"
    return "n8n running; " + install_flow(docker)


def install_flow(docker: str) -> str:
    """Import and publish ops/n8n/tmco-alerts.json when it changed since the last start."""
    stamp = config.home() / "n8n-flow.sha256"
    digest = hashlib.sha256(FLOW.read_bytes()).hexdigest()
    if stamp.exists() and stamp.read_text() == digest:
        return "flow up to date"
    env = os.environ | {"MSYS_NO_PATHCONV": "1"}
    steps = [
        [docker, "cp", str(FLOW), f"{N8N_CONTAINER}:/tmp/tmco-alerts.json"],
        [docker, "exec", N8N_CONTAINER, "n8n", "import:workflow", "--input=/tmp/tmco-alerts.json"],
        [docker, "exec", N8N_CONTAINER, "n8n", "publish:workflow", f"--id={FLOW_ID}"],
    ]
    for _ in range(30):  # n8n needs a few seconds after the container starts
        if (
            subprocess.run(
                [docker, "exec", N8N_CONTAINER, "wget", "-qO-", "http://localhost:5678/healthz"],
                capture_output=True,
            ).returncode
            == 0
        ):
            break
        time.sleep(1)
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return f"flow not installed: {(r.stderr or r.stdout).strip()[-300:]}"
    stamp.write_text(digest)
    return "flow installed"


def fail_stale_runs(dagster_home: Path) -> int:
    """Mark runs left 'in progress' by a platform that was stopped as failed; nothing runs before
    Dagster starts, and the replay clock waits while any run looks active."""
    os.environ["DAGSTER_HOME"] = str(dagster_home)
    import dagster as dg

    with dg.DagsterInstance.get() as inst:
        active = [
            dg.DagsterRunStatus.QUEUED,
            dg.DagsterRunStatus.NOT_STARTED,
            dg.DagsterRunStatus.STARTING,
            dg.DagsterRunStatus.STARTED,
        ]
        runs = inst.get_runs(filters=dg.RunsFilter(statuses=active))
        for run in runs:
            inst.report_run_failed(run, message="the platform was stopped while this run was in progress")
    return len(runs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tmco-up", description=__doc__.split("\n\n")[0])
    ap.add_argument("--reset", action="store_true", help="start a new replay")
    ap.add_argument("--no-n8n", action="store_true", help="do not start n8n")
    args = ap.parse_args(argv)

    cfg = config.load()
    home = config.home()
    if args.reset:
        state.reset()
        for sub in ("months", "models"):
            shutil.rmtree(home / sub, ignore_errors=True)
    rid = state.init()
    dagster_home = home / "dagster_home"
    dagster_home.mkdir(exist_ok=True)
    (dagster_home / "dagster.yaml").touch()
    stale = fail_stale_runs(dagster_home)

    env = os.environ | {"DAGSTER_HOME": str(dagster_home), "PYTHONUTF8": "1"}
    svc = cfg["services"]
    n8n = "n8n skipped" if args.no_n8n else start_n8n()
    dashboard = Path(__file__).with_name("dashboard.py")
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "dagster", "dev", "-m", "tmcyberops.platform.definitions", "-p", str(svc["dagster_port"])],
            env=env,
        ),
        subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", str(dashboard),
                "--server.port", str(svc["dashboard_port"]), "--server.headless", "true",
                "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer",
                # dark navy theme; the chart colours have matching dark steps
                "--theme.base", "dark", "--theme.backgroundColor", "#0b1220",
                "--theme.secondaryBackgroundColor", "#131d33", "--theme.textColor", "#e6eaf2",
                "--theme.primaryColor", "#7f9cc7",
            ],
            env=env,
        ),
    ]  # fmt: skip
    print(
        f"replay {rid}{f' ({stale} stale runs marked failed)' if stale else ''}\n"
        f"  dashboard  http://localhost:{svc['dashboard_port']}\n"
        f"  dagster    http://localhost:{svc['dagster_port']}\n"
        f"  n8n        http://localhost:5678  ({n8n})\n"
        "Ctrl+C stops Dagster and the dashboard.",
        flush=True,
    )
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
