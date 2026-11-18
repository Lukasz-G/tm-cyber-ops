"""Malicious PyPI packages from Datadog's dataset, and a matched benign set drawn from PyPI.

    uv run python -m tmcyberops.datasets.datadog download   # malicious-intent zips at a pinned commit
    uv run python -m tmcyberops.datasets.datadog benign     # random PyPI sdists, same quarters, size-matched
    uv run python -m tmcyberops.datasets.datadog prepare    # tokens -> binary features per package

The malicious samples are live malware. They stay in the encrypted zips as published (password
`infected`): files are read into memory and parsed as text by Python's tokenizer. Nothing is unpacked
to disk, installed, imported or run, and nothing from either side is redistributed. Card:
datasets/datadog/DATASET_CARD.md.

Each malicious zip holds the package's PyPI metadata (`package_info-*.json`), so both sides are dated
by the sdist's PyPI upload time and matched on the sdist's size, taken from the same source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tmcyberops.paths import data_dir

GITHUB = "DataDog/malicious-software-packages-dataset"
COMMIT = "72c723be547033c558244520524b117c8d71663b"  # main on 2026-11-05
API = f"https://api.github.com/repos/{GITHUB}"
RAW = f"https://raw.githubusercontent.com/{GITHUB}/{COMMIT}/"
SUBDIR = "samples/pypi/malicious_intent"
PASSWORD = b"infected"
WINDOW = ("2022-10-01", "2026-06-30")  # quarters 2022Q4 .. 2026Q2
MAX_SDIST = 5_000_000
UA = {"User-Agent": "tm-cyber-ops dataset fetcher"}


def root() -> Path:
    return data_dir() / "datadog"


def fetch(url: str, headers: dict | None = None, tries: int = 6) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA | (headers or {}))
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(1.5 * (k + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"could not fetch {url}")


def git_blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def quarter(date: str) -> str:
    return f"{date[:4]}Q{(int(date[5:7]) - 1) // 3 + 1}"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sdist(meta: dict) -> dict | None:
    return next((u for u in meta.get("urls", []) if u.get("packagetype") == "sdist"), None)


# --- malicious side -------------------------------------------------------------------------------


def download(workers: int = 8) -> None:
    """With GITHUB_TOKEN set, blobs come from the GitHub API (fast, 5,000 requests an hour); without
    it, from raw.githubusercontent.com (slow, resets often). Each file is checked against its git hash."""
    out = root() / "malicious"
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("GITHUB_TOKEN", "")
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    listing = json.loads(fetch(f"{API}/contents/samples/pypi?ref={COMMIT}", auth))
    sha = next(e["sha"] for e in listing if e["name"] == "malicious_intent")
    tree = json.loads(fetch(f"{API}/git/trees/{sha}?recursive=1", auth))
    assert not tree.get("truncated"), "tree listing truncated"
    blobs = [e for e in tree["tree"] if e["type"] == "blob" and e["path"].endswith(".zip")]
    (root() / "manifest-pypi.json").write_bytes(fetch(RAW + "samples/pypi/manifest.json"))

    # stored flat by git hash: some paths (version strings of hundreds of characters) exceed Windows' limit
    (root() / "malicious-paths.json").write_text(
        json.dumps({e["sha"]: e["path"] for e in blobs}), encoding="utf-8"
    )

    def one(e: dict) -> str:
        p = out / f"{e['sha']}.zip"
        if p.exists() and git_blob_sha(p.read_bytes()) == e["sha"]:
            return "kept"
        try:
            if token:
                b = fetch(f"{API}/git/blobs/{e['sha']}", auth | {"Accept": "application/vnd.github.raw+json"})
            else:
                b = fetch(RAW + SUBDIR + "/" + e["path"])
        except Exception:
            return "failed"  # kept for the next run, which fetches only what is missing
        if git_blob_sha(b) != e["sha"]:
            return "failed"
        p.write_bytes(b)
        return "new"

    done = Counter()
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, blobs), 1):
            done[r] += 1
            if i % 250 == 0:
                print(f"  {i}/{len(blobs)} {dict(done)}", flush=True)
    print(f"malicious zips: {dict(done)} of {len(blobs)}, commit {COMMIT[:10]}")
    index_malicious()


def index_malicious() -> list[dict]:
    rows = []
    paths = json.loads((root() / "malicious-paths.json").read_text(encoding="utf-8"))
    for sha, rel in sorted(paths.items(), key=lambda kv: kv[1]):
        p = root() / "malicious" / f"{sha}.zip"
        if not p.exists():
            continue
        found = re.search(r"(\d{4}-\d{2}-\d{2})-", rel.rsplit("/", 1)[-1])
        discovered = found.group(1) if found else None
        meta = {}
        with zipfile.ZipFile(p) as z:
            info = next((n for n in z.namelist() if Path(n).name.startswith("package_info")), None)
            if info:
                try:
                    meta = json.loads(z.read(info, pwd=PASSWORD))
                except ValueError:
                    meta = {}
        sd = _sdist(meta) or {}
        published = (sd.get("upload_time_iso_8601") or "")[:10] or None
        date = published or discovered
        name = (meta.get("info") or {}).get("name") or rel.split("/")[0]
        rows.append(dict(
            id=f"m:{sha[:12]}", label=1, source=p.name, path=rel, name=name, version=rel.split("/")[1][:40],
            discovered=discovered, published=published, date=date, size=sd.get("size"),
            quarter=quarter(date) if date else None,
        ))  # fmt: skip
    (root() / "malicious.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print(
        f"indexed {len(rows)} malicious packages; quarters: {sorted(Counter(r['quarter'] for r in rows).items())}"
    )
    return rows


# --- benign side ------------------------------------------------------------------------------------


def _candidates(name: str) -> list[dict]:
    try:
        meta = json.loads(fetch(f"https://pypi.org/pypi/{name}/json", tries=3))
    except Exception:
        return []
    out = []
    for version, files in (meta.get("releases") or {}).items():
        for f in files:
            day = (f.get("upload_time_iso_8601") or "")[:10]
            if (
                f.get("packagetype") == "sdist"
                and not f.get("yanked")
                and WINDOW[0] <= day <= WINDOW[1]
                and 0 < (f.get("size") or 0) <= MAX_SDIST
                and f["filename"].endswith((".tar.gz", ".zip"))
            ):
                out.append(dict(
                    name=meta["info"]["name"], version=version, filename=f["filename"], url=f["url"],
                    sha256=f["digests"]["sha256"], size=f["size"], published=day, date=day, quarter=quarter(day),
                ))  # fmt: skip
    return out


def benign(per_malicious: int = 2, pool_factor: float = 1.5, max_names: int = 90_000, workers: int = 40,
           seed: int = 0) -> None:  # fmt: skip
    mal = json.loads((root() / "malicious.json").read_text(encoding="utf-8"))
    mal = [m for m in mal if m["quarter"] and WINDOW[0] <= m["date"] <= WINDOW[1]]
    need = Counter(m["quarter"] for m in mal)
    need = {q: per_malicious * n for q, n in need.items()}
    known = {norm(n) for n in json.loads((root() / "manifest-pypi.json").read_text(encoding="utf-8"))}
    known |= {norm(m["name"]) for m in mal}

    pool_path = root() / "benign-candidates.jsonl"
    pool: dict[str, list[dict]] = defaultdict(list)
    asked = set()
    if pool_path.exists():
        for line in pool_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            asked.add(r["asked"])
            for c in r["candidates"]:
                pool[c["quarter"]].append(c)
    simple = json.loads(fetch("https://pypi.org/simple/", {"Accept": "application/vnd.pypi.simple.v1+json"}))
    names = [p["name"] for p in simple["projects"] if norm(p["name"]) not in known]
    rng = np.random.default_rng(seed)
    names = [names[i] for i in rng.permutation(len(names))[:max_names]]
    todo = [n for n in names if n not in asked]

    def short() -> dict:
        return {q: n for q, n in need.items() if len({c["name"] for c in pool[q]}) < pool_factor * n}

    print(f"benign: {len(names)} names drawn, {len(asked)} asked before; need {need}", flush=True)
    with ThreadPoolExecutor(workers) as ex, open(pool_path, "a", encoding="utf-8") as log:
        for start in range(0, len(todo), 1000):
            if not short():
                break
            batch = todo[start : start + 1000]
            for name, cands in zip(batch, ex.map(_candidates, batch), strict=True):
                log.write(json.dumps({"asked": name, "candidates": cands}) + "\n")
                for c in cands:
                    pool[c["quarter"]].append(c)
            log.flush()
            print(f"  asked {len(asked) + start + len(batch)}; still short: {short()}", flush=True)

    # match: for each malicious package, the unused candidates of its quarter nearest in log size
    mal_sizes = [m["size"] for m in mal if m["size"]]
    median = float(np.median(mal_sizes))
    used_names, chosen = set(), []
    for q in sorted(need):
        cands = [c for c in pool[q] if norm(c["name"]) not in known]
        rng.shuffle(cands)
        logs = np.array([math.log(c["size"]) for c in cands])
        free = np.ones(len(cands), bool)
        for m in sorted((m for m in mal if m["quarter"] == q), key=lambda m: m["id"]):
            target = math.log(m["size"] or median)
            for _ in range(per_malicious):
                ok = free & np.array([norm(c["name"]) not in used_names for c in cands], bool)
                if not ok.any():
                    break
                i = int(np.argmin(np.where(ok, np.abs(logs - target), np.inf)))
                free[i] = False
                used_names.add(norm(cands[i]["name"]))
                chosen.append(cands[i] | {"matched_to": m["id"]})
    print(f"matched {len(chosen)} benign sdists for {len(mal)} malicious (target {per_malicious} each)")

    out = root() / "benign"
    out.mkdir(parents=True, exist_ok=True)

    def get(c: dict) -> dict | None:
        p = out / c["filename"]
        if not (p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == c["sha256"]):
            try:
                b = fetch(c["url"], tries=4)
            except Exception:
                return None
            if hashlib.sha256(b).hexdigest() != c["sha256"]:
                return None
            p.write_bytes(b)
        return dict(id=f"b:{c['filename']}", label=0, source=c["filename"], **{k: c[k] for k in (
            "name", "version", "published", "date", "size", "quarter", "sha256", "matched_to")})  # fmt: skip

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(get, chosen) if r]
    (root() / "benign.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print(
        f"benign sdists: {len(rows)} of {len(chosen)}; quarters: {sorted(Counter(r['quarter'] for r in rows).items())}"
    )


# --- reading a package, in memory -------------------------------------------------------------------


def python_files(row: dict, max_files: int = 400, max_bytes: int = 1_000_000) -> dict[str, str]:
    """{path inside the package: text} for every .py file; read in memory, never unpacked or run."""
    p = root() / ("malicious" if row["label"] else "benign") / row["source"]
    files: dict[str, str] = {}
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if i.filename.endswith(".py") and not i.is_dir() and i.file_size <= max_bytes:
                    data = z.read(i, pwd=PASSWORD if i.flag_bits & 1 else None)
                    files[i.filename.split("/", 1)[-1]] = data.decode("utf-8", "replace")
                if len(files) >= max_files:
                    break
    else:
        with tarfile.open(p, mode="r:*") as t:
            for m in t:
                if m.isfile() and m.name.endswith(".py") and m.size <= max_bytes:
                    f = t.extractfile(m)
                    if f is not None:
                        files[m.name.split("/", 1)[-1]] = f.read().decode("utf-8", "replace")
                if len(files) >= max_files:
                    break
    return files


def packages() -> list[dict]:
    rows = []
    for side in ("malicious", "benign"):
        rows += json.loads((root() / f"{side}.json").read_text(encoding="utf-8"))
    return [r for r in rows if r["date"] and WINDOW[0] <= r["date"] <= WINDOW[1]]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tmcyberops.datasets.datadog")
    ap.add_argument("step", choices=["download", "index", "benign", "prepare"])
    a = ap.parse_args(argv)
    if a.step == "download":
        download()
    elif a.step == "index":
        index_malicious()
    elif a.step == "benign":
        benign()
    else:
        from tmcyberops.datasets.codefeatures import prepare

        prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
