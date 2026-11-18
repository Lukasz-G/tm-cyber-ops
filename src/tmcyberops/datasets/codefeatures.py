"""Binary features from Python source read as text: tokens, never imported or run.

Each .py file of a package goes through Python's `tokenize` (a regex scan for files it cannot
tokenise). Features, each with the (file, line) of every occurrence:

* `import:<module>`  modules imported; `import a.b` gives import:a and import:a.b
* `call:<name>`      a name or dotted chain that is called: call:exec, call:os.system
* `attr:<chain>`     a dotted chain that is not called: attr:sys.platform
* `name:<id>`        any other identifier
* `str:<class>`      string literals by class: url, ip, webhook, filehost, base64, hex, winpath,
                     browser, wallet, shell, executable, secrets, discord

Features are kept on the training period only: present in at least `min_df` training packages, then
the `k` with the largest chi-squared statistic against the label. `mask` hides URLs, IP addresses,
e-mail addresses and long encoded strings before any source is shown.
"""

from __future__ import annotations

import io
import json
import keyword
import re
import tokenize
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STRING_CLASSES = {
    "url": r"(?:https?|ftp)://",
    "webhook": r"discord(?:app)?\.com/api/webhooks|hooks\.slack\.com|api\.telegram\.org/bot",
    "filehost": r"pastebin\.com|transfer\.sh|ngrok|anonfiles|gofile\.io|filebin|termbin|paste\.ee",
    "ip": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    "base64": r"[A-Za-z0-9+/]{60,}={0,2}",
    "hex": r"(?:\\x[0-9a-fA-F]{2}){8,}|\b[0-9a-fA-F]{64,}\b",
    "winpath": r"(?i)appdata|roaming|\\\\startup|programdata|\\\\microsoft\\\\windows",
    "browser": r"(?i)login data|local state|cookies|chrome|opera|brave|firefox",
    "wallet": r"(?i)exodus|electrum|metamask|wallet",
    "shell": r"(?i)powershell|cmd(?:\.exe)?\s*/c|\bcurl\s|\bwget\s|/bin/(?:ba)?sh|bash\s+-c|chmod\s|nohup|mshta|certutil",
    "executable": r"(?i)\.(?:exe|dll|bat|vbs|ps1|scr)\b",
    "secrets": r"(?i)\.ssh|\.aws|\.bash_history|/etc/passwd|/etc/shadow|\.npmrc|\.pypirc",
    "discord": r"(?i)discord",
}
STRING_RX = {k: re.compile(v) for k, v in STRING_CLASSES.items()}
IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
CHAIN_RX = re.compile(rf"(?<![\w.]){IDENT}(?:\.{IDENT})*(\s*\()?")
QUOTED_RX = re.compile(r"(['\"])(.*?)(?<!\\)\1")
MASKS = [
    (re.compile(r"(?:https?|ftp)://[^\s'\"<>)]+"), "[URL]"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "[IP]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"[A-Za-z0-9+/_-]{40,}={0,2}"), "[ENCODED]"),
    (re.compile(r"(?:\\x[0-9a-fA-F]{2}){8,}"), "[ENCODED]"),
    # names in package metadata may belong to real people (or be stolen): never shown
    (re.compile(r"""((?:author|maintainer)\s*=\s*)(['"]).*?\2"""), r"\1\2[NAME]\2"),
]

Occ = dict[str, list[tuple[int, int]]]
warnings.filterwarnings(
    "ignore", category=SyntaxWarning
)  # the tokenizer flags bad escapes in package strings


def mask(text: str) -> str:
    for rx, repl in MASKS:
        text = rx.sub(repl, text)
    return text


def _strings(s: str, line: int, fi: int, occ: Occ) -> None:
    for name, rx in STRING_RX.items():
        if rx.search(s):
            occ.setdefault(f"str:{name}", []).append((fi, line))


def _regex_scan(text: str, fi: int, occ: Occ) -> None:
    for ln, line in enumerate(text.splitlines(), 1):
        for m in QUOTED_RX.finditer(line):
            _strings(m.group(2), ln, fi, occ)
        bare = QUOTED_RX.sub("''", line)
        for m in CHAIN_RX.finditer(bare):
            chain = m.group(0).rstrip("( \t")
            if keyword.iskeyword(chain):
                continue
            kind = "call" if m.group(1) else ("attr" if "." in chain else "name")
            occ.setdefault(f"{kind}:{chain}", []).append((fi, ln))


def scan(text: str, fi: int, occ: Occ) -> None:
    """Adds the features of one file to `occ` (feature -> [(file index, line)])."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)]  # fmt: skip
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        _regex_scan(text, fi, occ)
        return
    fstring = {getattr(tokenize, n) for n in ("FSTRING_MIDDLE", "TSTRING_MIDDLE") if hasattr(tokenize, n)}
    i, n = 0, len(toks)
    aliases: dict[str, str] = {}
    while i < n:
        t = toks[i]
        line = t.start[0]
        if t.type == tokenize.STRING or t.type in fstring:
            _strings(t.string, line, fi, occ)
            i += 1
            continue
        if (
            t.type == tokenize.NAME
            and t.string in ("import", "from")
            and (i == 0 or toks[i - 1].type == tokenize.NEWLINE)
        ):
            # `import a.b as c` / `from a.b import c as d`: modules, and the names they bind (aliases)
            j, base, cur, alias, bound = i + 1, "", "", None, []
            while j < n and toks[j].type != tokenize.NEWLINE:
                s = toks[j].string
                if s == "import" and t.string == "from":
                    base, cur = cur, ""
                elif s in (",", "(", ")"):
                    bound.append((base, cur, alias))
                    cur, alias = "", None
                elif s == "as" and j + 1 < n:
                    alias, j = toks[j + 1].string, j + 1
                elif toks[j].type == tokenize.NAME or s == ".":
                    cur += s
                j += 1
            bound.append((base, cur, alias))
            mods = []
            for b, c, a in bound:
                if not c:
                    continue
                full = f"{b}.{c}".strip(".") if b else c.strip(".")
                mods.append(full)
                if a:
                    aliases[a] = full
                elif b:
                    aliases[c] = full
            for m in mods:
                parts = [p for p in m.split(".") if p]
                for k in sorted({1, len(parts) - 1, len(parts)} - {0}):
                    occ.setdefault("import:" + ".".join(parts[:k]), []).append((fi, line))
            i = j
            continue
        if t.type == tokenize.NAME and not keyword.iskeyword(t.string):
            if i > 0 and toks[i - 1].string == ".":  # the tail of a chain after a call or index
                i += 1
                continue
            parts, j = [t.string], i + 1
            while j + 1 < n and toks[j].string == "." and toks[j + 1].type == tokenize.NAME:
                parts.append(toks[j + 1].string)
                j += 2
            if parts[0] in aliases:  # b.b64decode after `import base64 as b` -> base64.b64decode
                parts = aliases[parts[0]].split(".") + parts[1:]
            chain = ".".join(parts)
            called = j < n and toks[j].string == "("
            kind = "call" if called else ("attr" if len(parts) > 1 else "name")
            occ.setdefault(f"{kind}:{chain}", []).append((fi, line))
            i = j
            continue
        i += 1


def package_features(row: dict) -> tuple[list[str], Occ]:
    """(file names, feature -> [(file index, line)]) for one package."""
    from tmcyberops.datasets.datadog import python_files

    files = python_files(row)
    names = sorted(files)
    occ: Occ = {}
    for fi, name in enumerate(names):
        scan(files[name], fi, occ)
    return names, occ


def _one(row: dict) -> tuple[list[str], Occ]:
    try:
        return package_features(row)
    except Exception:
        return [], {}


def select(present: list[set], y: np.ndarray, train: np.ndarray, k: int, min_df: int) -> list[str]:
    """The k features with the largest chi-squared against the label, among those present in at least
    min_df training packages; training rows only."""
    pos, neg = Counter(), Counter()
    for s, label, tr in zip(present, y, train, strict=True):
        if tr:
            (pos if label else neg).update(s)
    n1, n0 = int(np.sum(y[train] == 1)), int(np.sum(y[train] == 0))
    scored = []
    for f in set(pos) | set(neg):
        a, b = pos[f], neg[f]
        if a + b < min_df:
            continue
        c, d = n1 - a, n0 - b
        n = a + b + c + d
        den = (a + b) * (c + d) * (a + c) * (b + d)
        chi2 = n * (a * d - b * c) ** 2 / den if den else 0.0
        scored.append((chi2, f))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [f for _, f in scored[:k]]


def prepare(k: int = 1000, min_df: int = 10, train_end: str = "2023-07-01", workers: int = 8) -> None:
    from tmcyberops.datasets.datadog import packages, root

    rows = packages()
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_one, rows, chunksize=16))
    keep = [i for i, (names, _) in enumerate(out) if names]
    rows = [rows[i] for i in keep]
    out = [out[i] for i in keep]
    y = np.array([r["label"] for r in rows], np.int8)
    train = np.array([r["date"] < train_end for r in rows], bool)
    feats = select([set(o) for _, o in out], y, train, k, min_df)
    col = {f: j for j, f in enumerate(feats)}
    X = np.zeros((len(rows), len(feats)), np.uint8)
    prep = root() / "prepared"
    prep.mkdir(parents=True, exist_ok=True)
    with open(prep / "occurrences.jsonl", "w", encoding="utf-8") as f:
        for i, (r, (names, occ)) in enumerate(zip(rows, out, strict=True)):
            sel = {col[ft]: v for ft, v in occ.items() if ft in col}
            X[i, list(sel)] = 1
            f.write(
                json.dumps({"id": r["id"], "files": names, "occ": {str(c): v for c, v in sel.items()}}) + "\n"
            )
    for r, (names, _) in zip(rows, out, strict=True):
        r["n_files"] = len(names)
    np.save(prep / "X.npy", X)
    (prep / "packages.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (prep / "features.json").write_text(json.dumps(feats, indent=0), encoding="utf-8")
    print(f"prepared {len(rows)} packages ({int(y.sum())} malicious), {len(feats)} features, "
          f"{int(train.sum())} in training (before {train_end}); {len(keep)} of {len(packages())} had .py files")  # fmt: skip


def load() -> tuple[np.ndarray, list[dict], list[str]]:
    from tmcyberops.datasets.datadog import root

    prep = root() / "prepared"
    X = np.load(prep / "X.npy")
    rows = json.loads((prep / "packages.json").read_text(encoding="utf-8"))
    feats = json.loads((prep / "features.json").read_text(encoding="utf-8"))
    return X, rows, feats


def occurrences(ids: set[str] | None = None) -> dict[str, dict]:
    from tmcyberops.datasets.datadog import root

    out = {}
    for line in (root() / "prepared" / "occurrences.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if ids is None or r["id"] in ids:
            out[r["id"]] = r
    return out


def occurrence_index() -> dict[str, int]:
    """{package id: byte offset of its line in occurrences.jsonl}, for reading one package at a time."""
    from tmcyberops.datasets.datadog import root

    index, pos = {}, 0
    with open(root() / "prepared" / "occurrences.jsonl", "rb") as f:
        for raw in f:
            pid = raw[8 : raw.index(b'"', 8)].decode("utf-8")  # each line starts with {"id": "
            index[pid] = pos
            pos += len(raw)
    return index


def occurrence(pid: str, index: dict[str, int]) -> dict:
    from tmcyberops.datasets.datadog import root

    with open(root() / "prepared" / "occurrences.jsonl", "rb") as f:
        f.seek(index[pid])
        return json.loads(f.readline())


def line_heat(phi: np.ndarray, occ: dict, n_files: int) -> tuple[list[dict], float]:
    """Per file, {line: heat}: each present feature's attribution split equally over its occurrences.
    Returns ([{line: heat} per file], the attribution of features with no occurrence: absent ones)."""
    heat = [dict() for _ in range(n_files)]
    drawn = 0.0
    for c, places in occ.items():
        v = float(phi[int(c)])
        drawn += v
        share = v / len(places)
        for fi, ln in places:
            heat[fi][ln] = heat[fi].get(ln, 0.0) + share
    return heat, float(phi.sum()) - drawn


if __name__ == "__main__":
    print(Path(__file__).name)
