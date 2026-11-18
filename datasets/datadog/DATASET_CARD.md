# Datadog malicious software packages (PyPI) with a matched benign set

## Source

Malicious side: the [Malicious Software Packages Dataset](https://github.com/DataDog/malicious-software-packages-dataset)
by Datadog Security Labs, pinned to commit `72c723be547033c558244520524b117c8d71663b` (2026-11-05). Only
`samples/pypi/malicious_intent` is used: packages published to deliver malware, as opposed to compromised
versions of benign packages. Each sample is an encrypted zip (password `infected`) holding the unpacked
sdist and the package's PyPI metadata (`package_info-*.json`).

Benign side: built here. Random PyPI projects (a seeded draw from PyPI's project index, excluding every
name in Datadog's `samples/pypi/manifest.json`) with an sdist uploaded in the same quarter as a malicious
package, matched to it on sdist size (two benign packages per malicious package, nearest in log size, at
most one release per project).

## Licence

The Datadog dataset is released under Apache-2.0; cite it as Datadog Security Labs, *Open-Source Dataset
of Malicious Software Packages* (2023), github.com/datadog/malicious-software-packages-dataset. Benign
packages keep their own licences. This repository redistributes neither side:
`tmcyberops.datasets.datadog download` and `... benign` fetch them, and every file is checked against
its git hash (malicious) or PyPI's SHA-256 (benign).

## Handling

The malicious packages are live malware. They stay in the encrypted zips as published. Their files are
read into memory and parsed as text by Python's tokenizer; nothing is unpacked to disk, installed,
imported or run. The dashboard shows source with URLs, IP addresses, e-mail addresses and long encoded
strings masked.

## Content used

- Dates: the sdist's upload time on PyPI, from the metadata in each zip (the discovery date in Datadog's
  file names when the metadata is missing). Benign packages are dated the same way.
- Period: packages published 2022-10 to 2026-06, grouped in quarters.
- Training: packages published before 2023-07; the rest are test packages, scored per quarter.

## Preprocessing

`tmcyberops.datasets.datadog prepare` (module `tmcyberops.datasets.codefeatures`) tokenises every `.py`
file and keeps, per package, binary features with the file and line of each occurrence: modules imported,
names and dotted chains called (with import aliases resolved: `b.b64decode` after `import base64 as b` is
`base64.b64decode`), dotted chains not called, other identifiers, and classes of string literals (URL,
IP address, webhook, file host, base64, hex, Windows path, browser data, wallet, shell command,
executable, secrets, Discord). Features are kept on the training packages only: present in at least 10,
then the 1,000 with the largest chi-squared statistic against the label.

## Known biases

- Selection: Datadog found most packages with one rule set (GuardDog), so a detector can relearn those
  rules. Results are about this set, not about PyPI malware in general.
- Size and form: many malicious packages are a single `setup.py`; size matching reduces, but does not
  remove, form differences such as `setup.py` against `pyproject.toml`-only packages.
- Label noise: random live PyPI projects may include a few malicious ones not yet found.
- Quarters 2023Q3 to 2024Q1 hold few malicious packages; their F1 is unstable.
