# tm-cyber-ops

Two methods from [tm-cyber-lab](https://github.com/Lukasz-G/tm-cyber-lab), applied to security-operations
tasks on public datasets:

- exact Shapley attributions for Fuzzy-Pattern Tsetlin machines, computed in closed form;
- attribution drift measured against the spread between models retrained on the same month (the noise floor).

A Tsetlin machine (TM) and XGBoost are trained on the same splits for every task, with the same threshold
rule and the same noise-floor protocol, and both are reported. Each result is a scoped demonstration
(dataset, period, size). The methods and their evidence are in tm-cyber-lab.

**Status:** under construction. No results yet. `demos/INDEX.md` will list each demonstration and its
verdict.

## Datasets

Features only. Each dataset is fetched from its publisher by a download script and is never redistributed
here. Each has a card in `datasets/<name>/DATASET_CARD.md`.

| dataset | task | status |
|---|---|---|
| APIGraph (Drebin features, Chen et al. 2023) | Android malware detection, 2012-2018 | adapter in progress |
| BODMAS | Windows PE malware detection | planned |
| AnoShift (Kyoto-2006+) | network intrusion detection, ten years | planned |
| PhreshPhish | phishing web pages | planned |
| BETH | host process-event anomalies | planned |

## Set-up

Requires [uv](https://docs.astral.sh/uv/) and Julia 1.11.

```
uv sync
julia --project=julia -e "using Pkg; Pkg.instantiate()"
uv run pytest
```

The Tsetlin machine is `TMCore` from [tm-lab](https://github.com/Lukasz-G/tm-lab), pinned to a commit in
`julia/Project.toml`. Python writes binarised feature matrices in the `.tmx` format (`docs/matrix-format.md`)
and Julia reads them.

## Licence

MIT. Dataset licences are given on each dataset card.
