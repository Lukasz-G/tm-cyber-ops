# APIGraph (Drebin features)

## Source

Drebin feature vectors for the APIGraph Android sample set, released by Yizheng Chen, Zhoujie Ding and
David Wagner with *Continuous Learning for Android Malware Detection* (USENIX Security 2023,
[arXiv:2302.04332](https://arxiv.org/abs/2302.04332)); code at
[github.com/wagner-group/active-learning](https://github.com/wagner-group/active-learning). The archive is
linked from that repository's README and hosted on
[Google Drive](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view).

The sample set is the one defined by Zhang et al., *Enhancing State-of-the-art Classifiers with API
Semantics to Detect Evolved Android Malware* (APIGraph, ACM CCS 2020). The APIGraph authors released
sample hashes only, so these features are a third party's extraction of that set, not the APIGraph
authors' own.

## Licence

Neither the archive nor the active-learning repository states a licence or terms of use. This repository
does not redistribute the archive or any part of it: `tmcyberops.datasets.apigraph download` fetches it
from the authors' link and checks its SHA-256. Cite both papers when using it.

## Content used

The archive (`drebin_features.zip`, 314,350,441 bytes, SHA-256
`827f36fc5affd58dd31b6393042d92cbe3bae012290a03b6fcccc6355266384d`) holds two datasets. Only
`gen_apigraph_drebin` is extracted; the AndroZoo part is not used.

- `2012-01to2012-12_selected.npz`: the training pool, all of 2012.
- `YYYY-MM_selected.npz`: one test set per month, 2013-01 to 2018-12.
- `2012-01to2012-12_selected_training_features.json`: feature name -> column index, 1,159 binary
  Drebin features (permissions, intents, API calls, URLs, components), selected on the 2012 pool.

## Preprocessing

`tmcyberops.datasets.apigraph prepare` writes one `.tmx` matrix (docs/matrix-format.md) and Arrow
metadata per file, plus `features.arrow`. No feature is changed or dropped.

- Label: `y_train` holds family indices; 0 is benign and every positive value is a malware family.
  `label = y_train > 0`. `y_train == 1` would select one family only.
- Family names are stored in the archive as pickled object arrays. They are not read
  (`np.load(..., allow_pickle=False)`), so loading the archive cannot execute code.
- Protocol: train on the 2012 pool, test on each later month, as the source publishes it. Results are
  reported per test year or per month, never pooled across years.

## Size

Width 1,159 features. Counts from `prepare`:

| split | months | rows | malware | malware share |
|---|---|---|---|---|
| train (2012 pool) | 12 | 30,533 | 3,061 | 10.0% |
| 2013 | 12 | 48,568 | 4,854 | 10.0% |
| 2014 | 12 | 58,485 | 5,809 | 9.9% |
| 2015 | 12 | 57,452 | 5,508 | 9.6% |
| 2016 | 12 | 56,036 | 5,324 | 9.5% |
| 2017 | 12 | 27,312 | 2,465 | 9.0% |
| 2018 | 12 | 41,929 | 3,783 | 9.0% |

The 73 `.tmx` matrices `prepare` writes are byte-identical to those tm-cyber-lab built from the same
archive.
