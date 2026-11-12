# The `.tmx` matrix format

Python prepares datasets and Julia trains the Tsetlin machine. They exchange binarised feature matrices
as `.tmx` files, with Arrow sidecars for row metadata and feature names. The format is the one used in
[tm-cyber-lab](https://github.com/Lukasz-G/tm-cyber-lab) (`docs/matrix-format.md` there).

The bit layout on disk is the layout TMCore's `TMInput` uses in memory, so the Julia reader copies words
and does no bit-level conversion. A fixed header followed by packed words can be read in any language
in a few lines.

## Layout

Little-endian throughout. One file holds one matrix.

Header, 64 bytes:

| offset | size | field | value |
|---|---|---|---|
| 0 | 8 | magic | ASCII `TMCYBERX` |
| 8 | 4 | version | `UInt32` = 1 |
| 12 | 4 | flags | `UInt32` = 0, reserved |
| 16 | 8 | n_rows | `UInt64` |
| 24 | 8 | n_cols | `UInt64`, feature bits per row |
| 32 | 32 | reserved | zero |

Body: `n_rows * ceil(n_cols / 64)` `UInt64` words, row-major. Feature `i` (1-based) sits at bit
`(i - 1) % 64` of word `(i - 1) ÷ 64 + 1` of its row, least significant bit first. Padding bits in a
row's last word are zero; TMCore requires its clause masks to be zero there as well, so non-zero padding
would change training.

In numpy, with `rows` a bool array padded on the right to a multiple of 64 columns:

```python
np.packbits(rows, axis=1, bitorder="little").view("<u8")
```

## Sidecars

- `<name>.meta.arrow`: one row per matrix row, same order. Required columns `label` (bool, true =
  malicious), `year` (int16), `month` (int8, 0 for a pool that spans months), `split` (string). Datasets
  add their own columns.
- `features.arrow`: one row per matrix column, in column order, with a `name` column.

## Invariants

1. `n_cols` is the true feature count; only zero padding follows it within a row's last word.
2. Row order is identical in `.tmx`, `.meta.arrow` and any file derived from them.
3. `features.arrow` has exactly `n_cols` rows.
4. A file is never rewritten in place; a subset or resample is a new file.

`tests/test_tmx.py` checks the Python writer against the Julia reader bit by bit.
