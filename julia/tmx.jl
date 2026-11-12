# Read the `.tmx` packed binary matrix, the boundary between Python and Julia.
#
# Format: docs/matrix-format.md. The on-disk layout is TMCore's `TMInput` layout, so a row becomes a
# `TMInput` by copying words. Python writes these files (src/tmcyberops/tmx.py); Julia only reads them.
# Ported from tm-cyber-lab (julia/tmx.jl, commit e6f8fb8).

module Tmx

using Arrow
using TMCore: TMInput

export TmxHeader, read_header, read_inputs, read_arrow, read_features

const MAGIC = b"TMCYBERX"
const VERSION = UInt32(1)

struct TmxHeader
    version::UInt32
    flags::UInt32
    n_rows::Int
    n_cols::Int
end

words_per_row(h::TmxHeader) = cld(h.n_cols, 64)

function read_header(io::IO, path::AbstractString="<io>")
    magic = read(io, 8)
    magic == MAGIC || error("$path: bad magic $(String(copy(magic))), expected $(String(copy(MAGIC)))")
    version = read(io, UInt32)
    version == VERSION || error("$path: version $version, this reader reads version $VERSION")
    flags = read(io, UInt32)
    n_rows = Int(read(io, UInt64))
    n_cols = Int(read(io, UInt64))
    read(io, 32)  # reserved
    return TmxHeader(version, flags, n_rows, n_cols)
end

read_header(path::AbstractString) = open(io -> read_header(io, path), path)

"""
    read_inputs(path; rows = nothing) -> (Vector{TMInput}, TmxHeader)

Load a `.tmx` as `TMInput`s. `rows` selects a subset by sorted, unique, 1-based index.
"""
function read_inputs(path::AbstractString;
                     rows::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    open(path, "r") do io
        h = read_header(io, path)
        wpr = words_per_row(h)
        if rows !== nothing
            issorted(rows) || error("rows must be sorted")
            allunique(rows) || error("rows must be unique")
            if !isempty(rows)
                (first(rows) >= 1 && last(rows) <= h.n_rows) || error("rows out of range 1:$(h.n_rows)")
            end
        end
        want = rows === nothing ? (1:h.n_rows) : rows
        base = position(io)
        out = Vector{TMInput}(undef, length(want))
        buf = Vector{UInt64}(undef, wpr)
        prev = 0
        for (k, r) in enumerate(want)
            r == prev + 1 || seek(io, base + (r - 1) * wpr * sizeof(UInt64))
            read!(io, buf)
            chunks = Memory{UInt64}(undef, wpr)
            copyto!(chunks, buf)
            out[k] = TMInput(chunks, h.n_cols)
            prev = r
        end
        return out, h
    end
end

"An Arrow sidecar: row metadata or labels, one row per matrix row, in matrix order."
read_arrow(path::AbstractString) = Arrow.Table(path)

"Feature names, one per matrix column, in column order."
read_features(path::AbstractString) = collect(String.(Arrow.Table(path).name))

end # module Tmx
