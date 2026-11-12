# Julia half of the boundary check (tests/test_tmx.py writes the fixtures and compares).
#
#   julia --project=julia tests/boundary.jl <dir>
#
# Reads each fixture through the code path training uses, Tmx.read_inputs into TMInput, and writes the
# 1-based indices of each row's set bits. An off-by-one or endianness error in the layout would train a
# different model without raising, so the comparison is bit by bit.

include(joinpath(@__DIR__, "..", "julia", "tmx.jl"))
using .Tmx: read_inputs, read_arrow, read_features

row_line(x) = join([i for i in eachindex(x) if x[i]], " ")

function main(dir::AbstractString)
    for stem in split(read(joinpath(dir, "manifest.txt"), String))
        inputs, h = read_inputs(joinpath(dir, "$stem.tmx"))
        meta = read_arrow(joinpath(dir, "$stem.meta.arrow"))
        feats = read_features(joinpath(dir, "$stem.features.arrow"))
        length(inputs) == h.n_rows || error("$stem: read $(length(inputs)) rows, header says $(h.n_rows)")
        length(meta.label) == h.n_rows || error("$stem: metadata rows disagree with the header")
        length(feats) == h.n_cols || error("$stem: $(length(feats)) feature names, $(h.n_cols) columns")
        all(length(x) == h.n_cols for x in inputs) || error("$stem: an input is not n_cols wide")
        if h.n_rows >= 3   # a subset read must agree with the full read
            want = [1, h.n_rows]
            sub, _ = read_inputs(joinpath(dir, "$stem.tmx"); rows=want)
            for (k, r) in enumerate(want)
                row_line(sub[k]) == row_line(inputs[r]) || error("$stem: subset read differs at row $r")
            end
        end
        open(joinpath(dir, "$stem.actual.txt"), "w") do io
            foreach(x -> println(io, row_line(x)), inputs)
        end
    end
end

length(ARGS) == 1 || error("usage: julia --project=julia tests/boundary.jl <dir>")
main(ARGS[1])
