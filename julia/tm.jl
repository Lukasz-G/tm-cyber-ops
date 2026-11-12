# Command-line entry to the Tsetlin machine. Python writes `.tmx` matrices and Arrow labels; this
# script trains, scores and attributes, and writes plain files back.
#
#   julia --project=julia -t auto julia/tm.jl fit     <train.tmx> <labels.arrow> <model.tmcore> <config.json>
#   julia --project=julia -t auto julia/tm.jl score   <model.tmcore> <X.tmx> <scores.arrow>
#   julia --project=julia -t auto julia/tm.jl shapley <model.tmcore> <X.tmx> <background.tmx> <phi.f64>
#
# fit: binary TMClassifier over classes [false, true], trained for `epochs` passes, pass e shuffled
#      with MersenneTwister(1000 * seed + e) and `parallel = :none`. This is the loop that produced
#      the tm-cyber-lab APIGraph numbers, so a given seed reproduces them.
# score: columns score_false, score_true (class scores) and pred (TMCore.predict: the higher class
#      score; ties go to false). pred is true exactly when score_true - score_false > 0.
# shapley: exact Shapley values of the margin score_true - score_false against the mean of the
#      background rows, Float64 little-endian, row-major n_rows x n_cols.

include(joinpath(@__DIR__, "tmx.jl"))
include(joinpath(@__DIR__, "shapley.jl"))
using .Tmx: read_inputs, read_arrow
using .Shapley: shapley_mean!
using TMCore
using Arrow, JSON, Random, Printf

const CLASSES = [false, true]

load(path) = open(load_model, path)

function fit(train_path, labels_path, model_path, config_path)
    cfg = JSON.parsefile(config_path)
    X, h = read_inputs(train_path)
    Y = collect(Bool.(read_arrow(labels_path).label))
    length(Y) == length(X) || error("$(length(Y)) labels for $(length(X)) rows")
    seed = Int(cfg["seed"])
    m = TMClassifier(CLASSES, h.n_cols; clauses_per_class=Int(cfg["clauses_per_class"]),
                     T=Int(cfg["T"]), S=Int(cfg["S"]), L=Int(cfg["L"]), LF=Int(cfg["LF"]))
    t = @elapsed for e in 1:Int(cfg["epochs"])
        train!(m, X, Y; rng=MersenneTwister(1000seed + e), parallel=:none)
    end
    open(io -> save_model(io, m), model_path, "w")
    predict(load(model_path), X) == predict(m, X) ||
        error("model reloaded from $model_path predicts differently from the trained model")
    @printf(stderr, "fit: %d rows x %d cols, seed %d, %d epochs in %.1f s\n",
            length(X), h.n_cols, seed, cfg["epochs"], t)
end

function score_cmd(model_path, x_path, out_path)
    m = load(model_path)
    X, _ = read_inputs(x_path)
    s0 = Vector{Int}(undef, length(X)); s1 = similar(s0)
    Threads.@threads for i in eachindex(X)
        s0[i] = score(m, 1, X[i]); s1[i] = score(m, 2, X[i])
    end
    Arrow.write(out_path, (score_false=s0, score_true=s1, pred=predict(m, X)))
end

as_bools(x, w) = Bool[x[i] for i in 1:w]

function shapley_cmd(model_path, x_path, bg_path, out_path)
    m = load(model_path)
    X, h = read_inputs(x_path)
    B, hb = read_inputs(bg_path)
    hb.n_cols == h.n_cols || error("background has $(hb.n_cols) columns, X has $(h.n_cols)")
    w = h.n_cols
    bgs = [as_bools(b, w) for b in B]
    P = zeros(Float64, w, length(X))       # column i holds row i's values: row-major on disk
    t = @elapsed Threads.@threads for i in eachindex(X)
        shapley_mean!(view(P, :, i), m, as_bools(X[i], w), bgs; plus=2, minus=1)
    end
    open(io -> write(io, P), out_path, "w")
    @printf(stderr, "shapley: %d rows x %d background rows in %.1f s on %d threads\n",
            length(X), length(B), t, Threads.nthreads())
end

function main(args)
    isempty(args) && error("usage: tm.jl fit|score|shapley ...")
    cmd, rest = args[1], args[2:end]
    if cmd == "fit" && length(rest) == 4
        fit(rest...)
    elseif cmd == "score" && length(rest) == 3
        score_cmd(rest...)
    elseif cmd == "shapley" && length(rest) == 4
        shapley_cmd(rest...)
    else
        error("bad arguments: $(join(args, ' '))")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
