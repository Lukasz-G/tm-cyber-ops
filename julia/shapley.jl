# Exact Shapley attribution for a Fuzzy-Pattern Tsetlin machine, in closed form.
#
# Ported from tm-cyber-lab (julia/shapley.jl, commit e6f8fb8), where the derivation is set out and
# the values are checked against brute-force enumeration over all coalitions. tests/shapley_closed_form.jl
# repeats that check here.
#
# Value function: interventional, v(S) = f(z_S), where z_S takes the explained instance's value on
# the features in S and a background instance's value elsewhere. A background set is an outer mean,
# because Shapley values are linear in v. This is the value function interventional TreeSHAP uses
# for tree ensembles, so the two models' attributions answer the same question.
#
# A TM class score is a signed sum of clause votes, and each vote is max(0, ceiling - misses). For one
# clause, sort its literals by their state at x and at the background b:
#
#   A  satisfied under both     never a miss
#   B  unsatisfied under both   always a miss                        (counted into beta)
#   C  satisfied only under x   a miss unless its feature is in S    (size g)
#   D  satisfied only under b   a miss when its feature is in S      (size d)
#
# so misses(S) = beta + (g - |C n S|) + |D n S|. Features outside C u D get zero; all members of C
# share one value and all members of D another; each marginal contribution is 0 or +-1. Over a uniform
# random permutation, the number r of relevant features preceding i is uniform on 0:(g+d-1) and the
# C/D split of those r is hypergeometric, which gives an O((g+d)^2) double sum per clause, independent
# of the feature count.
#
# A feature with both polarities included lands in C and D at once. Exactly one of the pair is
# unsatisfied whatever the feature's value, so its value is zero and it adds one constant miss to beta.

module Shapley

using TMCore: TMClassifier, ClauseBank, CeilingPolicy, LiteralCapped, FlatLF

export clause_literals, shapley, shapley_mean, shapley_mean!

# Binomials through log-factorials in Float64. The summed terms are positive hypergeometric
# probabilities, so there is no cancellation; relative error stays near 1e-14. An exact BigInt version
# allocated per term and was too slow to run threaded at dataset scale.
const _LOGFACT_N = 1 << 17

const _LOGFACT = let v = Vector{Float64}(undef, _LOGFACT_N + 1)
    v[1] = 0.0
    @inbounds for n in 1:_LOGFACT_N
        v[n + 1] = v[n] + log(n)
    end
    v
end

@inline function _logfact(n::Int)
    n <= _LOGFACT_N || error("log-factorial table too small for n = $n; raise _LOGFACT_N")
    return @inbounds _LOGFACT[n + 1]
end

@inline function _logbinom(n::Int, k::Int)
    (n < 0 || k < 0 || k > n) && return -Inf
    return _logfact(n) - _logfact(k) - _logfact(n - k)
end

"Included literals of clause `j` as `(feature, is_negated)` pairs."
function clause_literals(b::ClauseBank, j::Integer)
    out = Tuple{Int,Bool}[]
    for n in 1:b.nchunks, (mask, neg) in ((b.included[n, j], false), (b.included_inv[n, j], true))
        iszero(mask) && continue
        for bit in 0:63
            f = (n - 1) * 64 + bit + 1
            f > b.width && break
            iszero(mask & (UInt64(1) << bit)) || push!(out, (f, neg))
        end
    end
    return out
end

ceiling_of(b::ClauseBank, j, LF, ::LiteralCapped) = (c = Int(b.count[j]); c == 0 ? LF : min(c, LF))
ceiling_of(::ClauseBank, ::Any, LF, ::FlatLF) = LF

"Shared Shapley value of a C member (`from_c`) or a D member, for one clause."
function _unit_value(beta::Int, g::Int, d::Int, ceil::Int, from_c::Bool)
    tot = g + d
    tot == 0 && return 0.0
    from_c ? (g == 0 && return 0.0) : (d == 0 && return 0.0)
    limit = from_c ? ceil : ceil - 1        # misses threshold for a +-1 marginal contribution
    nc = from_c ? g - 1 : g                 # other C members
    nd = from_c ? d : d - 1                 # other D members
    acc = 0.0
    for r in 0:(tot - 1)
        ldenom = _logbinom(tot - 1, r)
        s = 0.0
        for k in max(0, r - nd):min(r, nc)
            beta + g + r - 2k <= limit || continue
            s += exp(_logbinom(nc, k) + _logbinom(nd, r - k) - ldenom)
        end
        acc += s
    end
    return acc / tot
end

"Accumulate one clause's exact Shapley values into `phi`."
function clause_shapley!(phi::AbstractVector{Float64}, bank::ClauseBank, j::Integer,
                         x::AbstractVector{Bool}, b::AbstractVector{Bool},
                         LF::Integer, pol::CeilingPolicy, sign::Int)
    lits = clause_literals(bank, j)
    isempty(lits) && return nothing         # an empty clause votes its ceiling whatever the input
    sat(v, neg) = neg ? !v : v
    beta = 0
    cset = Set{Int}(); dset = Set{Int}()
    for (f, neg) in lits
        sx, sb = sat(x[f], neg), sat(b[f], neg)
        if sx && sb
            # A
        elseif !sx && !sb
            beta += 1
        elseif sx && !sb
            push!(cset, f)
        else
            push!(dset, f)
        end
    end
    both = intersect(cset, dset)
    beta += length(both)                    # one of the pair always misses; see the header
    C = setdiff(cset, both); D = setdiff(dset, both)
    g, d = length(C), length(D)
    g + d == 0 && return nothing            # this clause cannot react to this (x, b) pair

    ceil = ceiling_of(bank, j, LF, pol)
    if g > 0
        v = _unit_value(beta, g, d, ceil, true)
        for f in C
            phi[f] += sign * v
        end
    end
    if d > 0
        v = _unit_value(beta, g, d, ceil, false)
        for f in D
            phi[f] -= sign * v
        end
    end
    return nothing
end

function shapley!(phi::AbstractVector{Float64}, m::TMClassifier, ci::Integer,
                  x::AbstractVector{Bool}, b::AbstractVector{Bool}, sign::Int=1)
    LF, pol = m.params.LF, m.ceiling
    for (bank, s) in ((m.positive[ci], sign), (m.negative[ci], -sign))
        for j in 1:bank.nclauses
            clause_shapley!(phi, bank, j, x, b, LF, pol, s)
        end
    end
    return phi
end

"""
    shapley(model, ci, x, b) -> Vector{Float64}

Exact Shapley values of every feature for class index `ci`, explaining `x` against one background
instance `b`. Sums to `score(m, ci, x) - score(m, ci, b)`.
"""
function shapley(m::TMClassifier, ci::Integer, x::AbstractVector{Bool}, b::AbstractVector{Bool})
    phi = zeros(Float64, length(x))
    shapley!(phi, m, ci, x, b)
    return phi
end

"""
    shapley_mean!(phi, model, x, backgrounds; plus, minus = nothing)

Exact Shapley values against a background set, written into `phi`. With `minus` set, the explained
quantity is the margin `score(plus) - score(minus)`; Shapley values are linear, so this is the
difference of the two classes' values.
"""
function shapley_mean!(phi::AbstractVector{Float64}, m::TMClassifier, x::AbstractVector{Bool},
                       backgrounds::AbstractVector{<:AbstractVector{Bool}};
                       plus::Integer, minus::Union{Nothing,Integer}=nothing)
    fill!(phi, 0.0)
    for b in backgrounds
        shapley!(phi, m, plus, x, b, 1)
        minus === nothing || shapley!(phi, m, minus, x, b, -1)
    end
    phi ./= length(backgrounds)
    return phi
end

shapley_mean(m::TMClassifier, ci::Integer, x::AbstractVector{Bool},
             backgrounds::AbstractVector{<:AbstractVector{Bool}}) =
    shapley_mean!(zeros(Float64, length(x)), m, x, backgrounds; plus=ci)

end # module Shapley
