################################
### L1 Gauss–Seidel Smoother ###
################################

@doc raw"""
    L1GaussSeidel(; sweep, iter, partsize, η, isSymA, cache_strategy, device)

Configuration for using ℓ₁ Gauss–Seidel as a *smoother* (e.g. inside a multigrid
V-cycle).

Callable as `(s)(A, x, b)` to update `x` in-place by applying `iter` smoothing
sweeps of the iteration

```math
    x^{(k+1)} = (D_{ℓ_1} + L_{\text{part}})^{-1} (b - L^T x^{(k)})
```

(forward sweep; mirror form for backward; composition for symmetric).

# Smoother vs Preconditioner

This is the smoother form of L1GS. The *preconditioner* form
([`L1GSPrecBuilder`](@ref)) provides ``M^{-1}`` such that ``G = I - M^{-1} A``,
and is applied to a residual via `LinearSolve.ldiv!(z, P, r)`. Both share the
same partitioned kernels for the triangular solves, but the *outer operator*
differs — see the docstring of [`SymmetricSweep`](@ref) for the derivation.

# Fields
- `sweep`: `ForwardSweep`, `BackwardSweep`, or `SymmetricSweep`.
- `iter`: number of smoothing sweeps per call (default `1`).
- `partsize`: size of each diagonal block (default `32`).
- `η`: L1 threshold parameter (default `1.5`).
- `isSymA`: whether `A` is symmetric (enables faster CSC access path).
- `cache_strategy`: `MatrixViewCache()` (CPU) or `PackedBufferCache()` (GPU).
- `device`: backend device (default `SequentialCPUDevice()`).
"""
struct L1GaussSeidel{S <: AbstractSweep, C <: AbstractCacheStrategy, D <: AbstractDevice} <: Smoother
    sweep::S
    iter::Int
    partsize::Int
    η::Float64
    isSymA::Bool
    cache_strategy::C
    device::D
end

function L1GaussSeidel(;
    sweep::AbstractSweep = SymmetricSweep(),
    iter::Integer = 1,
    partsize::Integer = 32,
    η = 1.5,
    isSymA::Bool = false,
    cache_strategy::AbstractCacheStrategy = MatrixViewCache(),
    device::AbstractDevice = PolyesterDevice(),
)
    return L1GaussSeidel(
        sweep, Int(iter), Int(partsize), Float64(η), isSymA, cache_strategy, device,
    )
end

# AMG.jl-style invocation: build per-A cache, then apply.
function (config::L1GaussSeidel)(A, x, b)
    cache = setup_smoother(config, A)
    LinearAlgebra.ldiv!(x, cache, b)
    return x
end

## Per-matrix caches ##

struct L1GSForwardSmootherCache{PT, MT}
    P::PT       # forward L1GS preconditioner — applies (D_l1 + L_part)⁻¹
    Ustrict::MT # strict upper of A — for the residual b - U·x
    iter::Int
end

struct L1GSBackwardSmootherCache{PT, MT}
    P::PT       # backward L1GS preconditioner
    Lstrict::MT # strict lower of A
    iter::Int
end

struct L1GSSymmetricSmootherCache{PFT, PBT, MLT, MUT}
    P_fwd::PFT
    P_bwd::PBT
    Lstrict::MLT
    Ustrict::MUT
    iter::Int
end

# Strict triangular extraction. `Symmetric` storage keeps only one triangle, so
# we materialize the full sparse matrix first (mirrors `get_data` in the
# preconditioner path).
_strict_lower(A::AbstractSparseMatrix) = LinearAlgebra.tril(A, -1)
_strict_upper(A::AbstractSparseMatrix) = LinearAlgebra.triu(A, 1)
_strict_lower(A::Symmetric) = LinearAlgebra.tril(get_data(A), -1)
_strict_upper(A::Symmetric) = LinearAlgebra.triu(get_data(A), 1)

function setup_smoother(config::L1GaussSeidel{<:ForwardSweep}, A)
    builder = L1GSPrecBuilder(config.device)
    P = builder(
        A, config.partsize;
        sweep = ForwardSweep(),
        cache_strategy = config.cache_strategy,
        isSymA = config.isSymA,
        η = config.η,
    )
    return L1GSForwardSmootherCache(P, _strict_upper(A), config.iter)
end

function setup_smoother(config::L1GaussSeidel{<:BackwardSweep}, A)
    builder = L1GSPrecBuilder(config.device)
    P = builder(
        A, config.partsize;
        sweep = BackwardSweep(),
        cache_strategy = config.cache_strategy,
        isSymA = config.isSymA,
        η = config.η,
    )
    return L1GSBackwardSmootherCache(P, _strict_lower(A), config.iter)
end

function setup_smoother(config::L1GaussSeidel{<:SymmetricSweep}, A)
    builder = L1GSPrecBuilder(config.device)
    P_fwd = builder(
        A, config.partsize;
        sweep = ForwardSweep(),
        cache_strategy = config.cache_strategy,
        isSymA = config.isSymA,
        η = config.η,
    )
    P_bwd = builder(
        A, config.partsize;
        sweep = BackwardSweep(),
        cache_strategy = config.cache_strategy,
        isSymA = config.isSymA,
        η = config.η,
    )
    return L1GSSymmetricSmootherCache(
        P_fwd, P_bwd, _strict_lower(A), _strict_upper(A), config.iter,
    )
end

## Smoother application ##
# `ldiv!(x, cache, b)` updates `x` in place using `b` as the right-hand side
# and the current `x` as the initial guess (one application = `cache.iter`
# sweeps).

function LinearAlgebra.ldiv!(
    x::AbstractVector, cache::L1GSForwardSmootherCache, b::AbstractVector,
)
    T = eltype(b)
    rhs = similar(b)
    for _ = 1:cache.iter
        # rhs = b - U·x
        copyto!(rhs, b)
        LinearAlgebra.mul!(rhs, cache.Ustrict, x, -one(T), one(T))
        # x ← (D_l1 + L_part)⁻¹ · rhs
        copyto!(x, rhs)
        _apply_sweep!(x, cache.P)
    end
    return x
end

function LinearAlgebra.ldiv!(
    x::AbstractVector, cache::L1GSBackwardSmootherCache, b::AbstractVector,
)
    T = eltype(b)
    rhs = similar(b)
    for _ = 1:cache.iter
        copyto!(rhs, b)
        LinearAlgebra.mul!(rhs, cache.Lstrict, x, -one(T), one(T))
        copyto!(x, rhs)
        _apply_sweep!(x, cache.P)
    end
    return x
end

function LinearAlgebra.ldiv!(
    x::AbstractVector, cache::L1GSSymmetricSmootherCache, b::AbstractVector,
)
    T = eltype(b)
    rhs = similar(b)
    for _ = 1:cache.iter
        # Forward half-step: x ← (D + L_part)⁻¹ (b - U·x)
        copyto!(rhs, b)
        LinearAlgebra.mul!(rhs, cache.Ustrict, x, -one(T), one(T))
        copyto!(x, rhs)
        _apply_sweep!(x, cache.P_fwd)
        # Backward half-step: x ← (D + U_part)⁻¹ (b - L·x)
        copyto!(rhs, b)
        LinearAlgebra.mul!(rhs, cache.Lstrict, x, -one(T), one(T))
        copyto!(x, rhs)
        _apply_sweep!(x, cache.P_bwd)
    end
    return x
end
