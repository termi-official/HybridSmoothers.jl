########################################################
########################################################
### Chebyshev Polynomial Smoothers                  ###
### D'Ambra, Durastante, Filippone, Massei, Thomas  ###
### (2024) arXiv:2407.09848                         ###
########################################################
########################################################

# Both variants use the ℓ1-Jacobi preconditioner as the basic smoother:
#   M_i = Σ_j |A_{ij}|   (L1 row norm of A)
#
# The polynomial acceleration relies only on SpMV products and diagonal
# scalings, making it fully parallel (no sequential sweep ordering).
# For strongly anisotropic problems GS/SOR-based smoothers may
# outperform these, since the cross-fiber high-frequency modes of M⁻¹A
# cluster near eigenvalue ≈ 0 and are invisible to polynomial smoothers.

# -----------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------

# Default KA workgroup sizes.
_workgroup_size(::KA.CPU) = 64
_workgroup_size(::KA.GPU) = 256

# Row-wise L1 norm computed from the CSC column structure (setup only, runs on CPU).
function _l1_rowsum(A::SparseMatrixCSC{Tv}) where Tv
    n = size(A, 1)
    l1 = zeros(Tv, n)
    @inbounds for col in 1:size(A, 2)
        for idx in A.colptr[col]:A.colptr[col+1]-1
            l1[A.rowval[idx]] += abs(A.nzval[idx])
        end
    end
    return l1
end

# Power iteration for ρ(M⁻¹A). A generic so it accepts SparseMatrixCSR (whose
# mul! is threaded) during setup.
function _l1jacobi_spectral_radius(A, l1inv::AbstractVector{Tv};
                                   power_iters::Int = 50) where Tv
    n = size(A, 1)
    v = fill(one(Tv) / sqrt(Tv(n)), n)
    w = similar(v)
    ρ = one(Tv)
    for _ in 1:power_iters
        mul!(w, A, v)
        w .*= l1inv
        ρ = norm(w)
        iszero(ρ) && break
        v .= w ./ ρ
    end
    return ρ
end

# -----------------------------------------------------------------------
# KA kernels shared by both smoothers
# -----------------------------------------------------------------------

# r[i] = b[i] - r[i]
@kernel function _cheb_b_minus_r_kernel!(r, @Const(b))
    i = @index(Global)
    @inbounds r[i] = b[i] - r[i]
end

# x[i] += z[i]
@kernel function _cheb_axpy_kernel!(x, @Const(z))
    i = @index(Global)
    @inbounds x[i] += z[i]
end

"""
    _cheb1_astar(k) → a*

Compute the optimal left-endpoint for the degree-`k` Chebyshev 1st-kind
polynomial smoother (D'Ambra et al. 2024, eq. 20) via bisection on
`x = √a*` in (0, 1):

    8k(1 - x²)^{2k} + x((1-x)^{4k} - (1+x)^{4k}) = 0
"""
function _cheb1_astar(k::Int)
    φ(x) = 8k * (1 - x^2)^(2k) + x * ((1 - x)^(4k) - (1 + x)^(4k))
    lo, hi = 1e-10, 1.0 - 1e-10
    for _ in 1:60
        mid = (lo + hi) / 2
        φ(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return ((lo + hi) / 2)^2
end

# -----------------------------------------------------------------------
# Chebyshev 4th-kind smoother (Lottes 2023, Section 2.1 of D'Ambra 2024)
# -----------------------------------------------------------------------

"""
    ChebyshevFourth(; degree = 4, backend = KA.CPU())

Chebyshev polynomial smoother of the 4th kind, as described by Lottes (2023)
and in D'Ambra et al. (2024, arXiv:2407.09848, Section 2.1).

Uses ℓ1-Jacobi (M_i = Σ_j |A_{ij}|) as the basic smoother with the
three-term recurrence (eq. 10):

    z⁽⁰⁾ = 0,   r⁽⁰⁾ = b - Ax
    z⁽ᵏ⁾ = ((2k-3)/(2k+1)) z⁽ᵏ⁻¹⁾ + (8k-4)/((2k+1)ρ) M⁻¹r⁽ᵏ⁻¹⁾
    x⁽ᵏ⁾ = x⁽ᵏ⁻¹⁾ + z⁽ᵏ⁾
    r⁽ᵏ⁾ = r⁽ᵏ⁻¹⁾ - A z⁽ᵏ⁾

V-cycle error bound (eq. 11): ‖E‖²_A ≤ C / (C + 4k(k+1)/3).
Costs `degree` SpMV products per `smooth!` call.

The SpMV uses `SparseMatrixCSR` (parallel `mul!` on CPU via SparseMatricesCSR);
element-wise updates use KA kernels dispatched on `backend`.
"""
struct ChebyshevFourth{BackendType <: KA.Backend}
    backend::BackendType
    degree::Int
end
ChebyshevFourth(; degree = 4, backend = KA.CPU()) = ChebyshevFourth(backend, degree)

struct ChebyshevFourthSmoother{BackendType <: KA.Backend, Tv, matT, vecT}
    A::matT        # SparseMatrixCSR (or device equivalent)
    l1inv::vecT
    ρ::Tv
    r::vecT
    z::vecT
    degree::Int
    backend::BackendType
end

# z[i] = αk·z[i] + βk·l1inv[i]·r[i];  x[i] += z[i]
@kernel function _cheb4_step_kernel!(z, x, @Const(r), @Const(l1inv), αk, βk)
    i = @index(Global)
    @inbounds begin
        zi = αk * z[i] + βk * l1inv[i] * r[i]
        z[i] = zi
        x[i] += zi
    end
end

function _build_smoother(config::ChebyshevFourth, A_csc::SparseMatrixCSC{Tv}) where Tv
    backend = config.backend
    l1inv_h = one(Tv) ./ _l1_rowsum(A_csc)
    A_csr   = SparseMatricesCSR.SparseMatrixCSR(A_csc)
    ρ       = _l1jacobi_spectral_radius(A_csr, l1inv_h)
    n       = size(A_csc, 1)
    return ChebyshevFourthSmoother(
        adapt(backend, A_csr),
        adapt(backend, l1inv_h),
        ρ,
        adapt(backend, zeros(Tv, n)),
        adapt(backend, zeros(Tv, n)),
        config.degree,
        backend,
    )
end

function smooth!(x, s::ChebyshevFourthSmoother, b)
    T       = eltype(x)
    n       = length(x)
    backend = s.backend
    wg      = _workgroup_size(backend)

    # r = b - A·x
    mul!(s.r, s.A, x)
    synchronize(backend)
    _cheb_b_minus_r_kernel!(backend, wg)(s.r, b; ndrange = n)
    synchronize(backend)

    fill!(s.z, zero(T))

    for k in 1:s.degree
        αk = T((2k - 3) / (2k + 1))
        βk = T((8k - 4) / ((2k + 1) * s.ρ))
        # z = αk·z + βk·M⁻¹·r;  x += z
        _cheb4_step_kernel!(backend, wg)(s.z, x, s.r, s.l1inv, αk, βk; ndrange = n)
        synchronize(backend)
        # r -= A·z   (r⁽ᵏ⁾ = r⁽ᵏ⁻¹⁾ - A z⁽ᵏ⁾)
        mul!(s.r, s.A, s.z, -one(T), one(T))
        synchronize(backend)
    end
    return nothing
end

# -----------------------------------------------------------------------
# Chebyshev 1st-kind smoother (D'Ambra et al. 2024, Section 2.2)
# -----------------------------------------------------------------------

"""
    ChebyshevFirst(; degree = 4, backend = KA.CPU())

Chebyshev polynomial smoother of the 1st kind with optimal left endpoint `a*`,
as proposed in D'Ambra et al. (2024, arXiv:2407.09848, Section 2.2).

Uses ℓ1-Jacobi (M_i = Σ_j |A_{ij}|) as the basic smoother with the
recurrence (eq. 16):

    r⁽⁰⁾ = (1/ρ) M⁻¹(b - Ax),   z⁽⁰⁾ = 2/(1+a*) · r⁽⁰⁾,   ρ₀ = (1-a*)/(1+a*)
    ρₖ = (2(1+a*)/(1-a*) - ρₖ₋₁)⁻¹
    x⁽ᵏ⁾ = x⁽ᵏ⁻¹⁾ + z⁽ᵏ⁻¹⁾
    r⁽ᵏ⁾ = r⁽ᵏ⁻¹⁾ - (1/ρ) M⁻¹A z⁽ᵏ⁻¹⁾
    z⁽ᵏ⁾ = ρₖ ρₖ₋₁ z⁽ᵏ⁻¹⁾ + 4ρₖ/(1-a*) · r⁽ᵏ⁾

`a*` is the unique root in (0, 1) of (eq. 20):

    8k(1 - x²)^{2k} + x((1-x)^{4k} - (1+x)^{4k}) = 0,   x = √a*

Precomputed tabulated values (Fig. 3):
    k=1: a*≈0.333  k=2: 0.181  k=3: 0.116  k=4: 0.082  k=5: 0.062
    k=6: 0.049     k=7: 0.040  k=8: 0.033

Provides lower V-cycle error bounds than `ChebyshevFourth` for k ≤ 5.
Same cost: `degree` SpMV products per `smooth!` call.

The SpMV uses `SparseMatrixCSR` (parallel `mul!` on CPU via SparseMatricesCSR);
element-wise updates use KA kernels dispatched on `backend`.
"""
struct ChebyshevFirst{BackendType <: KA.Backend}
    backend::BackendType
    degree::Int
end
ChebyshevFirst(; degree = 4, backend = KA.CPU()) = ChebyshevFirst(backend, degree)

struct ChebyshevFirstSmoother{BackendType <: KA.Backend, Tv, matT, vecT}
    A::matT        # SparseMatrixCSR (or device equivalent)
    l1inv::vecT
    ρ::Tv
    a::Tv
    r::vecT
    z::vecT
    tmp::vecT
    degree::Int
    backend::BackendType
end

# r[i] = l1inv[i]·(b[i] - tmp[i])·inv_ρ;  z[i] = c·r[i]
@kernel function _cheb1_init_kernel!(r, z, @Const(b), @Const(tmp), @Const(l1inv), inv_ρ, c)
    i = @index(Global)
    @inbounds begin
        ri   = l1inv[i] * (b[i] - tmp[i]) * inv_ρ
        r[i] = ri
        z[i] = c * ri
    end
end

# r[i] -= l1inv[i]·tmp[i]·inv_ρ
@kernel function _cheb1_update_r_kernel!(r, @Const(tmp), @Const(l1inv), inv_ρ)
    i = @index(Global)
    @inbounds r[i] -= l1inv[i] * tmp[i] * inv_ρ
end

# z[i] = c1·z[i] + c2·r[i]
@kernel function _cheb1_update_z_kernel!(z, @Const(r), c1, c2)
    i = @index(Global)
    @inbounds z[i] = c1 * z[i] + c2 * r[i]
end

function _build_smoother(config::ChebyshevFirst, A_csc::SparseMatrixCSC{Tv}) where Tv
    backend = config.backend
    l1inv_h = one(Tv) ./ _l1_rowsum(A_csc)
    A_csr   = SparseMatricesCSR.SparseMatrixCSR(A_csc)
    ρ       = _l1jacobi_spectral_radius(A_csr, l1inv_h)
    astar   = Tv(_cheb1_astar(config.degree))
    n       = size(A_csc, 1)
    return ChebyshevFirstSmoother(
        adapt(backend, A_csr),
        adapt(backend, l1inv_h),
        ρ,
        astar,
        adapt(backend, zeros(Tv, n)),
        adapt(backend, zeros(Tv, n)),
        adapt(backend, zeros(Tv, n)),
        config.degree,
        backend,
    )
end

function smooth!(x, s::ChebyshevFirstSmoother, b)
    T       = eltype(x)
    n       = length(x)
    backend = s.backend
    wg      = _workgroup_size(backend)
    a       = s.a
    inv_ρ   = one(T) / s.ρ

    # r⁽⁰⁾ = (1/ρ) M⁻¹(b - Ax),  z⁽⁰⁾ = 2/(1+a*) · r⁽⁰⁾
    mul!(s.tmp, s.A, x)
    synchronize(backend)
    _cheb1_init_kernel!(backend, wg)(
        s.r, s.z, b, s.tmp, s.l1inv,
        inv_ρ, T(2) / (one(T) + a);
        ndrange = n,
    )
    synchronize(backend)

    ρ_s = (one(T) - a) / (one(T) + a)

    for k in 1:s.degree
        # x += z
        _cheb_axpy_kernel!(backend, wg)(x, s.z; ndrange = n)
        synchronize(backend)

        if k < s.degree
            # r -= (1/ρ) M⁻¹ A z
            mul!(s.tmp, s.A, s.z)
            synchronize(backend)
            _cheb1_update_r_kernel!(backend, wg)(s.r, s.tmp, s.l1inv, inv_ρ; ndrange = n)
            synchronize(backend)

            # z = ρₖ ρₖ₋₁ z + 4ρₖ/(1-a*) r
            ρ_s_new = one(T) / (T(2) * (one(T) + a) / (one(T) - a) - ρ_s)
            _cheb1_update_z_kernel!(backend, wg)(
                s.z, s.r,
                ρ_s_new * ρ_s, T(4) * ρ_s_new / (one(T) - a);
                ndrange = n,
            )
            synchronize(backend)
            ρ_s = ρ_s_new
        end
    end
    return nothing
end
