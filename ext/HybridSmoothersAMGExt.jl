module HybridSmoothersAMGExt

using HybridSmoothers
using LinearAlgebra
import AlgebraicMultigrid as AMG
import KernelAbstractions as KA

struct HybridSmoother{AT, ST, VT}
    A::AT
    s::ST
    residual::VT
    correction::VT
end

function AMG.setup_smoother(config::HybridSmoothers.L1GSPrecBuilder{<: KA.CPU}, A, symmetry)
    n = size(A, 2)
    # Partition size: all partitions are processed simultaneously in L1-GS (block-Jacobi
    # between partitions, GS within). For strongly anisotropic problems where strong
    # coupling crosses partition boundaries, sequential SSOR (AMG.GaussSeidel) is
    # more effective as a multigrid smoother. L1-GS is preferred when parallelism
    # is the primary concern (GPU backends or many-core CPU with moderate anisotropy).
    partsize = max(1, n ÷ Threads.nthreads())
    prec = config(A, partsize; sweep = HybridSmoothers.SymmetricSweep(), isSymA = false)
    return HybridSmoother(A, prec, zeros(eltype(A), n), zeros(eltype(A), n))
end

function AMG.smooth!(x, s::HybridSmoother, b)
    for _ in 1:2
        mul!(s.residual, s.A, x)
        s.residual .-= b                                      # residual = A·x - b
        ldiv!(s.correction, s.s, s.residual)                  # correction = P⁻¹·(A·x - b)
        x .-= s.correction                                    # x += P⁻¹·(b - A·x)
    end
end

function AMG.setup_smoother(config::HybridSmoothers.ChebyshevFourth, A, symmetry)
    return HybridSmoothers._build_smoother(config, A)
end

function AMG.setup_smoother(config::HybridSmoothers.ChebyshevFirst, A, symmetry)
    return HybridSmoothers._build_smoother(config, A)
end

function AMG.smooth!(x, s::HybridSmoothers.ChebyshevFourthSmoother, b)
    HybridSmoothers.smooth!(x, s, b)
end

function AMG.smooth!(x, s::HybridSmoothers.ChebyshevFirstSmoother, b)
    HybridSmoothers.smooth!(x, s, b)
end

end
