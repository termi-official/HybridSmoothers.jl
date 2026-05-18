module HybridSmoothersAMGExt

using HybridSmoothers
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
    prec = config(A, n ÷ Threads.nthreads(); sweep = HybridSmoothers.SymmetricSweep(), isSymA = true)
    return HybridSmoother(A, prec, zeros(eltype(A), n), zeros(eltype(A), n))
end

function AMG.smooth!(x, s::HybridSmoother, b)
    for _ in 1:10
        mul!(s.residual, s.A, x)
        s.residual .-= b                                      # residual = b - A·x
        LinearAlgebra.ldiv!(s.correction, s.s, s.residual)    # correction = P⁻¹·residual
        x .-= s.correction                                    # smoother update: x += correction
    end
end

end
