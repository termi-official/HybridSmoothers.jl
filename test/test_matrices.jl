using SparseArrays
using LinearAlgebra: I

# 1D Poisson with Dirichlet BCs: symmetric tridiagonal CSC.
poisson_test_matrix(N) = spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))

# 2D Poisson five-point stencil on an n×n grid with the second direction scaled by ε:
# SPD, n² unknowns, strongly coupled along the first (row-contiguous) direction for ε < 1.
function poisson2d_matrix(n; ε = 1.0)
    T = poisson_test_matrix(n)
    return kron(sparse(I, n, n), T) + ε * kron(T, sparse(I, n, n))
end

# poisson2d_matrix plus upwind convection of strength c along the first direction:
# non-symmetric, weakly diagonally dominant.
function convection_diffusion_matrix(n; ε = 1.0, c = 1.0)
    upwind = spdiagm(0 => ones(n), -1 => -ones(n - 1))
    return poisson2d_matrix(n; ε) + c * kron(sparse(I, n, n), upwind)
end
