# HybridSmoothers.jl

Smoothers and preconditioners for sparse linear systems that run on heterogeneous backends via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Currently implemented

- L1 Gauss–Seidel preconditioner (`L1GSPrecBuilder`)

## L1 Gauss–Seidel preconditioner (`L1GSPrecBuilder`)

Partitioned, parallel-friendly Gauss–Seidel. Runs on CPU and GPU (CUDA, AMDGPU).

### Quick start

```julia
using HybridSmoothers, LinearSolve, SparseArrays, CUDA  # or AMDGPU
import KernelAbstractions as KA

N = 128 * 16
A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
b = ones(N)

builder = L1GSPrecBuilder(CUDABackend(); threads = 256, blocks = 20) # ROCBackend() for AMDGPU
# CPU: builder = L1GSPrecBuilder(KA.CPU(); chunks = 4)

P = builder(A, 16; isSymA = true, sweep = SymmetricSweep())

sol = solve(LinearProblem(A, b), KrylovJL_CG(); Pl = P)
```

### Notes

> [!NOTE]
> **Sweep.** `ForwardSweep` uses the lower triangular part (`M = D + L`),
> `BackwardSweep` uses the upper triangular part (`M = D + Lᵀ`), and
> `SymmetricSweep` combines both (`M = (D + L) D⁻¹ (D + Lᵀ)`). Default is
> `SymmetricSweep`.
>
> **Cache strategy.** `PackedBufferCache` stores the off-diagonal triangular
> entries in a packed vector — efficient when `partsize` is relatively small
> (typical on GPU). `MatrixViewCache` keeps a view of the matrix and reads
> off-diagonal entries directly — preferable for larger `partsize` (typical on
> CPU).
>
> **Symmetry.** Pass `isSymA = true` (or feed a `LinearAlgebra.Symmetric` wrapper)
> to enable a faster CSC code path that exploits *row i = column i*. Otherwise
> the non-symmetric path is used and a `DiagonalIndices` table is built so the
> kernel can locate diagonal entries in CSC storage.

### Reference

Baker, A. H., Falgout, R. D., Kolev, T. V., & Yang, U. M. (2011).
*Multigrid Smoothers for Ultraparallel Computing*, SIAM J. Sci. Comput. 33(5), 2864–2887.
[doi:10.1137/100798806](https://doi.org/10.1137/100798806)
