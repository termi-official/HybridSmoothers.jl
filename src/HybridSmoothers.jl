module HybridSmoothers

using SparseArrays:
    SparseArrays, SparseMatrixCSC, AbstractSparseMatrix,
    spdiagm, rowvals, getcolptr, getnzval
using SparseMatricesCSR: SparseMatricesCSR, SparseMatrixCSR
using LinearSolve: LinearSolve
using Adapt: Adapt, adapt
using LinearAlgebra: LinearAlgebra, Symmetric
using TimerOutputs: @timeit_debug
import Base: \
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index, functional, CPU, synchronize

# ----------------------------------------------------------------------------
# Sparse matrix traits used by the preconditioners.
# CSR/CSC are identical for symmetric matrices, so we carry a separate symmetry
# trait that lets each backend pick its preferred access pattern.
# ----------------------------------------------------------------------------
abstract type AbstractMatrixSymmetry end
struct SymmetricMatrix <: AbstractMatrixSymmetry end
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end

abstract type AbstractMatrixFormat end
struct CSRFormat <: AbstractMatrixFormat end
struct CSCFormat <: AbstractMatrixFormat end

# These traits exist because device sparse matrix types (e.g. CuSparseDeviceMatrixCSR)
# do not share a supertype with their host counterparts, so dispatch on
# AbstractSparseMatrixCSC/CSR is not portable across backends.
sparsemat_format_type(::SparseMatrixCSC) = CSCFormat()
sparsemat_format_type(::SparseMatrixCSR) = CSRFormat()

# Wrapped to avoid type piracy when extending for device-side sparse types.
colvals(A::SparseMatrixCSR) = SparseMatricesCSR.getcolval(A)
getrowptr(A::SparseMatrixCSR) = SparseMatricesCSR.getrowptr(A)

include("l1_gauss_seidel.jl")

export L1GSPrecBuilder
export ForwardSweep, BackwardSweep, SymmetricSweep
export PackedBufferCache, MatrixViewCache

end
