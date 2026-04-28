module HybridSmoothers

using SparseArrays, SparseMatricesCSR
using LinearSolve
using Adapt
using LinearAlgebra
using TimerOutputs: @timeit_debug
import Base: \
import KernelAbstractions: @kernel, @index, functional, CPU, synchronize
import SparseArrays: getcolptr, getnzval

include("devices.jl")
include("utils.jl")

# ----------------------------------------------------------------------------
# Sparse matrix traits shared by smoothers and preconditioners.
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
sparsemat_format_type(::Union{SparseMatrixCSR, ThreadedSparseMatrixCSR}) = CSRFormat()

# Wrapped to avoid type piracy when extending for device-side sparse types.
colvals(A::Union{SparseMatrixCSR, ThreadedSparseMatrixCSR}) = SparseMatricesCSR.getcolval(A)
getrowptr(A::Union{SparseMatrixCSR, ThreadedSparseMatrixCSR}) = SparseMatricesCSR.getrowptr(A)

# ----------------------------------------------------------------------------
# Smoother abstraction (matches the AlgebraicMultigrid.jl convention:
# a `Smoother` is callable as `(s)(A, x, b)` and updates `x` in-place toward
# the solution of `Ax = b`).
# ----------------------------------------------------------------------------
abstract type Smoother end

include("l1_gauss_seidel.jl")
include("smoother.jl")

export L1GSPrecBuilder, L1GaussSeidel
export ForwardSweep, BackwardSweep, SymmetricSweep
export PackedBufferCache, MatrixViewCache
export Smoother, setup_smoother

export AbstractDevice, AbstractCPUDevice, AbstractGPUDevice,
    SequentialCPUDevice, PolyesterDevice, CudaDevice,
    default_backend, value_type, index_type

export ThreadedSparseMatrixCSR

end
