module HybridSmoothersAMDGPUExt

# HOTFIX. AMDGPU.jl is missing two pieces that CUDA.jl already has:

using AMDGPU
using Adapt
using SparseMatricesCSR: SparseMatrixCSR

# (1) TODO: remove when https://github.com/JuliaGPU/AMDGPU.jl/pull/905 is merged and released.
Adapt.adapt_storage(::AMDGPU.ROCBackend, a::AbstractArray) =
    Adapt.adapt(AMDGPU.ROCArray, a)

# (2) TODO: remove when https://github.com/JuliaGPU/AMDGPU.jl/pull/906 is merged and released.
AMDGPU.rocSPARSE.ROCSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
    AMDGPU.rocSPARSE.ROCSparseMatrixCSR{T}(
        AMDGPU.ROCVector{Cint}(Mat.rowptr),
        AMDGPU.ROCVector{Cint}(Mat.colval),
        AMDGPU.ROCVector{T}(Mat.nzval),
        size(Mat),
    )
AMDGPU.rocSPARSE.ROCSparseMatrixCSR(Mat::SparseMatrixCSR{<:Any, T}) where {T} =
    AMDGPU.rocSPARSE.ROCSparseMatrixCSR{T}(Mat)

Adapt.adapt_storage(::Type{AMDGPU.ROCArray}, xs::SparseMatrixCSR) =
    AMDGPU.rocSPARSE.ROCSparseMatrixCSR(xs)
Adapt.adapt_storage(::Type{AMDGPU.ROCArray{T}}, xs::SparseMatrixCSR) where {T} =
    AMDGPU.rocSPARSE.ROCSparseMatrixCSR{T}(xs)

end
