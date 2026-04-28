using SparseArrays, SparseMatricesCSR
using LinearAlgebra: Transpose
using Polyester: @batch
import LinearAlgebra
import SparseArrays: AbstractSparseMatrix

"""
    ThreadedSparseMatrixCSR
Threaded version of SparseMatrixCSR.

Based on https://github.com/BacAmorim/ThreadedSparseCSR.jl .
"""
struct ThreadedSparseMatrixCSR{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
    A::SparseMatrixCSR{1, Tv, Ti}
end

function ThreadedSparseMatrixCSR(
    m::Integer,
    n::Integer,
    rowptr::Vector{Ti},
    colval::Vector{Ti},
    nzval::Vector{Tv},
) where {Tv, Ti <: Integer}
    ThreadedSparseMatrixCSR(SparseMatrixCSR{1}(m, n, rowptr, colval, nzval))
end

function ThreadedSparseMatrixCSR(a::Transpose{Tv, <:SparseMatrixCSC} where {Tv})
    ThreadedSparseMatrixCSR(SparseMatrixCSR(a))
end

function LinearAlgebra.mul!(
    y::AbstractVector{<:Number},
    A_::ThreadedSparseMatrixCSR,
    x::AbstractVector{<:Number},
    alpha::Number,
    beta::Number,
)
    A = A_.A
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    @batch minbatch = size(y, 1) ÷ Threads.nthreads() for row = 1:size(y, 1)
        @inbounds begin
            v = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz]
                v += A.nzval[nz] * x[col]
            end
            y[row] = alpha * v + beta * y[row]
        end
    end

    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector{<:Number},
    A_::ThreadedSparseMatrixCSR,
    x::AbstractVector{<:Number},
)
    A = A_.A
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    @batch minbatch = max(1, size(y, 1) ÷ Threads.nthreads()) for row = 1:size(y, 1)
        @inbounds begin
            v = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz]
                v += A.nzval[nz] * x[col]
            end
            y[row] = v
        end
    end

    return y
end

function _tspmcsr_mul(A::ThreadedSparseMatrixCSR, x::AbstractVector)
    y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
    return LinearAlgebra.mul!(y, A, x)
end
Base.:*(A::ThreadedSparseMatrixCSR, v::AbstractVector) = _tspmcsr_mul(A, v)

Base.eltype(A::ThreadedSparseMatrixCSR)            = Base.eltype(A.A)
Base.size(A::ThreadedSparseMatrixCSR)              = Base.size(A.A)
Base.size(A::ThreadedSparseMatrixCSR, i)           = Base.size(A.A, i)
Base.IndexStyle(::Type{<:ThreadedSparseMatrixCSR}) = IndexCartesian()

SparseMatricesCSR.getrowptr(A::ThreadedSparseMatrixCSR) = SparseMatricesCSR.getrowptr(A.A)
SparseMatricesCSR.getnzval(A::ThreadedSparseMatrixCSR)  = SparseMatricesCSR.getnzval(A.A)
SparseMatricesCSR.getcolval(A::ThreadedSparseMatrixCSR) = SparseMatricesCSR.getcolval(A.A)

SparseArrays.issparse(A::ThreadedSparseMatrixCSR) = issparse(A.A)
SparseArrays.nnz(A::ThreadedSparseMatrixCSR)      = nnz(A.A)
SparseArrays.nonzeros(A::ThreadedSparseMatrixCSR) = nonzeros(A.A)

Base.@propagate_inbounds function SparseArrays.getindex(
    A::ThreadedSparseMatrixCSR{T},
    i0::Integer,
    i1::Integer,
) where {T}
    getindex(A.A, i0, i1)
end
SparseArrays.getindex(A::ThreadedSparseMatrixCSR, ::Colon, ::Colon) = copy(A)
SparseArrays.getindex(A::ThreadedSparseMatrixCSR, i::Int, ::Colon) = getindex(A.A, i, 1:size(A, 2))
SparseArrays.getindex(A::ThreadedSparseMatrixCSR, ::Colon, i::Int) = getindex(A.A, 1:size(A, 1), i)
