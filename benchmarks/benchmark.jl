using AlgebraicMultigrid, HybridSmoothers, LinearAlgebra, SparseArrays
using BenchmarkTools
using Test

import KernelAbstractions as KA

# Config
N = 100000
threads = Threads.nthreads()

# Problem
x = rand(N)
b = zeros(N)
A = Tridiagonal(-ones(N-1), 2ones(N), -ones(N-1))
As = sparse(A)

# Warmup and rough test
y1 = zeros(N); ldiv!(y1, L1GSPrecBuilder(KA.CPU(); chunks = 1)(As, N; isSymA = true), b-A*x); x+y1
y2 = copy(x); GaussSeidel(;iter=1)(As, y2, b); y2

@test y2 ≈ x+y1

@info "GaussSeidel"
@btime GaussSeidel(;iter=1)($As, $y2, $b)

@info "L1GS Setup"
@btime L1GSPrecBuilder(KA.CPU(); chunks = threads)($As, N÷threads)
@btime L1GSPrecBuilder(KA.CPU(); chunks = threads)($As, N÷threads; isSymA = true)

@info "L1GS Apply"
Ap = L1GSPrecBuilder(KA.CPU(); chunks = threads)(As, N÷threads)
@btime ldiv!($y1, $Ap, $b)
Ap = L1GSPrecBuilder(KA.CPU(); chunks = threads)(As, N÷threads; isSymA = true)
@btime ldiv!($y1, $Ap, $b)
