using HybridSmoothers
using SparseArrays
using LinearAlgebra: norm, ldiv!
using Test

##########################################
## L1 Gauss Seidel Smoother - CPU       ##
##########################################

function poisson_test_matrix(N)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end

@testset "L1GaussSeidel Smoother" begin
    @testset "Standalone iterative solve" begin
        # With partsize=N (single partition) and η large enough that the L1
        # correction vanishes, the L1GS smoother reduces to standard Gauss–Seidel.
        # Verify the same convergence relations as the dense GS sanity check:
        #   - Forward and Backward sweeps take the same number of iterations
        #   - Symmetric sweep takes about half as many
        N = 50
        A = poisson_test_matrix(N)
        b = ones(N)
        tol = 1e-10
        maxiter = 10_000

        function gs_solve(smoother, A, b)
            x = zeros(length(b))
            for iter = 1:maxiter
                smoother(A, x, b)
                norm(A * x - b) < tol && return x, iter
            end
            return x, maxiter
        end

        s_fwd = L1GaussSeidel(sweep = ForwardSweep(),  partsize = N, η = 1.0)
        s_bwd = L1GaussSeidel(sweep = BackwardSweep(), partsize = N, η = 1.0)
        s_sym = L1GaussSeidel(sweep = SymmetricSweep(), partsize = N, η = 1.0)

        _, iters_fwd = gs_solve(s_fwd, A, b)
        _, iters_bwd = gs_solve(s_bwd, A, b)
        _, iters_sym = gs_solve(s_sym, A, b)

        println(
            "L1GaussSeidel smoother iterations: " *
            "Forward=$iters_fwd, Backward=$iters_bwd, Symmetric=$iters_sym",
        )

        @test iters_fwd == iters_bwd
        @test abs(iters_sym - iters_fwd ÷ 2) <= iters_fwd ÷ 100
    end

    @testset "Multiple inner iters per call" begin
        # iter=k applied once should match iter=1 applied k times.
        N = 50
        A = poisson_test_matrix(N)
        b = ones(N)
        x_a = zeros(N)
        x_b = zeros(N)

        s_one  = L1GaussSeidel(sweep = ForwardSweep(), partsize = N, iter = 1, η = 1.0)
        s_many = L1GaussSeidel(sweep = ForwardSweep(), partsize = N, iter = 5, η = 1.0)

        for _ = 1:5
            s_one(A, x_a, b)
        end
        s_many(A, x_b, b)

        @test x_a ≈ x_b
    end

    @testset "setup_smoother + ldiv! API" begin
        # Driving the cache directly (the path AMG.jl uses inside V-cycles).
        N = 50
        A = poisson_test_matrix(N)
        b = ones(N)

        config = L1GaussSeidel(sweep = SymmetricSweep(), partsize = N, η = 1.0)
        cache = setup_smoother(config, A)

        x = zeros(N)
        for _ = 1:5000
            ldiv!(x, cache, b)
            norm(A * x - b) < 1e-10 && break
        end
        @test norm(A * x - b) < 1e-10
    end
end
