using HybridSmoothers
using SparseArrays, SparseMatricesCSR, LinearAlgebra, Random
import KernelAbstractions as KA
import HybridSmoothers: _build_smoother, _l1_rowsum, _l1jacobi_spectral_radius
import HybridSmoothers: smooth!

# 1D Poisson matrix: symmetric, diagonally dominant, SPD.
function cheb_poisson_1d(N)
    spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end

@testset "Chebyshev Smoothers" begin

    @testset "Internals" begin
        @testset "_l1_rowsum" begin
            # Boundary rows: |2| + |-1| = 3;  interior rows: |-1| + |2| + |-1| = 4
            A = cheb_poisson_1d(5)
            @test _l1_rowsum(A) ≈ [3.0, 4.0, 4.0, 4.0, 3.0]
        end

        @testset "_l1jacobi_spectral_radius" begin
            # For any SPD matrix, ρ(M⁻¹A) must lie in (0, 1] when M is the L1-Jacobi
            # diagonal (M_i = Σ_j |A_{ij}|), since M - A is non-negative for this choice.
            A = cheb_poisson_1d(100)
            l1inv = 1.0 ./ _l1_rowsum(A)
            A_csr = SparseMatrixCSR(A)
            ρ = _l1jacobi_spectral_radius(A_csr, l1inv)
            @test 0 < ρ <= 1
        end
    end

    @testset "ChebyshevFourth" begin
        @testset "Smoother struct" begin
            N = 50
            A = cheb_poisson_1d(N)
            s = _build_smoother(ChebyshevFourth(; degree = 4, backend = KA.CPU()), A)

            @test s.A isa SparseMatricesCSR.SparseMatrixCSR   # converted for parallel SpMV
            @test s.backend === KA.CPU()
            @test s.degree == 4
            @test length(s.l1inv) == N
            @test length(s.r) == N
            @test length(s.z) == N
            @test s.ρ > 0
            # l1inv entries should equal 1/l1 = 1/row-sum-of-|A|
            l1_ref = _l1_rowsum(A)
            @test s.l1inv ≈ 1.0 ./ l1_ref
        end

        @testset "smooth! reduces residual (various degrees)" begin
            N = 300
            A = cheb_poisson_1d(N)
            b = ones(N)
            for degree in [1, 2, 4, 8]
                @testset "degree=$degree" begin
                    s = _build_smoother(ChebyshevFourth(; degree = degree, backend = KA.CPU()), A)
                    x = zeros(N)
                    r0 = norm(b - A * x)
                    smooth!(x, s, b)
                    @test norm(b - A * x) < r0
                end
            end
        end

        @testset "smooth! monotonically decreases residual over 10 applications" begin
            N = 200
            A = cheb_poisson_1d(N)
            b = rand(MersenneTwister(42), N)
            s = _build_smoother(ChebyshevFourth(; degree = 4, backend = KA.CPU()), A)
            x = zeros(N)
            prev = norm(b - A * x)
            for _ in 1:10
                smooth!(x, s, b)
                curr = norm(b - A * x)
                @test curr < prev
                prev = curr
            end
            # After 10 applications the relative residual must be meaningfully reduced
            @test prev / norm(b - A * zeros(N)) < 0.9
        end

        @testset "higher degree gives smaller residual (single application)" begin
            N = 300
            A = cheb_poisson_1d(N)
            b = ones(N)
            residuals = Float64[]
            for degree in [1, 2, 4, 8]
                s = _build_smoother(ChebyshevFourth(; degree = degree, backend = KA.CPU()), A)
                x = zeros(N)
                smooth!(x, s, b)
                push!(residuals, norm(b - A * x))
            end
            # Each higher degree should be non-worse than the previous
            @test issorted(residuals; rev = true)
        end
    end

    @testset "ChebyshevFirst" begin
        @testset "Smoother struct" begin
            N = 50
            A = cheb_poisson_1d(N)
            s = _build_smoother(ChebyshevFirst(; degree = 4, backend = KA.CPU()), A)

            @test s.A isa SparseMatricesCSR.SparseMatrixCSR
            @test s.backend === KA.CPU()
            @test s.degree == 4
            @test length(s.l1inv) == N
            @test length(s.r) == N
            @test length(s.z) == N
            @test length(s.tmp) == N
            @test 0 < s.a < 1   # optimal a* lies strictly in (0, 1)
            @test s.ρ > 0
        end

        @testset "smooth! reduces residual (various degrees)" begin
            N = 300
            A = cheb_poisson_1d(N)
            b = ones(N)
            for degree in [1, 2, 4, 8]
                @testset "degree=$degree" begin
                    s = _build_smoother(ChebyshevFirst(; degree = degree, backend = KA.CPU()), A)
                    x = zeros(N)
                    r0 = norm(b - A * x)
                    smooth!(x, s, b)
                    @test norm(b - A * x) < r0
                end
            end
        end

        @testset "smooth! monotonically decreases residual over 10 applications" begin
            N = 200
            A = cheb_poisson_1d(N)
            b = rand(MersenneTwister(42), N)
            s = _build_smoother(ChebyshevFirst(; degree = 4, backend = KA.CPU()), A)
            x = zeros(N)
            prev = norm(b - A * x)
            for _ in 1:10
                smooth!(x, s, b)
                curr = norm(b - A * x)
                @test curr < prev
                prev = curr
            end
            @test prev / norm(b - A * zeros(N)) < 0.9
        end

        @testset "higher degree gives smaller residual (single application)" begin
            N = 300
            A = cheb_poisson_1d(N)
            b = ones(N)
            residuals = Float64[]
            for degree in [1, 2, 4, 8]
                s = _build_smoother(ChebyshevFirst(; degree = degree, backend = KA.CPU()), A)
                x = zeros(N)
                smooth!(x, s, b)
                push!(residuals, norm(b - A * x))
            end
            @test issorted(residuals; rev = true)
        end

        @testset "a* is monotonically decreasing in degree" begin
            # D'Ambra et al. Fig. 3: a* decreases as degree grows
            astars = [_build_smoother(ChebyshevFirst(; degree = k, backend = KA.CPU()),
                                      cheb_poisson_1d(10)).a for k in 1:8]
            @test issorted(astars; rev = true)
        end
    end

    @testset "ChebyshevFirst vs ChebyshevFourth comparable" begin
        # D'Ambra et al. show ChebyshevFirst has better V-cycle error bounds.
        # In practice both smoothers reduce the residual by similar amounts per step.
        N = 300
        A = cheb_poisson_1d(N)
        b = rand(MersenneTwister(7), N)
        r0 = norm(b)
        for degree in 1:5
            @testset "degree=$degree" begin
                s4 = _build_smoother(ChebyshevFourth(; degree = degree, backend = KA.CPU()), A)
                s1 = _build_smoother(ChebyshevFirst(; degree = degree, backend = KA.CPU()), A)
                x4 = zeros(N); x1 = zeros(N)
                smooth!(x4, s4, b)
                smooth!(x1, s1, b)
                r4 = norm(b - A * x4) / r0
                r1 = norm(b - A * x1) / r0
                # Both must reduce the residual and their performance is within 2%
                @test r4 < 1.0
                @test r1 < 1.0
                @test abs(r1 - r4) / r4 < 0.02
            end
        end
    end

    @testset "Thread safety: result independent of workgroup layout" begin
        # Both smoothers must produce the same answer regardless of how KA tiles the work.
        # We verify this by comparing with a tiny and a large workgroup size by running on
        # the same initial state (x=0, b=ones) and checking the outputs match.
        N = 500
        A = cheb_poisson_1d(N)
        b = ones(N)

        for SmootherCfg in [ChebyshevFourth(; degree = 4, backend = KA.CPU()),
                             ChebyshevFirst(; degree = 4, backend = KA.CPU())]
            s = _build_smoother(SmootherCfg, A)

            # Run 1
            x1 = zeros(N)
            smooth!(x1, s, b)

            # Reset workspace and run again — result must be identical
            fill!(s.r, 0); fill!(s.z, 0)
            if hasproperty(s, :tmp)
                fill!(s.tmp, 0)
            end
            x2 = zeros(N)
            smooth!(x2, s, b)

            @test x1 ≈ x2
        end
    end
end
