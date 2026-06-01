using Test
using SparseArrays
using LinearSolve
using SparseMatricesCSR
using KernelAbstractions
using MatrixDepot
using HybridSmoothers
using LinearAlgebra: Symmetric

############################################
## L1 Gauss–Seidel Preconditioner — GPU   ##
##                                        ##
## Backend-agnostic test functions. Each  ##
## takes the KA backend as the first arg. ##
## Drive the suite via `nvidia.jl` /      ##
## `amd.jl`, which call                   ##
## `run_l1gs_gpu_tests(backend; …)`.      ##
############################################

function poisson_test_matrix(N)
    # 1D Poisson with Dirichlet BCs (symmetric tridiagonal CSC).
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end

function test_sym_result(
    backend,
    testname,
    A,
    x,
    y_exp,
    D_DL1_exp,
    partsize,
    sweep = ForwardSweep(),
    cache_strategy = PackedBufferCache(),
    test_buffer_fn = (P) -> nothing,
)
    @testset "$testname Symmetric" begin
        total_nblocks = 10
        total_nthreads = 10
        for nblocks = 1:total_nblocks # answer is invariant to nblocks/nthreads
            for nthreads = 1:total_nthreads
                builder = L1GSPrecBuilder(backend; threads = nthreads, blocks = nblocks)
                P =
                    A isa Symmetric ?
                    builder(A, partsize; sweep = sweep, cache_strategy = cache_strategy) :
                    builder(
                        A,
                        partsize;
                        isSymA = true,
                        sweep = sweep,
                        cache_strategy = cache_strategy,
                    )
                if sweep isa SymmetricSweep
                    @test Vector(P.sweep.lop.D_DL1) ≈ D_DL1_exp
                    @test Vector(P.sweep.uop.D_DL1) ≈ D_DL1_exp
                else
                    @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
                end
                test_buffer_fn(P)
                y = P \ x
                @test y ≈ y_exp
            end
        end
    end
end

function test_l1gs_prec(
    backend,
    testname,
    A,
    b,
    sweep = ForwardSweep(),
    cache_strategy = PackedBufferCache();
    nblocks = 20,
    nthreads = 256,
    partsize = nothing,
    isSymA = false,
    solver = KrylovJL_GMRES(),
)
    @testset "$testname" begin
        partsize = partsize === nothing ? ceil(Int, size(A, 1) / (nblocks * nthreads)) : partsize

        prob = LinearProblem(A, b)

        sol_unprec = solve(prob, solver)
        @test isapprox(A * sol_unprec.u, b, rtol = 1e-1, atol = 1e-1)

        P = if A isa Symmetric
            L1GSPrecBuilder(backend; threads = nthreads, blocks = nblocks)(
                A,
                partsize;
                sweep = sweep,
                cache_strategy = cache_strategy,
            )
        else
            L1GSPrecBuilder(backend; threads = nthreads, blocks = nblocks)(
                A,
                partsize;
                sweep = sweep,
                cache_strategy = cache_strategy,
                isSymA = isSymA,
            )
        end

        sol_prec = solve(prob, solver; Pl = P)

        println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
        println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
        @test isapprox(A * sol_prec.u, b, rtol = 1e-1, atol = 1e-1)
        @test sol_prec.iters <= sol_unprec.iters
    end
end


function run_l1gs_gpu_tests(backend; testset_name = "L1GS Preconditioner - GPU")
    @testset "$testset_name" begin
        @testset "Algorithm" begin
            N = 9
            A = poisson_test_matrix(N)
            x = 0:(N-1) |> collect .|> Float64
            y_exp_fwd = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
            y_exp_bwd = [0.25, 0.5, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0]
            y_exp_sym = [0.25, 0.5, 2.0, 2.0, 3.75, 3.5, 5.5, 5.0, 4.0]
            D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii (2 >= 1.5*1)
            SLbuffer_exp = Float64.([-1, -1, -1, -1])
            SUbuffer_exp = Float64.([-1, -1, -1, -1])

            test_fwd_buffer_fn = (P) -> @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp
            test_bwd_buffer_fn = (P) -> @test Vector(P.sweep.op.U.SUbuffer) ≈ SUbuffer_exp
            test_sym_buffer_fn = (P) -> begin
                @test Vector(P.sweep.lop.L.SLbuffer) ≈ SLbuffer_exp
                @test Vector(P.sweep.uop.U.SUbuffer) ≈ SUbuffer_exp
            end

            # Forward sweep packed buffer tests
            test_sym_result(
                backend,
                "Packed, Forward, GPU CSC",
                A,
                x,
                y_exp_fwd,
                D_DL1_exp,
                2,
                ForwardSweep(),
                PackedBufferCache(),
                test_fwd_buffer_fn,
            )
            B = SparseMatrixCSR(A)
            test_sym_result(
                backend,
                "Packed, Forward, GPU CSR",
                B,
                x,
                y_exp_fwd,
                D_DL1_exp,
                2,
                ForwardSweep(),
                PackedBufferCache(),
                test_fwd_buffer_fn,
            )

            # Backward sweep packed buffer tests
            test_sym_result(
                backend,
                "Packed, Backward, GPU CSC",
                A,
                x,
                y_exp_bwd,
                D_DL1_exp,
                2,
                BackwardSweep(),
                PackedBufferCache(),
                test_bwd_buffer_fn,
            )
            test_sym_result(
                backend,
                "Packed, Backward, GPU CSR",
                B,
                x,
                y_exp_bwd,
                D_DL1_exp,
                2,
                BackwardSweep(),
                PackedBufferCache(),
                test_bwd_buffer_fn,
            )

            # Symmetric sweep packed buffer tests
            test_sym_result(
                backend,
                "Packed, Symmetric, GPU CSC",
                A,
                x,
                y_exp_sym,
                D_DL1_exp,
                2,
                SymmetricSweep(),
                PackedBufferCache(),
                test_sym_buffer_fn,
            )
            test_sym_result(
                backend,
                "Packed, Symmetric, GPU CSR",
                B,
                x,
                y_exp_sym,
                D_DL1_exp,
                2,
                SymmetricSweep(),
                PackedBufferCache(),
                test_sym_buffer_fn,
            )

            # MatrixViewCache tests
            test_sym_result(
                backend,
                "MatrixView, Forward, GPU CSC",
                A,
                x,
                y_exp_fwd,
                D_DL1_exp,
                2,
                ForwardSweep(),
                MatrixViewCache(),
            )
            test_sym_result(
                backend,
                "MatrixView, Forward, GPU CSR",
                B,
                x,
                y_exp_fwd,
                D_DL1_exp,
                2,
                ForwardSweep(),
                MatrixViewCache(),
            )

            test_sym_result(
                backend,
                "MatrixView, Backward, GPU CSC",
                A,
                x,
                y_exp_bwd,
                D_DL1_exp,
                2,
                BackwardSweep(),
                MatrixViewCache(),
            )
            test_sym_result(
                backend,
                "MatrixView, Backward, GPU CSR",
                B,
                x,
                y_exp_bwd,
                D_DL1_exp,
                2,
                BackwardSweep(),
                MatrixViewCache(),
            )

            test_sym_result(
                backend,
                "MatrixView, Symmetric, GPU CSC",
                A,
                x,
                y_exp_sym,
                D_DL1_exp,
                2,
                SymmetricSweep(),
                MatrixViewCache(),
            )
            test_sym_result(
                backend,
                "MatrixView, Symmetric, GPU CSR",
                B,
                x,
                y_exp_sym,
                D_DL1_exp,
                2,
                SymmetricSweep(),
                MatrixViewCache(),
            )

            @testset "η parameter" begin
                # η = 2.0: a_ii = 2 >= η*dl1_ii = 2 → dl1star = 0 → D_DL1 = a_ii = 2
                η = 2.0
                D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
                SLbuffer_exp = Float64.([-1, -1, -1, -1])
                builder = L1GSPrecBuilder(backend; threads = 2, blocks = 2)
                P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
                @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp

                # η = 3.0: condition NOT satisfied → dl1star = 0.5 → D_DL1 = 2.5
                η = 3.0
                D_DL1_exp = Float64.([2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
                P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
                @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp

                # η = 1.0: condition satisfied → D_DL1 = a_ii = 2
                η = 1.0
                D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
                P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
                @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp

                y_exp = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
                η = 2.0
                P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
                y = P \ x
                @test y ≈ y_exp
            end

            @testset "Non-Symmetric CSC" begin
                A2 = copy(A)
                A2[1, 8] = -1.0  # won't affect the result
                A2[2, 8] = -1.0  # 1/2 → 1/3
                y2_fwd_exp = [0, 1.0 / 3.0, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
                y2_bwd_exp = [1.0 / 6.0, 1.0 / 3.0, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0]
                D_DL1_exp2 = Float64.([2, 3, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: only row 1 has a_ii < η*dl1_ii (2 < 1.5*2)
                SLbuffer_exp2 = Float64.([-1, -1, -1, -1])
                SUbuffer_exp2 = Float64.([-1, -1, -1, -1])

                builder = L1GSPrecBuilder(backend; threads = 2, blocks = 2)

                P = builder(A2, 2; sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
                @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp2
                @test P \ x ≈ y2_fwd_exp

                P = builder(A2, 2; sweep = ForwardSweep(), cache_strategy = MatrixViewCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
                @test P \ x ≈ y2_fwd_exp

                P = builder(A2, 2; sweep = BackwardSweep(), cache_strategy = PackedBufferCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
                @test Vector(P.sweep.op.U.SUbuffer) ≈ SUbuffer_exp2
                @test P \ x ≈ y2_bwd_exp

                P = builder(A2, 2; sweep = BackwardSweep(), cache_strategy = MatrixViewCache())
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
                @test P \ x ≈ y2_bwd_exp
            end

            @testset "Partsize" begin
                partsize = 3
                D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
                SLbuffer_exp = Float64.([-1, 0, -1, -1, 0, -1, -1, 0, -1])
                builder = L1GSPrecBuilder(backend; threads = 2, blocks = 2)
                P = builder(
                    A,
                    partsize;
                    isSymA = true,
                    sweep = ForwardSweep(),
                    cache_strategy = PackedBufferCache(),
                )
                @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
                @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp
            end
        end

        @testset "Solution with LinearSolve" begin
            @testset "Non-Symmetric A (HB/sherman5)" begin
                md = mdopen("HB/sherman5")
                A = md.A
                b = md.b[:, 1]
                test_l1gs_prec(
                    backend,
                    "PackedBuffer, ForwardSweep HB/sherman5",
                    A,
                    b,
                    ForwardSweep(),
                    PackedBufferCache();
                    partsize = 10,
                )
                test_l1gs_prec(
                    backend,
                    "MatrixView, ForwardSweep HB/sherman5",
                    A,
                    b,
                    ForwardSweep(),
                    MatrixViewCache();
                    partsize = 10,
                )
                test_l1gs_prec(
                    backend,
                    "PackedBuffer, BackwardSweep HB/sherman5",
                    A,
                    b,
                    BackwardSweep(),
                    PackedBufferCache();
                    partsize = 10,
                )
                test_l1gs_prec(
                    backend,
                    "MatrixView, BackwardSweep HB/sherman5",
                    A,
                    b,
                    BackwardSweep(),
                    MatrixViewCache();
                    partsize = 10,
                )
            end

            @testset "Symmetric A" begin
                @testset "Wathen Matrix" begin
                    A = matrixdepot("wathen", 120)
                    b = ones(size(A, 1))

                    # Forward/Backward sweeps yield a non-symmetric preconditioner; use GMRES, not CG.
                    test_l1gs_prec(
                        backend,
                        "PackedBuffer, ForwardSweep wathen",
                        A,
                        b,
                        ForwardSweep(),
                        PackedBufferCache();
                        isSymA = true,
                        partsize = 10,
                        solver = KrylovJL_GMRES(),
                    )
                    test_l1gs_prec(
                        backend,
                        "MatrixView, ForwardSweep wathen",
                        A,
                        b,
                        ForwardSweep(),
                        MatrixViewCache();
                        isSymA = true,
                        partsize = 10,
                        solver = KrylovJL_GMRES(),
                    )

                    test_l1gs_prec(
                        backend,
                        "PackedBuffer, BackwardSweep wathen",
                        A,
                        b,
                        BackwardSweep(),
                        PackedBufferCache();
                        isSymA = true,
                        partsize = 10,
                        solver = KrylovJL_GMRES(),
                    )
                    test_l1gs_prec(
                        backend,
                        "MatrixView, BackwardSweep wathen",
                        A,
                        b,
                        BackwardSweep(),
                        MatrixViewCache();
                        isSymA = true,
                        partsize = 10,
                        solver = KrylovJL_GMRES(),
                    )

                    test_l1gs_prec(
                        backend,
                        "PackedBuffer, SymmetricSweep wathen",
                        A,
                        b,
                        SymmetricSweep(),
                        PackedBufferCache();
                        isSymA = true,
                        partsize = 10,
                        solver = KrylovJL_CG(),
                    )
                    test_l1gs_prec(
                        backend,
                        "MatrixView, SymmetricSweep wathen",
                        A,
                        b,
                        SymmetricSweep(),
                        MatrixViewCache();
                        isSymA = true,
                        partsize = 10,
                        solver = KrylovJL_CG(),
                    )
                end

                # Pothen/bodyy6: 19366×19366 Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
                @testset "Pothen/bodyy6 (Symmetric type)" begin
                    md = mdopen("Pothen/bodyy6")
                    A = md.A
                    b = ones(size(A, 1))

                    test_l1gs_prec(
                        backend,
                        "MatrixViewCache, ForwardSweep bodyy6",
                        A,
                        b,
                        ForwardSweep(),
                        MatrixViewCache();
                        partsize = 10,
                    )
                    test_l1gs_prec(
                        backend,
                        "PackedBufferCache, ForwardSweep bodyy6",
                        A,
                        b,
                        ForwardSweep(),
                        PackedBufferCache();
                        partsize = 10,
                    )

                    test_l1gs_prec(
                        backend,
                        "MatrixViewCache, BackwardSweep bodyy6",
                        A,
                        b,
                        BackwardSweep(),
                        MatrixViewCache();
                        partsize = 10,
                    )
                    test_l1gs_prec(
                        backend,
                        "PackedBufferCache, BackwardSweep bodyy6",
                        A,
                        b,
                        BackwardSweep(),
                        PackedBufferCache();
                        partsize = 10,
                    )

                    test_l1gs_prec(
                        backend,
                        "MatrixViewCache, SymmetricSweep bodyy6",
                        A,
                        b,
                        SymmetricSweep(),
                        MatrixViewCache();
                        partsize = 10,
                    )
                    test_l1gs_prec(
                        backend,
                        "PackedBufferCache, SymmetricSweep bodyy6",
                        A,
                        b,
                        SymmetricSweep(),
                        PackedBufferCache();
                        partsize = 10,
                    )
                end
            end
        end
    end
end
