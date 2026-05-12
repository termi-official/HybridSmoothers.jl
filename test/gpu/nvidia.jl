using CUDA

include("gpu.jl")

@test CUDA.functional()
CUDA.allowscalar(false)

run_l1gs_gpu_tests(CUDABackend(); testset_name = "L1GS Preconditioner - Nvidia CUDA")
