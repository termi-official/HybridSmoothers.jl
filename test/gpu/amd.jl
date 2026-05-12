using AMDGPU

include("gpu.jl")

@test AMDGPU.functional()
AMDGPU.allowscalar(false)

run_l1gs_gpu_tests(ROCBackend(); testset_name = "L1GS Preconditioner - AMD ROCm")
