using oneAPI

include("gpu.jl")

@test oneAPI.functional()
oneAPI.allowscalar(false)

run_l1gs_gpu_tests(oneAPIBackend(); testset_name = "L1GS Preconditioner - Intel oneAPI")
