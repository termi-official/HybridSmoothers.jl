using HybridSmoothers
using Test

@testset "HybridSmoothers" begin
    include("test_preconditioners.jl")
    include("test_smoothers.jl")
end
