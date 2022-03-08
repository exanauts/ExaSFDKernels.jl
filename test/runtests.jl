using Random
using Test

@testset "ExaTronKernels" begin
    @testset "CUDA.jl" begin
        include("CUDA.jl")
    end
    @testset "KA.jl" begin
        include("KA.jl")
    end
end
