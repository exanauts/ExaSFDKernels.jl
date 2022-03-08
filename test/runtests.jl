using Random
using Test

@testset "ExaSGDKernels" begin
    @testset "CUDA.jl" begin
        include("CUDA.jl")
    end
    @testset "KA.jl" begin
        include("KA.jl")
    end
end
