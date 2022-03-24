using Random
using Test

@testset "ExaTronKernels" begin
    using CUDA
    @testset "CUDA.jl" begin
        if CUDA.has_cuda_gpu()
            include("CUDA.jl")
        end
    end
    @testset "AMDGPU.jl" begin
        # include("AMDGPU.jl")
    end
    @testset "oneAPI.jl" begin
        # include("oneAPI.jl")
    end
    @testset "KA.jl" begin
        if CUDA.has_cuda_gpu()
            include("KA.jl")
        end
    end
end
