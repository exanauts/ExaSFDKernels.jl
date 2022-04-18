using Random
using Test

@testset "ExaTronKernels" begin
    using CUDA
    using AMDGPU
    using oneAPI
    @testset "CUDA.jl" begin
        if CUDA.has_cuda_gpu()
            include("CUDA.jl")
        end
    end
    @testset "AMDGPU.jl" begin
        if AMDGPU.has_rocm_gpu()
            include("AMDGPU.jl")
        end
    end
    @testset "oneAPI.jl" begin
        if oneAPI.functional()
            include("oneAPI.jl")
        end
    end
    @testset "KA.jl" begin
        if CUDA.has_cuda_gpu() || AMDGPU.has_rocm_gpu()
            include("KA.jl")
        end
    end
end
