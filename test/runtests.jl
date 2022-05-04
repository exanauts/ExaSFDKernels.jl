using Random
using Test

@testset "ExaTronKernels" begin
    using CUDA
    using AMDGPU
    @testset "CUDA.jl" begin
        if CUDA.has_cuda_gpu()
            include("CUDA.jl")
        end
    end
    @testset "AMDGPU.jl" begin
        if has_rocm_gpu()
            include("AMDGPU.jl")
        end
    end
    # @testset "oneAPI.jl" begin
    #     include("oneAPI.jl")
    # end
    @testset "KA.jl" begin
        if CUDA.has_cuda_gpu() || AMDGPU.has_rocm_gpu()
            include("KA.jl")
        end
    end
end
