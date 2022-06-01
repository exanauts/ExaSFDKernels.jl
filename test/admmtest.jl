

using KernelAbstractions
using AMDGPU
using ROCKernels
using CUDA
using CUDAKernels

# One day for Intel GPUs
# using oneAPI
# using OneKernels

using LinearAlgebra
using Printf
using PowerModels
using LazyArtifacts
using ExaTronKernels
using Test
const KA = KernelAbstractions
include("../example/admm/opfdata.jl")
include("../example/admm/environment.jl")
include("../example/admm/generator_kernel.jl")
include("../example/admm/eval_kernel.jl")
include("../example/admm/polar_kernel.jl")
include("../example/admm/bus_kernel.jl")
include("../example/admm/tron_kernel.jl")
include("../example/admm/acopf_admm_gpu.jl")

CASE = joinpath(artifact"ExaData", "ExaData", "matpower", "case9.m")

function one_level_admm(case::String, device::KA.Device)
    # NB: Need to run almost 2,000 iterations to reach convergence with this
    # set of parameters.
    env = admm_rect_gpu(
        case; 
        verbose=2, 
        iterlim=2000, 
        rho_pq=400.0, 
        rho_va=40000.0,
        device=device
    )
    @test isa(env, AdmmEnv)

    model = env.model
    ngen = model.gen_mod.ngen

    par = env.params
    sol = env.solution

    # Check results
    pg = active_power_generation(env)
    qg = reactive_power_generation(env)

    @test sol.status == HAS_CONVERGED
    # Test with solution returned by PowerModels + Ipopt
    @test sol.objval ≈ 5296.6862 rtol=1e-4
    @test pg ≈ [0.897987, 1.34321, 0.941874] rtol=1e-4
    @test qg ≈ [0.1296564, 0.00031842, -0.226342] rtol=1e-2

    # Test restart API
    admm_restart!(env)
end
