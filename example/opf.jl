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
const KA = KernelAbstractions
include("admm/opfdata.jl")
include("admm/environment.jl")
include("admm/generator_kernel.jl")
include("admm/eval_kernel.jl")
include("admm/polar_kernel.jl")
include("admm/bus_kernel.jl")
include("admm/tron_kernel.jl")
include("admm/acopf_admm_gpu.jl")

# `datafile`: the name of the test file of type `String`
# here: MATPOWER case2868rte.m in ExaData project Artifact
# datafile = joinpath(artifact"ExaData", "ExaData", "matpower", "case2868rte.m")
datafile = joinpath(artifact"ExaData", "ExaData", "matpower", "case9.m")
# `rho_pq`: ADMM parameter for power flow of type `Float64`
rho_pq = 10.0
# `rho_va`: ADMM parameter for voltage and angle of type `Float64`
rho_va = 1000.0
# `max_iter`: maximum number of iterations of type `Int`
max_iter = 1000
# Indicate which GPU device to use
device = CPU()
# device = CUDADevice()
# device = ROCDevice()
# device = OneDevice()
# Use polar formulation for branch problems
use_polar = true
# verbose = 0: No output
# verbose = 1: Final result metrics
# verbose = 2: Iteration output
verbose = 1

env = admm_rect_gpu(
    datafile;
    iterlim=max_iter,
    rho_pq=rho_pq,
    rho_va=rho_va,
    scale=1e-4,
    use_polar=use_polar,
    device=device,
    verbose=verbose
)
