module ExaTronKernels
    using LinearAlgebra
    using Requires
    using Printf
    using PowerModels

    export daxpy, dcopy, ddot, dmid, dnrm2, dgpnorm, dscal, dssyax, dnsol, dtsol, dtrqsol, dbreakpt, dgpstep
    export dicf, dicfs, dprsrch, dcauchy, dtrpcg, dspcg, dtron
    
    export nrm2!, reorder!
    export @amdlocalmem

    include("TronMatrix.jl")
    include("ihsort.jl")
    include("insort.jl")
    include("driver.jl")
    include("dsel2.jl")

    # Default CPU kernels
    include("CPU/ExaTronCPUKernels.jl")

    function __init__()
        # @require oneAPI="8f75cd03-7ff8-4ecb-9b8f-daf728133b1b" include("oneAPI/ExaTronOneAPIKernels.jl")
        # @require oneAPI="8f75cd03-7ff8-4ecb-9b8f-daf728133b1b" using ExaTronKernels.ExaTronOneAPIKernels
        # @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("CUDA/ExaTronCUDAKernels.jl")
        # @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" using ExaTronKernels.ExaTronCUDAKernels
        # @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("AMDGPU/ExaTronAMDGPUKernels.jl")
        # @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" using ExaTronKernels.ExaTronAMDGPUKernels
        # @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" include("KA/ExaTronKAKernels.jl")
        # @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" using ExaTronKernels.ExaTronKAKernels
    end
    using KernelAbstractions
    using CUDA
    include("CUDA/ExaTronCUDAKernels.jl")
    using ExaTronKernels.ExaTronCUDAKernels
    using CUDAKernels
    using ROCKernels
    KA = KernelAbstractions
    include("KA/ExaTronKAKernels.jl")
    using ExaTronKernels.ExaTronKAKernels
    include("admm/opfdata.jl")
    include("admm/environment.jl")
    include("admm/generator_kernel.jl")
    include("admm/eval_kernel.jl")
    include("admm/polar_kernel.jl")
    include("admm/bus_kernel.jl")
    include("admm/tron_kernel.jl")
    include("admm/acopf_admm_gpu.jl")
    include("admm/acopf_admm_gpu_two_level.jl")
end
