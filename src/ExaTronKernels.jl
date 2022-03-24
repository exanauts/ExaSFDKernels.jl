module ExaTronKernels
    using LinearAlgebra
    using Requires

    abstract type AbstractKernels end
    struct CUDAKernels   <: AbstractKernels end
    struct AMDKernels    <: AbstractKernels end
    struct OneAPIKernels <: AbstractKernels end
    struct KAKernels     <: AbstractKernels end
    supports(::AbstractKernels) = false
    export supports
    export daxpy, dcopy, ddot, dmid, dnrm2, dgpnorm, dscal, dssyax, dnsol, dtsol, dtrqsol, dbreakpt, dgpstep
    export dicf, dicfs, dprsrch, dcauchy, dtrpcg, dspcg, dtron
    
    export nrm2!, reorder!

    include("TronMatrix.jl")
    include("ihsort.jl")
    include("insort.jl")
    include("driver.jl")
    include("dsel2.jl")

    # Default CPU kernels
    include("CPU/ExaTronCPUKernels.jl")

    function __init__()
        @require oneAPI="8f75cd03-7ff8-4ecb-9b8f-daf728133b1b" include("oneAPI/ExaTronOneAPIKernels.jl")
        @require oneAPI="8f75cd03-7ff8-4ecb-9b8f-daf728133b1b" using ExaTronKernels.ExaTronOneAPIKernels
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("CUDA/ExaTronCUDAKernels.jl")
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" using ExaTronKernels.ExaTronCUDAKernels
        @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" include("KA/ExaTronKAKernels.jl")
        @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" using ExaTronKernels.ExaTronKAKernels
    end
end
