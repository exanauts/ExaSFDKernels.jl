module ExaTronAMDGPUKernels
    macro amdlocalmem(T, n)
        id = gensym("static_shmem")
        ex = quote
            ROCDeviceArray(($n,),alloc_local($(QuoteNode(id)), $T, $n))
        end
        esc(ex)
    end

    macro amdlocalmem(T, n1, n2)
        id = gensym("static_shmem")
        ex = quote
            ROCDeviceArray(($n1,$n2),alloc_local($(QuoteNode(id)), $T, $n1 * $n2))
        end
        esc(ex)
    end
    export @amdlocalmem
    using ..ExaTronKernels
    using ..AMDGPU
    include("daxpy.jl")
    include("dcopy.jl")
    include("ddot.jl")
    include("dmid.jl")
    include("dnrm2.jl")
    include("dgpnorm.jl")
    include("dscal.jl")
    include("dssyax.jl")
    include("dnsol.jl")
    include("dtsol.jl")
    include("dtrqsol.jl")
    include("dbreakpt.jl")
    include("dgpstep.jl")
    include("dicf.jl")
    include("dicfs.jl")
    include("dprsrch.jl")
    include("dcauchy.jl")
    include("dtrpcg.jl")
    include("dspcg.jl")
    include("dtron.jl")
    include("TronMatrix.jl")
end