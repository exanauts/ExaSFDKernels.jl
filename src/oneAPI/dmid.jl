@inline function ExaTronKernels.dmid(n::Int, x::oneDeviceArray{Float64,1},
                      xl::oneDeviceArray{Float64,1}, xu::oneDeviceArray{Float64,1})
    tx = get_local_id()

    if tx <= n
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    barrier()

    return
end
