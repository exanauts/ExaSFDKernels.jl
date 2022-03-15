@inline function dmid(n::Int, x::oneDeviceArray{Float64,1},
                      xl::oneDeviceArray{Float64,1}, xu::oneDeviceArray{Float64,1})
    tx = get_local_id()
    ty = get_group_id()

    if tx <= n && ty == 1
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    barrier()

    return
end
