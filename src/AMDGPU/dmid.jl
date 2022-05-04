@inline function ExaTronKernels.dmid(n::Int, x::ROCDeviceArray{Float64,1},
                      xl::ROCDeviceArray{Float64,1}, xu::ROCDeviceArray{Float64,1})
    tx = workitemIdx().x

    if tx <= n
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    AMDGPU.sync_workgroup()

    return
end
