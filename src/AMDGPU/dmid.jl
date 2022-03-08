@inline function dmid(n::Int, x::CuDeviceArray{Float64,1},
                      xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    ty = threadIdx().y

    if tx <= n && ty == 1
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    AMDGPU.sync_workgroup()

    return
end
