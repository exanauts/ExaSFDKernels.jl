@inline function ExaTronKernels.dnrm2(n::Int,x::ROCDeviceArray{Float64,1},incx::Int)
    tx = workitemIdx().x

    AMDGPU.sync_workgroup()
    v = 0.0
    for i in 1:n
        @inbounds v += x[i]*x[i]
    end

    AMDGPU.sync_workgroup()
    v = sqrt(v)

    return v
end
