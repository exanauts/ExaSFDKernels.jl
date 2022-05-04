@inline function ExaTronKernels.dcopy(n::Int,dx::ROCDeviceArray{Float64,1},incx::Int,
                       dy::ROCDeviceArray{Float64,1},incy::Int)
    tx = workitemIdx().x

    # Ignore incx and incy for now.
    if tx <= n
        @inbounds dy[tx] = dx[tx]
    end
    AMDGPU.sync_workgroup()

    return
end
