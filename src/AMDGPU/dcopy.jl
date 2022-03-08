@inline function dcopy(n::Int,dx::CuDeviceArray{Float64,1},incx::Int,
                       dy::CuDeviceArray{Float64,1},incy::Int)
    tx = threadIdx().x
    ty = threadIdx().y

    # Ignore incx and incy for now.
    if tx <= n && ty == 1
        @inbounds dy[tx] = dx[tx]
    end
    AMDGPU.sync_workgroup()

    return
end
