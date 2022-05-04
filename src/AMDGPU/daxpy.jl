@inline function ExaTronKernels.daxpy(n::Int,da::Float64,
                       dx::ROCDeviceArray{Float64,1},incx::Int,
                       dy::ROCDeviceArray{Float64,1},incy::Int)
    tx = workitemIdx().x

    if tx <= n
        @inbounds dy[tx] = dy[tx] + da*dx[tx]
    end
    AMDGPU.sync_workgroup()

    return
end
