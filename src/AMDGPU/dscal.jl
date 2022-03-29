@inline function ExaTronKernels.dscal(n::Int,da::Float64,dx::ROCDeviceArray{Float64,1},incx::Int)
    tx = workitemIdx().x

    # Ignore incx for now.
    if tx <= n
        @inbounds dx[tx] = da*dx[tx]
    end
    AMDGPU.sync_workgroup()

    return
end