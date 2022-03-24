@inline function ExaTronKernels.dscal(n::Int,da::Float64,dx::oneDeviceArray{Float64,1},incx::Int)
    tx = get_local_id()

    # Ignore incx for now.
    @inbounds dx[tx] = da*dx[tx]
    barrier()

    return
end