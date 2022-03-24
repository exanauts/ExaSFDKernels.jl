@inline function ExaTronKernels.dcopy(n::Int,dx::oneDeviceArray{Float64,1},incx::Int,
                       dy::oneDeviceArray{Float64,1},incy::Int)
    tx = get_local_id()

    # Ignore incx and incy for now.
    @inbounds dy[tx] = dx[tx]
    barrier()

    return
end
