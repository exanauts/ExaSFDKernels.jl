@inline function ExaTronKernels.daxpy(n::Int,da::Float64,
                       dx::oneDeviceArray{Float64,1},incx::Int,
                       dy::oneDeviceArray{Float64,1},incy::Int)
    tx = get_local_id()

    if tx <= n
        @inbounds dy[tx] = dy[tx] + da*dx[tx]
    end
    barrier()

    return
end
