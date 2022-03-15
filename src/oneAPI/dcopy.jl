@inline function dcopy(n::Int,dx::oneDeviceArray{Float64,1},incx::Int,
                       dy::oneDeviceArray{Float64,1},incy::Int)
    tx = get_local_id()
    ty = get_group_id()

    # Ignore incx and incy for now.
    if tx <= n
        @inbounds dy[tx] = dx[tx]
    end
    barrier()

    return
end
