@inline function ddot(n::Int,dx::oneDeviceArray{Float64,1},incx::Int,
                      dy::oneDeviceArray{Float64,1},incy::Int)
    # Currently, all threads compute the same dot product,
    # hence, no sync_threads() is needed.
    # For very small n, we may want to gauge how much gains
    # we could get by run it in parallel.

    v = 0
    @inbounds for i=1:n
        v += dx[i]*dy[i]
    end
    barrier()
    return v
end
