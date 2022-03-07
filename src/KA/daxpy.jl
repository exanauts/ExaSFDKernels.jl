@inline function daxpy(n::Int,da::Float64,
                       dx,incx::Int,
                       dy,incy::Int,
                       I, J)
    tx = J
    ty = 1

    if tx <= n && ty == 1
        @inbounds dy[tx] = dy[tx] + da*dx[tx]
    end
    @synchronize

    return
end
