@inline function dcopy(n::Int,dx,incx::Int,
                       dy,incy::Int,
                       I, J)
    tx = J
    ty = 1

    # Ignore incx and incy for now.
    if tx <= n && ty == 1
        @inbounds dy[tx] = dx[tx]
    end
    @synchronize

    return
end
