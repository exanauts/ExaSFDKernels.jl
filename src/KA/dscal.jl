@inline function dscal(n::Int,da::Float64,dx,incx::Int, I, J)
    tx = J
    ty = 1

    # Ignore incx for now.
    if tx <= n && ty == 1
        @inbounds dx[tx] = da*dx[tx]
    end
    @synchronize

    return
end
