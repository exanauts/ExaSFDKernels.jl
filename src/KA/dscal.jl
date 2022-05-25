@inline function ExaTronKernels.dscal(n::Int,da::Float64,dx,incx::Int, I, J)
    tx = J
    ty = 1

    # Ignore incx for now.
    if tx <= n
        @inbounds dx[tx] = da*dx[tx]
    end
    @synchronize

    return
end
