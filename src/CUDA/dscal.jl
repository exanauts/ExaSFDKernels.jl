@inline function dscal(n::Int,da::Float64,dx::CuDeviceArray{Float64,1},incx::Int)
    tx = threadIdx().x
    ty = threadIdx().y

    # Ignore incx for now.
    if tx <= n && ty == 1
        @inbounds dx[tx] = da*dx[tx]
    end
    CUDA.sync_threads()

    return
end