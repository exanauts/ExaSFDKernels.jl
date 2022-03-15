@inline function dnrm2(n::Int,x::oneDeviceArray{Float64,1},incx::Int)
    tx = get_local_id()

    barrier()
    v = 0.0
    for i in 1:n
        @inbounds v += x[i]*x[i]
    end

    barrier()
    v = sqrt(v)

    return v
end
