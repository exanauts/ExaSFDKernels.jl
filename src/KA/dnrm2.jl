@inline function ExaTronKernels.dnrm2(n::Int,x,incx::Int, I, J)

    @synchronize
    v = 0.0
    for i in 1:n
        @inbounds v += x[i]*x[i]
    end

    @synchronize
    v = sqrt(v)
    @synchronize

    return v
end
