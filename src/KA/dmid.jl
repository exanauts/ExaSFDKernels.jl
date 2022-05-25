@inline function ExaTronKernels.dmid(n::Int, x,
                      xl, xu,
                      I, J)
    tx = J

    if tx <= n
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    @synchronize

    return
end
