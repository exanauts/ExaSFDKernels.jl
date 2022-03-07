@inline function dmid(n::Int, x,
                      xl, xu,
                      I, J)
    tx = J
    ty = 1

    if tx <= n && ty == 1
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    @synchronize

    return
end
