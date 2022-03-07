@inline function dtrqsol(n::Int,x,
                         p,delta::Float64,
                         I, J)
    zero = 0.0
    sigma = zero

    ptx = ddot(n, p, 1, x, 1, I, J)
    ptp = ddot(n, p, 1, p, 1, I, J)
    xtx = ddot(n, x, 1, x, 1, I, J)
    dsq = delta^2

    # Guard against abnormal cases.
    rad = ptx^2 + ptp*(dsq - xtx)
    rad = sqrt(max(rad, zero))

    if ptx > zero
        sigma = (dsq - xtx)/(ptx + rad)
    elseif rad > zero
        sigma = (rad - ptx)/ptp
    else
        sigma = zero
    end
    @synchronize

    return sigma
end
