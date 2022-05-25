@inline function ExaTronKernels.dnsol(n::Int, L,
                       r,
                       I, J)
    # Solve L*x = r and store the result in r.

    tx = J
    ty = I

    @inbounds for j=1:n
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        @synchronize

        if tx > j && tx <= n
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        @synchronize
    end

    return
end
