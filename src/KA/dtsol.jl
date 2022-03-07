@inline function dtsol(n::Int, L,
                       r,
                       I, J)
    # Solve L'*x = r and store the result in r.

    tx = J
    ty = 1

    @inbounds for j=n:-1:1
        if tx == 1 && ty == 1
            r[j] = r[j] / L[j,j]
        end
        @synchronize

        if tx < j && ty == 1
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        @synchronize
    end

    return
end
