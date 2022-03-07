# Left-looking Cholesky
@inline function dicf(n::Int,L,I,J)
    tx = J
    ty = I
    @inbounds for j=1:n
        # Apply the pending updates.
        if tx >= j && tx <= n && ty == 1
            for k=1:j-1
                L[tx,j] -= L[tx,k] * L[j,k]
            end
        end
        @synchronize

        if (L[j,j] <= 0)
            @synchronize
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n && ty == 1
            L[tx,j] /= Ljj
        end
        @synchronize
    end

    if tx <= n && ty == 1
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    @synchronize

    return 0
end
