# Left-looking Cholesky
@inline function dicf(n::Int,L::oneDeviceArray{Float64,2})
    tx = get_local_id()
    ty = get_group_id()

    @inbounds for j=1:n
        # Apply the pending updates.
        if j > 1
            if tx >= j && tx <= n
                for k=1:j-1
                    L[tx,j] -= L[tx,k] * L[j,k]
                end
            end
        end
        barrier()

        if (L[j,j] <= 0)
            barrier()
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n
            L[tx,j] /= Ljj
        end
        barrier()
    end

    if tx <= n
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    barrier()

    return 0
end
