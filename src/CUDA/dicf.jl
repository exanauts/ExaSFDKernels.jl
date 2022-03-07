# Left-looking Cholesky
@inline function dicf(n::Int,L::CuDeviceArray{Float64,2})
    tx = threadIdx().x
    ty = threadIdx().y

    @inbounds for j=1:n
        # Apply the pending updates.
        if j > 1
            if tx >= j && tx <= n && ty == 1
                for k=1:j-1
                    L[tx,j] -= L[tx,k] * L[j,k]
                end
            end
        end
        CUDA.sync_threads()

        if (L[j,j] <= 0)
            CUDA.sync_threads()
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n && ty == 1
            L[tx,j] /= Ljj
        end
        CUDA.sync_threads()
    end

    if tx <= n && ty == 1
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    CUDA.sync_threads()

    return 0
end
