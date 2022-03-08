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
        AMDGPU.sync_workgroup()

        if (L[j,j] <= 0)
            AMDGPU.sync_workgroup()
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n && ty == 1
            L[tx,j] /= Ljj
        end
        AMDGPU.sync_workgroup()
    end

    if tx <= n && ty == 1
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    AMDGPU.sync_workgroup()

    return 0
end
