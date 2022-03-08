@inline function dnsol(n::Int, L::CuDeviceArray{Float64,2},
                       r::CuDeviceArray{Float64,1})
    # Solve L*x = r and store the result in r.

    tx = threadIdx().x
    ty = threadIdx().y

    @inbounds for j=1:n
        if tx == 1 && ty == 1
            r[j] = r[j] / L[j,j]
        end
        AMDGPU.sync_workgroup()

        if tx > j && tx <= n && ty == 1
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        AMDGPU.sync_workgroup()
    end

    return
end
