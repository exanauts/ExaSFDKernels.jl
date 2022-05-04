@inline function ExaTronKernels.dtsol(n::Int, L::ROCDeviceArray{Float64,2},
                       r::ROCDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.

    tx = workitemIdx().x

    @inbounds for j=n:-1:1
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        AMDGPU.sync_workgroup()

        if tx < j
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        AMDGPU.sync_workgroup()
    end

    return
end
