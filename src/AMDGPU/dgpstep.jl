@inline function ExaTronKernels.dgpstep(n::Int,x::ROCDeviceArray{Float64,1},xl::ROCDeviceArray{Float64,1},
                         xu::ROCDeviceArray{Float64,1},alpha,w::ROCDeviceArray{Float64,1},
                         s::ROCDeviceArray{Float64,1})
    tx = workitemIdx().x

    if tx <= n
        @inbounds begin
            # It might be better to process this using just a single thread,
            # rather than diverging between multiple threads.

            if x[tx] + alpha*w[tx] < xl[tx]
                s[tx] = xl[tx] - x[tx]
            elseif x[tx] + alpha*w[tx] > xu[tx]
                s[tx] = xu[tx] - x[tx]
            else
                s[tx] = alpha*w[tx]
            end
        end
    end
    AMDGPU.sync_workgroup()

    return
end
