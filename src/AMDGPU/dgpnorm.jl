@inline function ExaTronKernels.dgpnorm(n::Int, x::ROCDeviceArray{Float64,1}, xl::ROCDeviceArray{Float64,1},
                         xu::ROCDeviceArray{Float64,1}, g::ROCDeviceArray{Float64,1})
    tx = workitemIdx().x

    inf_norm = @amdlocalmem(Float64, 1)
    AMDGPU.sync_workgroup()
    res = 0.0

    v = 0.0
    if tx == 1
        inf_norm[1] = 0.0
        for i in 1:n
            @inbounds begin
                if xl[i] != xu[i]
                    if x[i] == xl[i]
                        v = min(g[i], 0.0)
                        v = v*v
                    elseif x[i] == xu[i]
                        v = max(g[i], 0.0)
                        v = v*v
                    else
                        v = g[i]*g[i]
                    end

                    v = sqrt(v)
                    if inf_norm[1] > v
                        inf_norm[1] = inf_norm[1]
                    else
                        inf_norm[1] = v
                    end
                end
            end
        end
    end

    AMDGPU.sync_workgroup()
    res = inf_norm[1]
    AMDGPU.sync_workgroup()
    return res
end
