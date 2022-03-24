@inline function ExaTronKernels.dgpnorm(n::Int, x::oneDeviceArray{Float64,1}, xl::oneDeviceArray{Float64,1},
                         xu::oneDeviceArray{Float64,1}, g::oneDeviceArray{Float64,1})
    tx = get_local_id()
    res = 0.0
    inf_norm = oneLocalArray(Float64, 1)

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

    barrier()
    res = inf_norm[1]
    barrier()
    return res
end
