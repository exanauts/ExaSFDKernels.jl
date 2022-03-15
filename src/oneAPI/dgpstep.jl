@inline function dgpstep(n::Int,x::oneDeviceArray{Float64,1},xl::oneDeviceArray{Float64,1},
                         xu::oneDeviceArray{Float64,1},alpha,w::oneDeviceArray{Float64,1},
                         s::oneDeviceArray{Float64,1})
    tx = get_local_id()
    ty = get_group_id()

    if tx <= n && ty == 1
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
    barrier()

    return
end
