@inline function dnsol(n::Int, L::oneDeviceArray{Float64,2},
                       r::oneDeviceArray{Float64,1})
    # Solve L*x = r and store the result in r.

    tx = get_local_id()
    ty = get_group_id()

    @inbounds for j=1:n
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        barrier()

        if tx > j && tx <= n
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        barrier()
    end

    return
end
