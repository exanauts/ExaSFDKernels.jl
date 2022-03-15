@inline function dtsol(n::Int, L::oneDeviceArray{Float64,2},
                       r::oneDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.

    tx = get_local_id()
    ty = get_group_id()

    @inbounds for j=n:-1:1
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        barrier()

        if tx < j
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        barrier()
    end

    return
end
