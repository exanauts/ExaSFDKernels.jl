
@inline function nrm2!(wa, A::CuDeviceArray{Float64,2}, n::Int)
    tx = threadIdx().x
    ty = threadIdx().y

    v = 0.0
    if tx <= n && ty == 1
        @inbounds for j=1:n
            v += A[j,tx]^2
        end
        @inbounds wa[tx] = sqrt(v)
    end
    #=
    v = A[tx,ty]^2

    if tx > n || ty > n
        v = 0.0
    end

    # Sum over the x-dimension.
    offset = div(blockDim().x, 2)
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset = div(offset, 2)
    end

    if tx == 1
        wa[ty] = sqrt(v)
    end
    =#
    AMDGPU.sync_workgroup()

    return
end

@inline function dssyax(n::Int, A::CuDeviceArray{Float64,2},
                        z::CuDeviceArray{Float64,1},
                        q::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    ty = threadIdx().y

    v = 0.0
    if tx <= n && ty == 1
        @inbounds for j=1:n
            v += A[tx,j]*z[j]
        end
        @inbounds q[tx] = v
    end
    #=
    v = 0.0
    if tx <= n && ty <= n
        v = A[ty,tx]*z[tx]
    end

    # Sum over the x-dimension: v = sum_tx A[ty,tx]*z[tx].
    # The thread with tx=1 will have the sum in v.

    offset = div(blockDim().x, 2)
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset = div(offset, 2)
    end

    if tx == 1
        q[ty] = v
    end
    =#
    AMDGPU.sync_workgroup()

    return
end

@inline function reorder!(n::Int, nfree::Int, B::CuDeviceArray{Float64,2},
                          A::CuDeviceArray{Float64,2}, indfree::CuDeviceArray{Int,1},
                          iwa::CuDeviceArray{Int,1})
    tx = threadIdx().x
    ty = threadIdx().y

    #=
    if tx == 1 && ty == 1
        @inbounds for j=1:nfree
            jfree = indfree[j]
            B[j,j] = A[jfree,jfree]
            for i=jfree+1:n
                if iwa[i] > 0
                    B[iwa[i],j] = A[i,jfree]
                    B[j,iwa[i]] = B[iwa[i],j]
                end
            end
        end
    end
    =#
    if tx <= nfree && ty == 1
        @inbounds begin
            jfree = indfree[tx]
            B[tx,tx] = A[jfree,jfree]
            for i=jfree+1:n
                if iwa[i] > 0
                    B[iwa[i],tx] = A[i,jfree]
                    B[tx,iwa[i]] = B[iwa[i],tx]
                end
            end
        end
    end

    AMDGPU.sync_workgroup()

    return
end
