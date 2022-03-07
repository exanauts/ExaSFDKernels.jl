
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
    CUDA.sync_threads()

    return
end

@inline getdiagvalue(A::TronSparseMatrixCSC, i) = A.diag_vals[i]
@inline getdiagvalue(A::TronDenseMatrix, i) = A.vals[i,i]

function dssyax(n, A::TronDenseMatrix, x, y)
    zero = 0.0

    @inbounds for i=1:n
        y[i] = A.vals[i,i]*x[i]
    end

    @inbounds for j=1:n
        rowsum = zero
        for i=j+1:n
            rowsum += A.vals[i,j]*x[i]
            y[i] += A.vals[i,j]*x[j]
        end
        y[j] += rowsum
    end

    return
end

"""
Subroutine dssyax

This subroutine computes the matrix-vector product y = A*x,
where A is a symmetric matrix with the strict lower triangular
part in compressed column storage.
"""
dssyax(A::TronSparseMatrixCSC, x, y) = dssyax(A.n, A, x, y)
function dssyax(n, A::TronSparseMatrixCSC, x, y)
    zero = 0.0

    @inbounds for i in 1:n
        y[i] = A.diag_vals[i]*x[i]
    end

    @inbounds for j in 1:n
        rowsum = zero
        for i in A.colptr[j]:A.colptr[j+1]-1
            row = A.rowval[i]
            rowsum += A.tril_vals[i]*x[row]
            @inbounds y[row] += A.tril_vals[i]*x[j]
        end
        y[j] += rowsum
    end

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
    CUDA.sync_threads()

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

    CUDA.sync_threads()

    return
end

function reorder!(B::TronDenseMatrix, A::TronDenseMatrix, indfree, nfree, iwa)
    nnz = 0
    @inbounds for j=1:nfree
        jfree = indfree[j]
        B.vals[j,j] = A.vals[jfree,jfree]
        for i=jfree+1:A.n
            if iwa[i] > 0
                nnz += 1
                B.vals[iwa[i],j] = A.vals[i,jfree]
            end
        end
    end
    B.n = nfree
    return nnz
end

# Update matrix B inplace
function reorder!(B::TronSparseMatrixCSC, A::TronSparseMatrixCSC, indfree, nfree, iwa)
    B.colptr[1] = 1
    nnz = 0
    @inbounds for j=1:nfree
        jfree = indfree[j]
        B.diag_vals[j] = A.diag_vals[jfree]
        for ip = A.colptr[jfree]:A.colptr[jfree+1]-1
            if iwa[A.rowval[ip]] > 0
                nnz = nnz + 1
                B.rowval[nnz] = iwa[A.rowval[ip]]
                B.tril_vals[nnz] = A.tril_vals[ip]
            end
        end
        B.colptr[j+1] = nnz + 1
    end
    return nnz
end

