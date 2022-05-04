function ExaTronKernels.TronDenseMatrix(I::VI, J::VI, V::ROCArray, n) where {VI}
    @assert n >= 1
    @assert length(I) == length(J) == length(V)

    A = TronDenseMatrix{ROCArray{Float64, 2}}(n, n, tron_zeros(ROCArray{eltype(V)}, (n, n)))
    for i=1:length(I)
        @assert 1 <= I[i] <= n && 1 <= J[i] <= n && I[i] >= J[i]
        @inbounds A.vals[I[i], J[i]] += V[i]
    end

    return A
end

@inline function ExaTronKernels.Base.fill!(w::ROCDeviceArray{Float64,1}, val::Float64)
    tx = workitemIdx().x
    bx = workgroupIdx().x

    if tx <= length(w) && bx == 1
        @inbounds w[tx] = val
    end
    AMDGPU.sync_workgroup()

    return
end
