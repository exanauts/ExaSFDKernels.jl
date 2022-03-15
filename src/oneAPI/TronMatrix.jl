function TronDenseMatrix(I::VI, J::VI, V::oneArray, n) where {VI}
    @assert n >= 1
    @assert length(I) == length(J) == length(V)

    A = TronDenseMatrix{oneArray{Float64, 2}}(n, n, tron_zeros(oneArray{eltype(V)}, (n, n)))
    for i=1:length(I)
        @assert 1 <= I[i] <= n && 1 <= J[i] <= n && I[i] >= J[i]
        @inbounds A.vals[I[i], J[i]] += V[i]
    end

    return A
end

@inline function Base.fill!(w::oneDeviceArray{Float64,1}, val::Float64)
    tx = get_local_id()
    ty = get_group_id()

    if tx <= length(w) && ty == 1
        @inbounds w[tx] = val
    end
    barrier()

    return
end
