@inline function ExaTronKernels.dgpstep(n::Int,x,xl,
                         xu,alpha,w,
                         s,
                         I, J)
    tx = J
    ty = 1

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
    @synchronize

    return
end