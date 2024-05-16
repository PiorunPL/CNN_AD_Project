# Dense Layer
function dense(w::Variable, x::BroadcastedOperator, b::Variable, activation, preallocation_A::Constant, preallocation_x::Constant)
    # println("typeof w: ", typeof(w.output))
    # println("typeof x: ", typeof(x.output))
    # println("typeof b: ", typeof(b.output))
    # println("typeof mul(w, x)", typeof(mul(w, x)))
    # println("typeof mul(w, x) .+ b: ", typeof(mul(w, x) .+ b))
    # println("typeof activation(mul(w, x) .+ b): ", typeof(activation(mul(w, x) .+ b)))
    return activation(mul(w, x, preallocation_A, preallocation_x) .+ b) 
end
# function dense(w, x, activation) return activation(w * x) end
# function dense(w, x) return w * x end

# Bias
function bias(x::GraphNode, w::Variable)
    return x .+ w
end