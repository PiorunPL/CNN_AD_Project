# Dense Layer
function dense(w::Variable, x::BroadcastedOperator, b::Variable, activation, preallocation_A::Constant, preallocation_x::Constant)
    return activation(mul(w, x, preallocation_A, preallocation_x) .+ b) 
end

function dense(w::Variable, x::BroadcastedOperator, b::Variable, preallocation_A::Constant, preallocation_x::Constant)
    return mul(w, x, preallocation_A, preallocation_x) .+ b
end
# function dense(w, x, activation) return activation(w * x) end
# function dense(w, x) return w * x end

# Bias
function bias(x::GraphNode, w::Variable)
    return x .+ w
end