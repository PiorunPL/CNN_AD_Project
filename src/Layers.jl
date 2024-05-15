# Dense Layer
function dense(w::Variable, x::BroadcastedOperator, b::Variable, activation)
    return activation(w * x .+ b) 
end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

# Bias
function bias(x::GraphNode, w::Variable)
    return x .+ w
end