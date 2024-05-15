# Dense Layer
function dense(w::Variable, x::BroadcastedOperator, b::Variable, activation)
    # println("typeof w: ", typeof(w))
    # println("typeof x: ", typeof(x))
    # println("typeof b: ", typeof(b))
    # println("typeof activation: ", typeof(activation))
    # println("typeof w * x .+ b: ", typeof(w * x .+ b))
    return activation(w * x .+ b) 
end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

# Bias
function bias(x::GraphNode, w::Variable)
    # println("typeof x: ", typeof(x))
    # println("typeof w: ", typeof(w))
    # println("typeof x .+ w: ", typeof(x .+ w)) 
    return x .+ w
end