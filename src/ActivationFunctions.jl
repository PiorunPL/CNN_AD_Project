# ReLU
function relu(x::BroadcastedOperator) return max.(x, Constant(0.0)) end

# Sigmoid σ
sigmoid(x::GraphNode) = BroadcastedOperator(sigmoid, x::BroadcastedOperator)
forward(::BroadcastedOperator{typeof(sigmoid)}, x) = let 
    return exp.(x) ./ sum(exp.(x))
end
backward(::BroadcastedOperator{typeof(sigmoid)}, x, g) = let
    result = exp.(-x) ./ ((exp.(-x) .+ 1) .^ 2)
    tuple(g .* result)
end

# Softmax
softmax(x::GraphNode) = BroadcastedOperator(softmax, x::BroadcastedOperator)
forward(::BroadcastedOperator{typeof(softmax)}, x) = let
    return exp.(x) ./ sum(exp.(x))
end
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    J = diagm(y) .- y * y'
    tuple(J' * g)
end