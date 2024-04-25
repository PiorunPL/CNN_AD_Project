# ReLU
function relu(x) return max.(x, Constant(0)) end

# Sigmoid σ
sigmoid(x::GraphNode) = BroadcastedOperator(sigmoid, x)
forward(::BroadcastedOperator{typeof(sigmoid)}, x) = return exp.(x) ./ sum(exp.(x))
backward(::BroadcastedOperator{typeof(sigmoid)}, x, g) = let
    result = exp.(-x) ./ ((exp.(-x) .+ 1) .^ 2)
    tuple(g .* result)
end

# Softmax
softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = return exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    J = diagm(y) .- y * y'
    tuple(J' * g)
end
