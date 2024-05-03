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
forward(::BroadcastedOperator{typeof(softmax)}, x) = let
    #display("x: $x")
    #display("exp.(x): $(exp.(x))")
    #display("softmax: $(exp.(x) ./ sum(exp.(x)))")
    return exp.(x) ./ sum(exp.(x))
end
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    #display("x: $x")
    #display("g: $g")
    y = node.output
    J = diagm(y) .- y * y'
    #display("y: $y")
    #display("out: $(J' * g)")
    tuple(J' * g)
end
