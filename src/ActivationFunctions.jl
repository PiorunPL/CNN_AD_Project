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
forward(::BroadcastedOperator{typeof(softmax)}, x::Vector) = let
# forward(::BroadcastedOperator{typeof(softmax)}, x::Vector{Float64}) = let
    # println("typeof(x): ", typeof(x))
    # println("typeof exp.(x) ./ sum(exp.(x))", typeof(exp.(x) ./ sum(exp.(x))))
    return exp.(x) ./ sum(exp.(x))
end
backward(node::BroadcastedOperator{typeof(softmax)}, x::Vector, g::Vector{Float32}) = let
    # println("typeof g: ", typeof(g))
    y = node.output
    # println("typeof y: ", typeof(y))
    J = diagm(y) .- y * y'
    # println("typeof J: ", typeof(J))
    # println("typeof J' * g: ", typeof(J' * g))
    tuple(J' * g)
end