# ReLU
function relu(x::BroadcastedOperator, preallocated_backward_result::Constant) return max(x, Constant(0.0f0), preallocated_backward_result) end

# # Sigmoid σ
# sigmoid(x::GraphNode) = BroadcastedOperator(sigmoid, x::BroadcastedOperator)
# forward(::BroadcastedOperator{typeof(sigmoid)}, x) = let 
#     return exp.(x) ./ sum(exp.(x))
# end
# backward(::BroadcastedOperator{typeof(sigmoid)}, x, g) = let
#     result = exp.(-x) ./ ((exp.(-x) .+ 1) .^ 2)
#     tuple(g .* result)
# end

# Softmax
softmax(x::GraphNode) = BroadcastedOperator(softmax, size(x.output), [x::GraphNode])
forward(::BroadcastedOperator{typeof(softmax)}, output, x::Vector{Float32}) = let
    return exp.(x) ./ sum(exp.(x))
end
backward(node::BroadcastedOperator{typeof(softmax)}, output, x::Vector{Float32}, g::Vector{Float32}) = let
    y = output
    J = diagm(y) .- y * y'
    tuple(J' * g)
end