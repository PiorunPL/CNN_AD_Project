# Forward pass
function forward!(order::Vector)
    for node in order
        compute!(node)
        reset_forward!(node)
    end
    return last(order).output
end

# Backward pass
function backward!(order::Vector; seed=1.0)

    result = last(order)
    result.gradient = seed
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, node.output, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update_graph!(input, gradient)
    end
    return nothing
end

# Helper methods
function reset!(order::Vector)
    for node in order
        reset!(node)
    end
end

reset!(node::Constant) = nothing
reset!(node::Variable) = fill!(node.gradient, 0.0f0)
reset!(node::ScalarOperator) = node.gradient =  0.0f0 
reset!(node::BroadcastedOperator) = fill!(node.gradient, 0.0f0)

reset_forward!(node::Constant) = nothing
reset_forward!(node::Variable) = nothing
reset_forward!(node::ScalarOperator) = node.gradient =  0.0f0 
reset_forward!(node::BroadcastedOperator) = fill!(node.gradient, 0.0f0)

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, node.output, [input.output for input in node.inputs]...)

update_graph!(node::Constant, gradient) = nothing
update_graph!(node::Variable, gradient) = node.gradient .+= gradient
update_graph!(node::ScalarOperator, gradient) = node.gradient += gradient
update_graph!(node::BroadcastedOperator, gradient) = node.gradient .+= gradient