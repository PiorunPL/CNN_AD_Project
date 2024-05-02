# Forward pass
function forward!(order::Vector)
    for node in order
        compute!(node)
        reset_forward!(node) # TODO: Prawdopodobnie trzeba będzie usunąć reset stąd i wydzielić na niego osobną metodę
    end
    return last(order).output
end

# Backward pass
function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    #display(result)
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
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
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

reset_forward!(node::Constant) = nothing
reset_forward!(node::Variable) = nothing
reset_forward!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

update_graph!(node::Constant, gradient) = nothing
update_graph!(node::GraphNode, gradient) = let
    if isnothing(node.gradient)
#        @info("
#------------------------------------------------------------------------
#New Gradient: $(gradient)")
        node.gradient = gradient
    else
#        @info("
#------------------------------------------------------------------------
#NodeName: $(node.name)
#Node.gradient: $(node.gradient)
#Gradient: $(gradient)")
        node.gradient .+= gradient
    end
    return nothing
end
