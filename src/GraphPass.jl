# Forward pass
function forward!(order::Vector)
    # println("=====================================================")
    # println("=====================================================")
    # println("=====================================================")
    # println("=====================================================")
    for node in order
        if isa(node, Constant)
            continue
        end
        # println("Node: ", node.name)
        # @time compute!(node)
        compute!(node)
        # println("Node output type: ", typeof(node.output))
        reset_forward!(node)
        # println(node.gradient)
    end
    return last(order).output
end
# global a = 1
# Backward pass
function backward!(order::Vector; seed=1.0)
    println("=====================================================")
    println("=====================================================")
    println("=====================================================")
    println("=====================================================")
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        if isa(node, Constant)
            continue
        end
        # if a == 2
        println("Node: ", node.name)
        @time backward!(node)
        # else
        # backward!(node)
        # end
    end
    # a = a + 1
    # if a ==3
        # throw(ArgumentError("ni mo"))
    # end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, node.output, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        # if isa(gradient, Float64) || isa(gradient, Int64)
        #     update_graph!(input, gradient)
        # else
        # println(gradient)
            update_graph!(input, gradient)
        # end
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
reset!(node::Variable) = fill!(node.gradient, 0.0f0)#node.gradient = zeros(Float32, size(node.gradient))
reset!(node::Operator) = node.gradient = isa(node.gradient, Float32) ? 0.0f0 : fill!(node.gradient, 0.0f0)#fill!(node.gradient, 0.0f0)#

reset_forward!(node::Constant) = nothing
reset_forward!(node::Variable) = nothing
reset_forward!(node::Operator) = node.gradient = isa(node.gradient, Float32) ? 0.0f0 : fill!(node.gradient, 0.0f0)#fill!(node.gradient, 0.0f0)#

# Base.fill!(x::Float32, y::Float32) = x = y

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = let 
    node.output = forward(node, node.output, [input.output for input in node.inputs]...)
end
update_graph!(node::Constant, gradient) = nothing
update_graph!(node::GraphNode, gradient) = let
    if isa(node.gradient, Float32)
        node.gradient += gradient
    else
        node.gradient .+= gradient
    end
    return nothing
end