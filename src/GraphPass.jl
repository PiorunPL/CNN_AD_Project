# Forward pass
function forward!(order::Vector)
    for node in order
        # println("Node: ", node)
        compute!(node)
        # if isa(node, BroadcastedOperator)
        #     println(typeof(node.gradient))
        # end
        reset_forward!(node)
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
        # println(typeof(gradient))
        if isa(gradient, Float64) || isa(gradient, Int64)
            update_graph!(input, gradient)
        else
            # dimensions = length(size(gradient))
            # if(dimensions == 1)
            #     gradient = reshape(gradient, size(gradient)[1], 1, 1, 1)
            # elseif (dimensions == 2)
            #     gradient = reshape(gradient, size(gradient)[1], size(gradient)[2], 1, 1)
            # elseif (dimensions == 3)
            #     gradient = reshape(gradient, size(gradient)[1], size(gradient)[2], size(gradient)[3], 1)
            # end
            # println(dimensions)
            # println(typeof(gradient))
            # println(typeof(input))
            update_graph!(input, gradient)
        end
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
compute!(node::Operator) = let 
    node.output = forward(node, [input.output for input in node.inputs]...)
    # dimensions = length(size(node.output))
    # if(dimensions == 1)
    #     node.output = reshape(node.output, size(node.output)[1], 1, 1, 1)
    # elseif (dimensions == 2)
    #     node.output = reshape(node.output, size(node.output)[1], size(node.output)[2], 1, 1)
    # elseif (dimensions == 3)
    #     node.output = reshape(node.output, size(node.output)[1], size(node.output)[2], size(node.output)[3], 1)
    # end
    # println(typeof(node.output))
end
update_graph!(node::Constant, gradient) = nothing
update_graph!(node::GraphNode, gradient) = let
    # if isa(gradient, Float64)
    #     gradient = [gradient]
    # end
    if isnothing(node.gradient)
#        @info("
#------------------------------------------------------------------------
#New Gradient: $(gradient)")
        # println(typeof(gradient))
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

# update_graph!(node::GraphNode, gradient::Float64) = let
#     if isa(node, Constant)
#         return nothing
#     end
#     if isnothing(node.gradient)
#         node.gradient = gradient
#     else
#         node.gradient .+= gradient
#     end
#     return nothing
# end

# update_graph!(node::GraphNode, gradient::Array{Float64}) = let
#     if isa(node, Constant)
#         return nothing
#     end
#     if isnothing(node.gradient)
#         node.gradient = gradient
#     else
#         node.gradient .+= gradient
#     end
#     return nothing
# end

# update_graph!(node::GraphNode, gradient::Vector{Float64}) = let
#     if isa(node, Constant)
#         return nothing
#     end
#     if isnothing(node.gradient)
#         node.gradient = gradient
#     else
#         node.gradient .+= gradient
#     end
#     return nothing
# end

# update_graph!(node::GraphNode, gradient::Matrix{Float64}) = let
#     if isa(node, Constant)
#         return nothing
#     end
#     if isnothing(node.gradient)
#         node.gradient = gradient
#     else
#         node.gradient .+= gradient
#     end
#     return nothing
# end

# update_graph!(node::GraphNode, gradient::Int64) = let
#     if isa(node, Constant)
#         return nothing
#     end
#     if isnothing(node.gradient)
#         node.gradient = gradient
#     else
#         node.gradient .+= gradient
#     end
#     return nothing
# end