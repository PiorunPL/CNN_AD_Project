# Functions to create a graph

# Visit
function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    # println("typeof(node): ", typeof(node))
    # println("typeof(visited): ", typeof(visited))
    # println("typeof(order): ", typeof(order))
    # println(typeof(node.output))
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited::Set{GraphNode}, order::Vector{GraphNode})
    # if isa(node, BroadcastedOperator)
    #     println(typeof(node.gradient))
    # end
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

# Sort
function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end