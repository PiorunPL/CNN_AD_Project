# Definition of used structures
abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output::T
end

mutable struct Variable <: GraphNode
    output::Array{Float32}
    gradient::Array{Float32}
    name::String
    Variable(output; name="?") = new(output, zeros(Float32,size(output)), name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs::Vector{GraphNode}
    output::Float32
    gradient::Float32
    name::String
    ScalarOperator(fun, inputs; name="?") = new{typeof(fun)}(inputs, 0, 0, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Vector{GraphNode}
    output::Array{Float32}
    gradient::Array{Float32}
    name::String
    BroadcastedOperator(fun, output_size, inputs; name="?") = new{typeof(fun)}(inputs, zeros(Float32, output_size), zeros(Float32,output_size), name)
end