module Layers

export Dense, Layer, 
    CalcOutput!, CalcOutputResursive!,
    InitLayer!,
    DisplayLayer, DisplayNetwork,
    DisplayShortLayer, DisplayShortNetwork


greet() = print("Hello World!")

abstract type Layer end

mutable struct Dense <: Layer
    previous::Union{Layer,Nothing}
    next::Union{Layer,Nothing}
    Input::Matrix
    Wages::Matrix
    Bias::Matrix
    Output::Matrix

    function Dense(input, wages::Matrix, bias::Matrix)
        new(nothing, nothing, input, wages, bias, zeros(Float64, 1, size(wages,2)))
    end

    function Dense(inputSize::Int64, outputSize::Int64)
        new(nothing,
            nothing,
            zeros(Float64, 1, inputSize), 
            zeros(Float64, inputSize, outputSize), 
            zeros(Float64, 1, outputSize), 
            zeros(Float64, 1, outputSize)
        )
    end
end

function InitLayer!(layer::Dense)
    layer.Wages = randn(size(layer.Wages,1), size(layer.Wages,2))
    layer.Bias = randn(1, size(layer.Wages,2))
end

function InitLayer!(layer::Dense, input::AbstractArray)
    layer.Input = input
    layer.Wages = randn(size(layer.Wages,1), size(layer.Wages,2))
    layer.Bias = randn(1, size(layer.Wages,2))#.+5
end

function InitLayer!(layer::Dense, input::Layer)
    display(typeof(input))
    layer.Input = input.Output
    layer.previous = input
    input.next = layer
    layer.Wages = randn(size(layer.Wages,1), size(layer.Wages,2))
    layer.Bias = randn(1, size(layer.Wages,2))
end

function CalcOutput!(layer::Dense)
    tmpZ = layer.Input * layer.Wages + layer.Bias
    layer.Output = max.(0,tmpZ)
end

function DisplayLayer(layer::Layer)
    display("Not specified layer")
    display(layer.Input)
    display(layer.Output)
end

function DisplayLayer(layer::Dense)
    display("Dense layer")
    display(layer.Input)
    display(layer.Wages)
    display(layer.Bias)
    display(layer.Output)
end

function DisplayShortLayer(layer::Layer)
    display("Not specified layer")
    display(layer.Input)
    display(layer.Output)
end

function DisplayShortLayer(layer::Dense)
    display("Dense layer")
    display(layer.Input)
    display(layer.Output)
end

function DisplayNetwork(layer::Layer)
    DisplayLayer(layer)
    DisplayNetwork(layer.next)
end
function DisplayNetwork(layer::Nothing) end

function DisplayShortNetwork(layer::Layer)
    DisplayShortLayer(layer)
    DisplayShortNetwork(layer.next)
end
function DisplayShortNetwork(layer::Nothing) end

function CalcOutputResursive!(layer::Nothing) end

# function CalcOutputResursive!(layer::Dense)
#     if !isnothing(layer.previous)
#         layer.Input = layer.previous.Output
#     end
#     CalcOutput!(layer)
#     CalcOutputResursive!(layer.next)
# end

function CalcOutputResursive!(layer::Layer)
    if !isnothing(layer.previous)
        layer.Input = layer.previous.Output
    end
    CalcOutput!(layer)
    CalcOutputResursive!(layer.next)
end

end # module Layers
