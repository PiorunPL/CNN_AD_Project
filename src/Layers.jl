greet() = print("Hello World!")

abstract type Layer end

function DisplayLayer(layer::Layer)
    display("Not specified layer")
    display(layer.Input)
    display(layer.Output)
end

function DisplayShortLayer(layer::Layer)
    display("Not specified layer")
    display(layer.Input)
    display(layer.Output)
end

function DisplayNetwork(layer::Layer)
    DisplayLayer(layer)
    DisplayNetwork(layer.next)
end

function DisplayShortNetwork(layer::Layer)
    DisplayShortLayer(layer)
    DisplayShortNetwork(layer.next)
end
function CalcOutputResursive!(layer::Layer)
    if !isnothing(layer.previous)
        layer.Input = layer.previous.Output
    end
    CalcOutput!(layer)
    CalcOutputResursive!(layer.next)
end

function DisplayNetwork(layer::Nothing) end
function DisplayShortNetwork(layer::Nothing) end
function CalcOutputResursive!(layer::Nothing) end