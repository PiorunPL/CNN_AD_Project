include("../Models.jl")
include("../ActivationFunctions.jl")
include("../GraphCreation.jl")
include("../GraphPass.jl")
include("../InputOutput.jl")
include("../Layers.jl")
include("../LossFunctions.jl")
include("../Operations.jl")
using LinearAlgebra
using ProgressMeter
using Logging

logger = SimpleLogger(open("./logs/log_testConv_1.txt", "w+"))
global_logger(logger)

y = Variable([10;22;;4;11;;;10;12;;10;23], name="y")
image_array = zeros(4,5,2)
image_array[:,:,1] = [  5 7 4 3 5;
                        8 9 1 4 6;
                        20 8 9 7 10;
                        0 1 2 5 0]
                        
image_array[:,:,2] = [  4 3 2 1 9;
                        5 9 0 8 0;
                        7 8 25 18 26;
                        8 4 7 3 1]

image = Variable(image_array, name="image")

function net2(image, y, n)
    ŷ = maxPool(image, Constant([2,2]))  
    ŷ.name = "ŷ"
    E = mean_squared_loss(y,ŷ,n)
    E.name = "Loss"
    return topological_sort(E)
end

graph2 = net2(image, y, 8)
result = forward!(graph2)
backward!(graph2)
display("After backward")
display(graph2)
