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

y = Variable([115;110;105;;111;106;99;;105;101;98;;;180;162;152;;163;150;141;;147;140;123], name="y")
image_array = zeros(4,4,2)
image_array[:,:,1] = [  1 2 3 4;
                        2 3 4 5;
                        3 4 5 6;
                        4 5 6 7]
                        
image_array[:,:,2] = [  9 8 7 6;
                        8 7 6 5;
                        7 6 5 4;
                        6 5 4 3]

filter1_array = zeros(2,2,2)
filter1_array[:,:,1] = [2 2;2 2]
filter1_array[:,:,2] = [3 3;3 3]

filter2_array = zeros(2,2,2)
filter2_array[:,:,1] = [1 2;1 2]
filter2_array[:,:,2] = [5 8;4 3]

filters = Variable([filter1_array, filter2_array], name="filters")
image = Variable(image_array, name="image")

function net1(image, filters)
    ŷ = conv(image, filters)
    ŷ.name = "ŷ"
    return topological_sort(ŷ)
end

function net2(image, filters, y, n)
    ŷ = conv(image, filters)  
    ŷ.name = "ŷ"
    E = mean_squared_loss(y,ŷ,n)
    E.name = "Loss"
    return topological_sort(E)
end

graph2 = net2(image, filters, y, 18)
result = forward!(graph2)
backward!(graph2)
display("After backward")
display(graph2)
