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

y = Variable([6], name="y")
image_array = zeros(6,6,2)
image_array[:,:,1] = [  1 4 3 1 4 5;
                        5 3 2 4 1 2;
                        1 4 2 5 3 2;
                        5 7 3 2 5 6;
                        2 4 5 1 2 4;
                        4 3 2 5 6 2]
image_array[:,:,2] = [  1 2 3 4 5 6;
                        6 7 8 9 10 10;
                        1 3 2 8 7 3;
                        2 6 3 4 7 5;
                        1 3 2 4 5 6;
                        4 6 2 3 4 7]

filter1_array = zeros(3,3,2)
filter1_array[:,:,1] = [2 1 3;1 1 2;2 3 1]
filter1_array[:,:,2] = [5 2 5;2 5 2;5 2 5]

filter2_array = zeros(3,3,2)
filter2_array[:,:,1] = [3 5 1;2 7 3;3 3 1]
filter2_array[:,:,2] = [2 3 2;2 2 2;4 3 1]

filters = Variable([filter1_array, filter2_array], name="filters")
image = Variable(image_array, name="image")

# display(filters)
# display(image)


function net(image, filters)
    ŷ = conv(image, filters)  
    ŷ.name = "ŷ"
    return topological_sort(ŷ)
end

graph = net(image, filters)
#for (i,n) in enumerate(graph)
#   println("$i. $n")
#end

result = forward!(graph)
display(graph)
display(result)
