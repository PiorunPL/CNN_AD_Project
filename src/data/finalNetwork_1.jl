using LinearAlgebra
using ProgressMeter
using Logging
using MLDatasets: MNIST
using Random
using Plots, KittyTerminalImages
using ProfileView

include("../Models.jl")
include("../ActivationFunctions.jl")
include("../GraphCreation.jl")
include("../GraphPass.jl")
include("../InputOutput.jl")
include("../Layers.jl")
include("../LossFunctions.jl")
include("../Operations.jl")
include("../WeightInit.jl")
include("../NetworkOperations.jl")

# logger = SimpleLogger(open("./logs/log_finalNetwork_1.txt", "w+"))
# global_logger(logger)

function net(image, filters1, filters2, wages1, wages2, y, bias1, bias2, bias3, bias4)
    a = conv(image, filters1)
    a.name = "a Convolution"
    a1 = bias(a, bias1)
    a1.name = "a1 Bias"
    a2 = relu(a1)
    a2.name = "a2 ReLU"
    b = maxPool(a2, Constant([2,2]))
    b.name = "b MaxPool"
    c = conv(b, filters2)
    c.name = "c Convolution"
    c1 = bias(c, bias2)
    c1.name = "c1 Bias"
    c2 = relu(c1)
    c2.name = "c2 ReLU"
    d = maxPool(c2, Constant([2,2]))
    d.name = "d MaxPool"
    e = flatten(d)
    e.name = "e Flatten"
    f = dense(wages1, e, bias3, relu)
    f.name = "f Dense"
    g = dense(wages2, f, bias4, softmax)
    g.name = "g Dense"
    loss = cross_entropy(y, g)
    loss.name = "Loss"
    return tuple(topological_sort(loss), topological_sort(g))
end

function main()
    image = Variable(randn(28,28,1), name="Image")
    filters1 = Variable(glorot_uniform(3,3,1,6,3*3*1), name="Filters1")
    filters2 = Variable(glorot_uniform(3,3,6,16,3*3*6), name="Filters2")
    wages1 = Variable(glorot_uniform(84,400,84*400), name="Wages1")
    wages2 = Variable(glorot_uniform(10,84,10*84), name="Wages2")
    y = Variable(randn(10), name="Expected result y")
    bias1 = Variable(glorot_uniform(26,26,6,26*26*6), name="Bias 1")
    bias2 = Variable(glorot_uniform(11,11,16,11*11*16), name="Bias 2")
    bias3 = Variable(glorot_uniform(84, 84), name="Bias 3")
    bias4 = Variable(glorot_uniform(10, 10), name="Bias 4")
    
    var_array = Variable[filters1, filters2, wages1, wages2, bias1, bias2, bias3, bias4]
    
    display(filters1)
    
    graph, test = net(image, filters1, filters2, wages1, wages2, y, bias1, bias2, bias3, bias4)
    
    trainDataset = MNIST(:train)
    trainData = [tuple(reshape(trainDataset.features[:,:,i],28,28,1), trainDataset.targets[i]) for i in 1:60000]
    testDataset = MNIST(:test)
    testData = [tuple(reshape(testDataset.features[:,:,i],28,28,1), testDataset.targets[i]) for i in 1:10000]
    
    losses = Float32[]
    batchsize = 100
    testBatchSize = 100
    batchsize_gradient = 100#batchsize
    numberOfBatchesInEpoch = length(trainDataset.targets)/batchsize
    epochs = 200
    step = 0.01
    
    shuffle!(trainData)
    shuffle!(testData)
    
    #Initial test of network, before learning
    accuracyArray = Float32[]
    accuracy = testNetwork(testData, test,testBatchSize, image, y)
    push!(accuracyArray, accuracy)
    
    expectedOutput = Array{Float32}(undef,10)
    
    @time @showprogress for i in 1:epochs
        @info("
    --------------------------------------------------------------
    Starting epoch $i
    --------------------------------------------------------------")
        currentloss =  batch_process(graph,trainData[(i-1)*batchsize+1:i*batchsize], image, y, expectedOutput)
    
        batch_update!(var_array, step, batchsize)
    
        accuracy = testNetwork(testData, test,testBatchSize, image, y)
        push!(accuracyArray, accuracy)
        push!(losses, currentloss)
        reset!(graph)
    end
    
    pushKittyDisplay!()
    
    # gr()
    plot(1:length(losses), losses, seriestype=:scatter) |> display
    plot(1:length(accuracyArray), accuracyArray, seriestype=:scatter) |> display
    image.output = trainDataset.features[:,:,40003]
    display(forward!(test))
end

main()

#image = Variable(randn(28,28,1)./2, name="Image")
#filters1 = Variable([randn(3,3,1)./2 for i in 1:6], name="Filters1")
#filters2 = Variable([randn(3,3,6)./2 for i in 1:16], name="Filters2")
#wages1 = Variable(randn(84,400)./2, name="Wages1")
#wages2 = Variable(randn(10,84)./2, name="Wages2")
#y = Variable(randn(10), name="Expected result y")


