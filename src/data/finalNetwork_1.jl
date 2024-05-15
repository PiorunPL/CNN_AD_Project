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
using LinearAlgebra
using ProgressMeter
using Logging
using MLDatasets: MNIST
using Random
Random.seed!(2222)
using Plots
# using ProfileView
# using Plots, KittyTerminalImages

# logger = SimpleLogger(open("./logs/log_finalNetwork_1.txt", "w+"))
# global_logger(logger)

function net(image::Variable, filters1::Variable, filters2::Variable, wages1::Variable, wages2::Variable, y::Variable, 
        bias1::Variable, bias2::Variable, bias3::Variable, bias4::Variable)
        # konwolucja git, jest typu array{float32, 3}
    a = conv(image, filters1)
    a.name = "a Convolution"
    a1 = bias(a, bias1)
    a1.name = "a1 Bias"
    # relu chyba ok
    a2 = relu(a1)
    a2.name = "a2 ReLU"
    # maxpool chyab ok
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

    image = Variable(Array{Float32, 3}(undef,28,28,1), name="Image")
    filters1 = Variable(glorot_uniform(3,3,1,6,Int32(3*3*1)), name="Filters1")
    filters2 = Variable(glorot_uniform(3,3,6,16,Int32(3*3*6)), name="Filters2")
    wages1 = Variable(glorot_uniform(84,400,Int32(84*400)), name="Wages1")
    wages2 = Variable(glorot_uniform(10,84,Int32(10*84)), name="Wages2")
    y = Variable(Vector{Float32}(undef,10), name="Expected result y")
    bias1 = Variable(glorot_uniform(26,26,6,Int32(26*26*6)), name="Bias 1")
    bias2 = Variable(glorot_uniform(11,11,16,Int32(11*11*16)), name="Bias 2")
    bias3 = Variable(glorot_uniform(84, Int32(84)), name="Bias 3")
    bias4 = Variable(glorot_uniform(10, Int32(10)), name="Bias 4")

    # all changed to array
    # println(typeof(image.output)) #array
    # println(typeof(filters1.output))#array
    # println(typeof(filters2.output))#array
    # println(typeof(wages1.output))#matrix
    # println(typeof(wages2.output))#matrix
    # println(typeof(y.output))#Vector
    # println(typeof(bias1.output))#array
    # println(typeof(bias2.output))#array
    # println(typeof(bias3.output))#Vector
    # println(typeof(bias4.output))#Vector

    # println("typeof image.output: ", typeof(image.output))
    # println("typeof filters1.output: ", typeof(filters1.output))
    # println("typeof filters2.output: ", typeof(filters2.output))
    # println("typeof wages1.output: ", typeof(wages1.output))
    # println("typeof wages2.output: ", typeof(wages2.output))
    # println("typeof y.output: ", typeof(y.output))
    # println("typeof bias1.output: ", typeof(bias1.output))
    # println("typeof bias2.output: ", typeof(bias2.output))
    # println("typeof bias3.output: ", typeof(bias3.output))
    # println("typeof bias4.output: ", typeof(bias4.output))

    

    var_array = Variable[filters1, filters2, wages1, wages2, bias1, bias2, bias3, bias4]

    # display(filters1)
    graph, test = net(image, filters1, filters2, wages1, wages2, y, bias1, bias2, bias3, bias4)

    trainDataset = MNIST(:train)
    trainData = [tuple(reshape(trainDataset.features[:,:,i],28,28,1), trainDataset.targets[i]) for i in 1:60000]
    testDataset = MNIST(:test)
    testData = [tuple(reshape(testDataset.features[:,:,i],28,28,1), testDataset.targets[i]) for i in 1:10000]

    # println("typeof(trainData): ", typeof(trainData))
    # println("typeof(testData): ", typeof(testData))

    losses = Float64[]
    batchsize = 100
    testBatchSize = 100
    batchsize_gradient = 100#batchsize
    numberOfBatchesInEpoch = length(trainDataset.targets)/batchsize
    epochs = 200
    step = 0.01f0

    shuffle!(trainData)
    shuffle!(testData)

    # println("111")

    #Initial test of network, before learning
    accuracyArray = Float64[]
    # println("typeof test: ", typeof(test))
    # println("typeof graph: ", typeof(graph))
    # println("typeof image: ", typeof(image))
    # println("typeof y: ", typeof(y))
    accuracy = testNetwork(testData, test, testBatchSize, image, y)
    # println("ddd")
    push!(accuracyArray, accuracy)

    # println("222")
    expectedOutput = Array{Float32}(undef,10)

    @time @showprogress for i in 1:epochs
        @info("
    --------------------------------------------------------------
    Starting epoch $i
    --------------------------------------------------------------")
        currentloss = batch_process(graph,trainData[(i-1)*batchsize+1:i*batchsize], image, y, expectedOutput)
        # println("typeof var_array: ", typeof(var_array))
        # println("typeof step: ", typeof(step))
        # println("typeof batchsize: ", typeof(batchsize))
        batch_update!(var_array, step, batchsize)

        accuracy = testNetwork(testData, test,testBatchSize, image, y)
        println("Accuracy: ", accuracy)
        push!(accuracyArray, accuracy)
        push!(losses, currentloss)
        reset!(graph)
    end

    # pushKittyDisplay!()

    plot(1:length(losses), losses, seriestype=:scatter)
    plot(1:length(accuracyArray), accuracyArray, seriestype=:scatter)
    # println("ttttt")
    println(typeof(trainDataset.features[:,:,40003]))
    image.output = trainDataset.features[:,:,40003]
    image.output = reshape(image.output, 28, 28, 1)
    display(forward!(test))
end

# @profview main()
main()