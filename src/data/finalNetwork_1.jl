include("../Models.jl")
include("../ActivationFunctions.jl")
include("../GraphCreation.jl")
include("../GraphPass.jl")
include("../InputOutput.jl")
include("../Layers.jl")
include("../LossFunctions.jl")
include("../Operations.jl")
include("../WeightInit.jl")
include("../NetworkLearning.jl")
using LinearAlgebra
using ProgressMeter
using Logging
using MLDatasets: MNIST
using Random

logger = SimpleLogger(open("./logs/log_finalNetwork_1.txt", "w+"))
global_logger(logger)

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

#image = Variable(randn(28,28,1)./2, name="Image")
#filters1 = Variable([randn(3,3,1)./2 for i in 1:6], name="Filters1")
#filters2 = Variable([randn(3,3,6)./2 for i in 1:16], name="Filters2")
#wages1 = Variable(randn(84,400)./2, name="Wages1")
#wages2 = Variable(randn(10,84)./2, name="Wages2")
#y = Variable(randn(10), name="Expected result y")

image = Variable(randn(28,28,1), name="Image")
filters1 = Variable([glorot_uniform(3,3,1,3*3*1) for i in 1:6], name="Filters1")
filters2 = Variable([glorot_uniform(3,3,6,3*3*6) for i in 1:16], name="Filters2")
wages1 = Variable(glorot_uniform(84,400,84*400), name="Wages1")
wages2 = Variable(glorot_uniform(10,84,10*84), name="Wages2")
y = Variable(randn(10), name="Expected result y")
bias1 = Variable(glorot_uniform(26,26,6,26*26*6), name="Bias 1")
bias2 = Variable(glorot_uniform(11,11,16,11*11*16), name="Bias 2")
bias3 = Variable(glorot_uniform(84, 84), name="Bias 3")
bias4 = Variable(glorot_uniform(10, 10), name="Bias 4")

display(filters1)
#display(filters2)

graph, test = net(image, filters1, filters2, wages1, wages2, y, bias1, bias2, bias3, bias4)
#result = forward!(graph)
#display(graph)
#backward!(graph)
#display(graph)

trainDataset = MNIST(:train)
trainData = [tuple(trainDataset.features[:,:,i], trainDataset.targets[i]) for i in 1:60000]
testDataset = MNIST(:test)
testData = [tuple(testDataset.features[:,:,i], testDataset.targets[i]) for i in 1:10000]

losses = Float64[]
batchsize = 100
testBatchSize = 100
batchsize_gradient = 100#batchsize
numberOfBatchesInEpoch = length(trainDataset.targets)/batchsize
epochs = 200
step = 0.01

shuffle!(trainData)
shuffle!(testData)

function testNetwork(testData, graph, batchsize, image, y)
    shuffle!(testData)
    accuracy = 0
    for i in 1:batchsize
        input, expectedOutput = testData[i]
        image.output = input
        y.output = expectedOutput
        result = forward!(graph)
        if netResult(result) == expectedOutput
            accuracy += 1
        end
    end
    return accuracy/batchsize*100
end

function netResult(x)
    maxValue, index = findmax(x)
    return index-1
end

#Initial test of network, before learning
accuracyArray = Float64[]
accuracy = testNetwork(testData, test,testBatchSize, image, y)
push!(accuracyArray, accuracy)

expectedOutput = Array{Float64}(undef,10)

@showprogress for i in 1:epochs
    @info("
--------------------------------------------------------------
Starting epoch $i
--------------------------------------------------------------")
    currentloss = @views @time batch_process(graph,trainData[(i-1)*batchsize+1:i*batchsize], image, y, expectedOutput)

    #if i == 1
        #println("Wh: $(Wh.gradient)")
        #println("Wo: $(Wo.gradient)")
    #end

    image.output -= step*(image.gradient/batchsize_gradient)
    wages1.output -= step*(wages1.gradient/batchsize_gradient)
    wages2.output -= step*(wages2.gradient/batchsize_gradient)
    for k in 1:length(filters1.output)
        filters1.output[k] -= step*(filters1.gradient[k]/batchsize_gradient)
    end
    for k in 1:length(filters2.output)
        filters2.output[k] -= step*(filters2.gradient[k]/batchsize_gradient)
    end
    bias1.output -= step*(bias1.gradient/batchsize_gradient)
    bias2.output -= step*(bias2.gradient/batchsize_gradient)
    bias3.output -= step*(bias3.gradient/batchsize_gradient)
    bias4.output -= step*(bias4.gradient/batchsize_gradient)

    accuracy = testNetwork(testData, test,testBatchSize, image, y)
    push!(accuracyArray, accuracy)
    push!(losses, currentloss)
    reset!(graph)
end

# display(losses)


using Plots, KittyTerminalImages
pushKittyDisplay!()

# gr()
plot(1:length(losses), losses, seriestype=:scatter) |> display
plot(1:length(accuracyArray), accuracyArray, seriestype=:scatter) |> display
image.output = trainDataset.features[:,:,40003]
display(forward!(test))
