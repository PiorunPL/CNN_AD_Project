include("../Models.jl")
include("../ActivationFunctions.jl")
include("../GraphCreation.jl")
include("../GraphPass.jl")
include("../InputOutput.jl")
include("../Layers.jl")
include("../LossFunctions.jl")
include("../Operations.jl")
include("../WeightInit.jl")
using LinearAlgebra
using ProgressMeter
using Logging
using MLDatasets: MNIST
using Random

logger = SimpleLogger(open("./logs/log_finalNetwork_1.txt", "w+"))
global_logger(logger)

function finalNet(image, filters1, filters2, wages1, wages2, y)
    a = conv(image, filters1)
    a.name = "a Convolution"
    a1 = relu(a)
    a1.name = "a1 ReLU"
    b = maxPool(a1, Constant([2,2]))
    b.name = "b MaxPool"
    c = conv(b, filters2)
    c.name = "c Convolution"
    c1 = relu(c)
    c1.name = "c1 ReLU"
    d = maxPool(c1, Constant([2,2]))
    d.name = "d MaxPool"
    e = flatten(d)
    e.name = "e Flatten"
    f = dense(wages1, e, relu)
    f.name = "f Dense"
    g = dense(wages2, f, softmax)
    g.name = "g Dense"
    return topological_sort(g)
end

function net(image, filters1, filters2, wages1, wages2, y)
    a = conv(image, filters1)
    a.name = "a Convolution"
    a1 = relu(a)
    a1.name = "a1 ReLU"
    b = maxPool(a, Constant([2,2]))
    b.name = "b MaxPool"
    c = conv(b, filters2)
    c.name = "c Convolution"
    c1 = relu(c)
    c1.name = "c1 ReLU"
    d = maxPool(c1, Constant([2,2]))
    d.name = "d MaxPool"
    e = flatten(d)
    e.name = "e Flatten"
    f = dense(wages1, e, relu)
    f.name = "f Dense"
    g = dense(wages2, f, softmax)
    g.name = "g Dense"
    #h = softmax(g)
    #h.name = "h Softmax"
    loss = mean_squared_loss(g, y, 10)
    loss.name = "Loss"
    return topological_sort(loss)
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

display(filters1)
#display(filters2)

graph = net(image, filters1, filters2, wages1, wages2, y)
test = finalNet(image, filters1, filters2, wages1, wages2, y)
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
batchsize_gradient = 1
numberOfBatchesInEpoch = length(trainDataset.targets)/batchsize
epochs = 100
step = 0.1

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

@showprogress for i in 1:epochs
    @info("
--------------------------------------------------------------
Starting epoch $i
--------------------------------------------------------------")
    currentloss = 0
    for j in 1:batchsize
        @info("
--------------------------------------------------------------
Starting batch $j in epoch $i
--------------------------------------------------------------")
    
        image.output = trainDataset.features[:,:,(i-1)*batchsize+j]
        y.output = zeros(10)
        y.output[trainDataset.targets[(i-1)*batchsize+j]+1] = 1
        
        
        currentloss += first(forward!(graph))
        @info("Current loss: $currentloss")
        backward!(graph)
    end

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
