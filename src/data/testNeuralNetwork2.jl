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

logger = SimpleLogger(open("./logs/log_testNN2.txt", "w+"))
global_logger(logger)

Wh = Variable(randn(100, 2), name="wh")
Wo = Variable(randn(1, 100), name="wo")
x = Variable([2, 4], name="x")
y = Variable([6], name="y")
losses = Float64[]

function net(x, wh, wo, y)
    x̂ = dense(wh, x, sigmoid)
    x̂.name = "x̂"
    ŷ = dense(wo, x̂)
    ŷ.name = "ŷ"
    E = mean_squared_loss(y, ŷ)
    E.name = "loss"

    return topological_sort(E)
end

graph = net(x, Wh, Wo, y)
#for (i,n) in enumerate(graph)
#   println("$i. $n")
#end

# forward!(graph)
# backward!(graph)

batchsize = 200
epochs = 600
step = 0.01

@info("First forward")
push!(losses, first(forward!(graph)))

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
        tmpX = rand(Int) % 10
        tmpY = rand(Int) % 10
        tmpZ = tmpX + tmpY
    
        x.output = [tmpX, tmpY]
        y.output = [tmpZ]
        
        
        currentloss += first(forward!(graph))
        @info("Current loss: $currentloss")
        backward!(graph)
    end

    #if i == 1
        #println("Wh: $(Wh.gradient)")
        #println("Wo: $(Wo.gradient)")
    #end

    Wh.output -= step*(Wh.gradient/batchsize)
    Wo.output -= step*(Wo.gradient/batchsize)
    push!(losses, currentloss/batchsize)
    reset!(graph)
end

# display(losses)


using Plots, KittyTerminalImages
pushKittyDisplay!()

# gr()
plot(1:length(losses), losses, seriestype=:scatter) |> display

#using PyPlot
#semilogy(losses, ".")
#xlabel("epoch")
#ylabel("loss")
#grid()
