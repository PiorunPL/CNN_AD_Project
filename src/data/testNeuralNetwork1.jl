include("../Models.jl")
include("../ActivationFunctions.jl")
include("../GraphCreation.jl")
include("../GraphPass.jl")
include("../InputOutput.jl")
include("../Layers.jl")
include("../LossFunctions.jl")
include("../Operations.jl")
using LinearAlgebra

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
for (i,n) in enumerate(graph)
    println("$i. $n")
end

forward!(graph)
backward!(graph)

