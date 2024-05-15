include("Utils.jl")

# Scalar Operators
import Base: ^, sin

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n, name="^")
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x^(n-1), g * log(abs(x)) * x^n)

sin(x::GraphNode) = ScalarOperator(sin, x, name="sin")
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = return tuple(g * cos(x))

# Broadcasted Operators
import Base: *
import LinearAlgebra: mul!

# Multiplication
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x, name="mul!")
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = let 
    # println("typeof A: $(typeof(A))")
    # println("typeof x: $(typeof(x))")
    result = A * x
    # println("typeof result: $(typeof(result))")
    if isa(result, Float64)
        result = convert(Float32, result)
    end
    return result
end
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = let
    tuple(g * x', A' * g)
end

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y, name="*")
forward(::BroadcastedOperator{typeof(*)}, x, y) = let 
    # println("typeof x: $(typeof(x))")
    # println("typeof y: $(typeof(y))")
    # println("typeof x .* y: $(typeof(x .* y))")
    return x .* y
end
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    x, y = convert_to_float32(x, y)
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

# Subtraction
Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y, name="-")
forward(::BroadcastedOperator{typeof(-)}, x, y) = let
    x, y = convert_to_float32(x, y)
    return x .- y
end

backward(::BroadcastedOperator{typeof(-)}, x, y, g) = return tuple(g, -g)

# Addition
Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y, name="+")
# ogarnięte, zwraca floaty32 forward
forward(::BroadcastedOperator{typeof(+)}, x, y) = let
    x, y = convert_to_float32(x, y)
    # if isa(x, Vector{Float64})
    #     x = convert(Vector{Float32}, x)
    # end
    # if isa(y, Vector{Float64})
    #     y = convert(Vector{Float32}, y)
    # end
    # if isa(x, Vector{Float64})
    #     x = convert(Vector{Float32}, x)
    # end
    # if isa(y, Vector{Float64})
    #     y = convert(Vector{Float32}, y)
    # end
    # println("typeof x: $(typeof(x))")
    # println("typeof y: $(typeof(y))")
    result = x .+ y
    # println("typeof result: $(typeof(result))")
    # if isa(result, Array{Float64, 3})
    #     result = convert(Array{Float32, 3}, result)
    # end
    # if isa( result, Vector{Float64})
    #     result = convert(Vector{Float32}, result)
    # end
    return x .+ y
end
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = return tuple(g, g)

# Division
Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y, name="/")
forward(::BroadcastedOperator{typeof(/)}, x, y) = let 
    x, y = convert_to_float32(x, y)
    # println("typeof x: $(typeof(x))")
    # println("typeof y: $(typeof(y))")
    return x ./ y
end
backward(node::BroadcastedOperator{typeof(/)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^ 2)
    tuple(Jx' * g, Jy' * g)
end

# Sum
import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x, name="sum")
forward(::BroadcastedOperator{typeof(sum)}, x::Vector{Float32}) = let
    # println("typeof x in sum: $(typeof(x))")
    # println("typeof sum(x) in sum: $(typeof(sum(x)))")
    return sum(x)
end
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(size(x))
    tuple(𝟏 .* g)
end

# Max
import Base: max
Base.Broadcast.Broadcast(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x , y, name="max")
# forward wygląda ok
forward(::BroadcastedOperator{typeof(max)}, x, y) = let 
    
    if isa(x, Vector{Float64})
        x = convert(Vector{Float32}, x)
    end
    if isa(y, Float64)
        y = convert(Float32, y)
    end
    # println("typeof x in max: $(typeof(x))")
    # println("typeof y in max: $(typeof(y))")
    # println("typeof max(x, y) in max: $(typeof(max.(x, y)))")
    return max.(x, y)
end
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = isless.(y,x)
    Jy = isless.(x,y)
    tuple(Jx .* g, Jy .* g)
end

# Power
Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y, name="^")
forward(::BroadcastedOperator{typeof(^)}, x, y) = let
    x, y = convert_to_float32(x, y)
    return x .^ y
end
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    x, y = convert_to_float32(x, y)
    𝟏 = ones(length(node.output))
    Jx = y .* x .^ (y .- 1)
    Jy = x .^ y .* log.(abs.(x))
    tuple(Jx .* g, Jy .* g)
end

# log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x::GraphNode, name="log")
forward(::BroadcastedOperator{typeof(log)}, x) = let 
    # println("typeof x in log: $(typeof(x))")
    # println("typeof log.(x) in log: $(typeof(log.(x)))")
    result = log.(x)
    if isa(result, Vector{Float64})
        result = convert(Vector{Float32}, result)
    end
    return result
end
backward(::BroadcastedOperator{typeof(log)}, x, g) = let
    logDerivative = 1.0 ./ x
    return tuple(logDerivative .* g)
end

############################################################################
# Needed for Convolution implementation
############################################################################
# Convolution
conv(image::GraphNode, filters::GraphNode) = BroadcastedOperator(conv, image::GraphNode, filters::GraphNode, name="Convolution")
# forward(::BroadcastedOperator{typeof(conv)}, image::Matrix{Float32}, filters::Array{Float64}) = forward(reshape(convert(Array{Float64}, image), size(image)[1], size(image)[2], 1), filters)
# forward ma output git - array{Float32}
forward(::BroadcastedOperator{typeof(conv)}, image::Array, filters::Array) = let
    # println("typeof image: $(typeof(image))")
    # println("typeof filters: $(typeof(filters))")
    # filters is an array of filters
    # image is an entry array
    filterWidth = length(filters[:,1,1,1])
    filterHeight = length(filters[1,:,1,1])

    targetWidth = length(image[:,1,1]) - filterWidth + 1
    targetHeight = length(image[1,:,1]) - filterHeight + 1
    targetChannels = length(filters[1,1,1,:])
    
    result = Array{Float32,3}(undef, targetWidth, targetHeight, targetChannels)
    # result = zeros(targetWidth, targetHeight, targetChannels)
    # result = convert(Array{Float32, 3}, result)
    for i in 1:targetChannels
        filter = filters[:,:,:,i]
        # println("typeof filter: $(typeof(filter))")
        for j in 1:targetWidth
            for k in 1:targetHeight
                result[j,k,i] = sum(image[j:(j+filterWidth-1),k:(k+filterHeight-1),:].*filter)
            end
        end
    end
    # println("typeof result: $(typeof(result))")
    # return convert(Array{Float32, 3}, result)
    return result
end
backward(node::BroadcastedOperator{typeof(conv)}, image, filters, g) = let
# backward(node::BroadcastedOperator{typeof(conv)}, image, filters::Array{Float32, 4}, g::Array{Float64, 3}) = let
    # println("typeof g in backward: $(typeof(g))")
    # Calculating backward of filters
    filtersResult = zeros(size(filters))
    # println("typeof filtersResult: $(typeof(filtersResult))")

    filterWidth = length(filters[:,1,1,1])
    filterHeight = length(filters[1,:,1,1])
    filterChannels = length(filters[1,1,:,1])
    numberOfFilters = length(filters[1,1,1,:])
    
    outputWidth = length(node.output[:,1,1])
    outputHeight = length(node.output[1,:,1])
    outputChannels = length(node.output[1,1,:])

    for n in 1:numberOfFilters
        g_layer = g[:,:,n]
        for i in 1:filterChannels
            for j in 1:filterWidth
                for k in 1:filterHeight
                    filtersResult[j,k,i,n]= sum(image[j:(j+outputWidth - 1),k:(k+outputHeight-1), i].*g_layer)
                end
            end
        end
    end

    reversedFilters = filters[end:-1:1, end:-1:1, :, :] 
    g_extended = zeros(2*(filterWidth-1)+outputWidth, 2*(filterHeight-1)+outputHeight, numberOfFilters)
    g_extended[filterWidth:(filterWidth+outputWidth-1), filterHeight:(filterHeight+outputHeight-1),:] = g
    
    inputWidth = length(image[:,1,1])
    inputHeight = length(image[1,:,1])

    # Prepare refersed filters matrices
    filtersToCalculate = Array{Float64,4}(undef,filterWidth, filterHeight, outputChannels, filterChannels)
    for i in 1:filterChannels
        for j in 1:outputChannels
            filtersToCalculate[:,:,j,i] = reversedFilters[:,:,i,j]
        end
    end

    inputResult = zeros(size(image))   
    # Tensors multiplication and addition for each element in image
    for i in 1:filterChannels
        for j in 1:inputWidth
            for k in 1:inputHeight
                inputResult[j,k,i] = sum(g_extended[j:(j+filterWidth-1), k:(k+filterHeight-1), :].*filtersToCalculate[:,:,:,i])
            end
        end
    end

    return tuple(inputResult, filtersResult)
end

#MaxPool
maxPool(input::GraphNode, poolSize::GraphNode) = BroadcastedOperator(maxPool, input::GraphNode, poolSize::Constant, name="Max Pool")
# forward wygląda ok
forward(node::BroadcastedOperator{typeof(maxPool)}, input::Array{Float32, 3}, poolSize::Vector{Int64}) = let
    # println("typeof input: $(typeof(input))");
    inputWidth = length(input[:,1,1])
    inputHeight = length(input[1,:,1])
    inputChannels = length(input[1,1,:])

    outputWidth = floor(Int, inputWidth/poolSize[1])
    outputHeight = floor(Int, inputHeight/poolSize[2])

    output = Array{Float32,3}(undef, outputWidth, outputHeight, inputChannels)
    # output = zeros(outputWidth, outputHeight, inputChannels)

    for i in 1:inputChannels
        for j in 1:outputWidth
            for k in 1:outputHeight
                output[j,k,i] = maximum(input[(2*j-1):(2*j-1+poolSize[1]-1),(2*k-1):(2*k-1+poolSize[2]-1), i])
            end
        end
    end
    # println("typeof output: $(typeof(output))")
    return output
end
backward(node::BroadcastedOperator{typeof(maxPool)}, input, poolSize::Vector{Int64}, g) = let
# backward(node::BroadcastedOperator{typeof(maxPool)}, input::Array{Float64, 3}, poolSize::Vector{Int64}, g::Array{Float64, 3}) = let
    # println("typeof g in backward: $(typeof(g))")
    result = zeros(size(input))
    inputWidth,inputHeight,inputChannels = size(input)
    
    output = node.output
    outputWidth,outputHeight,outputChannels = size(output)
    #display("g_size: $(size(g))")
    #display("output_size: $(size(output))")

    for i in 1:inputChannels
        for j in 1:(outputWidth*2)
            for k in 1:(outputHeight*2)
                if input[j,k,i] == output[floor(Int,(j-1)/2)+1, floor(Int,(k-1)/2)+1, i]
                    result[j,k,i] = g[floor(Int,(j-1)/2)+1, floor(Int,(k-1)/2)+1, i]
                end
            end
        end
    end

    return tuple(result, 0.0)
end

#Flatten
flatten(input::GraphNode) = BroadcastedOperator(flatten, input::GraphNode, name="Flatten")
forward(::BroadcastedOperator{typeof(flatten)}, input::Array{Float32, 3}) = let
    # println("typeof input: $(typeof(input))")
    # println("typeof reshape(input, length(input))", typeof(reshape(input, length(input))))
    return reshape(input, length(input))
end
backward(node::BroadcastedOperator{typeof(flatten)}, input::Array{Float32, 3}, g::Vector{Float64}) = let
    # println("typeof g in backward: $(typeof(g))")
    return tuple(reshape(g, size(input)))
end