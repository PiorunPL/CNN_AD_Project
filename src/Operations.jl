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
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = let
    #display("--- Mul ---")
    #display("W: $(size(A))")
    #display("input: $(size(x))")
    #display("g: $(size(g))}")
    tuple(g * x', A' * g)
end
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y, name="*")
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    #display("--- * ---")
    #display("x: $(size(x))")
    #display("y: $(size(y))")
    #display("g: $(size(g))}")
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

# Subtraction
Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y, name="-")
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = return tuple(g, -g)

# Addition
Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y, name="+")
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = return tuple(g, g)

# Division
Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y, name="/")
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^ 2)
    tuple(Jx' * g, Jy' * g)
end

# Sum
import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x, name="sum")
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    #display("--- SUM ---")
    #display("x: $x")
    #display("g: $g")
    𝟏 = ones(size(x))
    #display("𝟏: $𝟏")
    tuple(𝟏 .* g)
end

# Max
import Base: max
Base.Broadcast.Broadcast(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x , y, name="max")
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = isless.(y,x)
    Jy = isless.(x,y)
    tuple(Jx .* g, Jy .* g)
end

# Power
Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y, name="^")
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    #display("--- ^ ---")
    𝟏 = ones(length(node.output))
    #display("x: $x")
    #display("y: $y")
    #display("g: $g")
    Jx = y .* x .^ (y .- 1)
    #display("Jx: $Jx")
    Jy = x .^ y .* log.(abs.(x))
    #display("Jy: $Jy")
    tuple(Jx .* g, Jy .* g)
end

# log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x, name="log")
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = let
    logDerivative = 1.0 ./ x
    return tuple(logDerivative .* g)
end

############################################################################
# Needed for Convolution implementation
############################################################################
# Convolution
conv(image::GraphNode, filters::GraphNode) = BroadcastedOperator(conv, image, filters, name="Convolution")
forward(::BroadcastedOperator{typeof(conv)}, image, filters) = let
    # filters is an array of filters
    # image is an entry array
    filterWidth = length(filters[:,1,1,1])
    filterHeight = length(filters[1,:,1,1])

    targetWidth = length(image[:,1,1]) - filterWidth + 1
    targetHeight = length(image[1,:,1]) - filterHeight + 1
    targetChannels = length(filters[1,1,1,:])
    
    result = zeros(targetWidth, targetHeight, targetChannels)
    for i in 1:targetChannels
        filter = filters[:,:,:,i]
        for j in 1:targetWidth
            for k in 1:targetHeight
                result[j,k,i] = sum(image[j:(j+filterWidth-1),k:(k+filterHeight-1),:].*filter)
            end
        end
    end
    return result
end
function backward(node::BroadcastedOperator{typeof(conv)}, image, filters, g)
    let
    # Calculating backward of filters
    # filtersResult = zeros(Float64,size(filters))

    filterHeight, filterWidth, filterChannels, numberOfFilters = size(filters)
    outputHeight, outputWidth, outputChannels = size(node.output)
    imageHeight, imageWidth, imageChannels = size(image)
    filtersResult = Array{Float64,4}(undef,size(filters))

    @inbounds @views for n in 1:numberOfFilters
        glayer=g[:,:,n]
        @inbounds for i in 1:filterChannels
            @inbounds for k in 1:filterWidth
                @inbounds  for j in 1:filterHeight
                    filtersResult[j,k,i,n] = sum(image[j:(j+outputWidth - 1),k:(k+outputHeight-1), i].*glayer)
                end
            end
        end
    end

    # for l in 1:numberOfFilters,
    #    k in 1:filterChannels,
    #    i in 1:filterWidth,
    #    m in 1:outputWidth,
    #    j in 1:filterHeight,
    #    n in 1:outputHeight
    #        filtersResult[j,i,k,l] += image[j+n-1,i+m-1,k]*g[n,m,l]
    # end

    # @time for k in 1:filterChannels,
    #     i in 1:filterWidth,
    #     j in 1:filterHeight,
    #     l in 1:numberOfFilters,
    #     m in 1:outputWidth,
    #     n in 1:outputHeight
    #        filtersResult[j,i,k,l] += image[j+n-1,i+m-1,k]*g[n,m,l]
    # end

    inputResult = zeros(Float64, size(image))

    # @inbounds for i in 1:targetChannels,
    #     j in 1:imageChannels,
    #     filterCol in 1:filterWidth,
    #     k in 1:targetWidth,
    #     l in 1:targetHeight,
    #     m in 1:filterHeight
    #         result[l,k,i] += image[l+m-1,filterCol+k-1,j]*filters[m,filterCol,j,i]
    # end
    # @inbounds for i in 1:outputChannels,
    #     l in 1:imageChannels,
    #     filterCol in 1:filterWidth,
    #     k in 1:outputWidth,
    #     m in 1:outputHeight,
    #     n in 1:filterHeight
    #         # filtersResult[n,filterCol,l,i] += image[m+n-1,filterCol+k-1,l]*g[m,k,i]
    #         inputResult[m+n-1,k+filterCol-1,l] += filters[n,filterCol,l,i] * g[m,k,i]    
    # end

    reversedFilters = @view filters[end:-1:1, end:-1:1, :, :] 
    g_extended = zeros(2*(filterWidth-1)+outputWidth, 2*(filterHeight-1)+outputHeight, numberOfFilters)
    g_extended[filterWidth:(filterWidth+outputWidth-1), filterHeight:(filterHeight+outputHeight-1),:] = g
    
    inputWidth = length(@view image[:,1,1])
    inputHeight = length(@view image[1,:,1])

    # display("----------------------------------------")
    # Prepare refersed filters matrices
    filtersToCalculate = Array{Float32,4}(undef,filterWidth, filterHeight, outputChannels, filterChannels)
    @inbounds for i in 1:filterChannels
        @inbounds for j in 1:outputChannels
            filtersToCalculate[:,:,j,i] = @view reversedFilters[:,:,i,j]
        end
    end

    inputResult = zeros(size(image))   
    # Tensors multiplication and addition for each element in image
    @inbounds @views for i in 1:filterChannels
        filterCalc = filtersToCalculate[:,:,:,i]
        @inbounds for j in 1:inputWidth
            jCalc = (j+filterWidth-1)
            @inbounds @views for k in 1:inputHeight
                inputResult[k,j,i] = sum(g_extended[k:(k+filterHeight-1),j:jCalc, :].*filterCalc)
            end
        end
    end

    return tuple(inputResult, filtersResult)
end
end

#MaxPool
maxPool(input::GraphNode, poolSize::GraphNode) = BroadcastedOperator(maxPool, input, poolSize, name="Max Pool")
forward(node::BroadcastedOperator{typeof(maxPool)}, input, poolSize) = let
    inputWidth = length(input[:,1,1])
    inputHeight = length(input[1,:,1])
    inputChannels = length(input[1,1,:])

    outputWidth = floor(Int, inputWidth/poolSize[1])
    outputHeight = floor(Int, inputHeight/poolSize[2])

    output = zeros(outputWidth, outputHeight, inputChannels)

    for i in 1:inputChannels
        for j in 1:outputWidth
            for k in 1:outputHeight
                output[j,k,i] = maximum(input[(2*j-1):(2*j-1+poolSize[1]-1),(2*k-1):(2*k-1+poolSize[2]-1), i])
            end
        end
    end

    return output
end
backward(node::BroadcastedOperator{typeof(maxPool)}, input, poolSize, g) = let
    result = zeros(size(input))
    inputWidth,inputHeight,inputChannels = size(input)
    
    output = node.output
    outputWidth,outputHeight,outputChannels = size(output)
    #display("g_size: $(size(g))")
    #display("output_size: $(size(output))")

    for i in 1:inputChannels
        for k in 1:(outputHeight*2)
            for j in 1:(outputWidth*2)
                if input[j,k,i] == output[floor(Int,(j-1)/2)+1, floor(Int,(k-1)/2)+1, i]
                    result[j,k,i] = g[floor(Int,(j-1)/2)+1, floor(Int,(k-1)/2)+1, i]
                end
            end
        end
    end

    return tuple(result, 0)
end

#Flatten
flatten(input::GraphNode) = BroadcastedOperator(flatten, input, name="Flatten")
forward(::BroadcastedOperator{typeof(flatten)}, input) = let
    return reshape(input, length(input))
end
backward(node::BroadcastedOperator{typeof(flatten)}, input, g) = let
    #display("--- Flatten ---")
    #display("input: $(size(input))")
    #display("g: $(size(g))")
    #display("g_output: $(size(reshape(g,size(input))))")
    #display("g: $(reshape(g, size(input)))")
    return tuple(reshape(g, size(input)))
end


