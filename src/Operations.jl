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
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y, name="*")
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
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
    Jx = diagm(isless.(y,x))
    Jy = diagm(isless.(x,y))
    tuple(Jx' * g, Jy' * g)
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

############################################################################
# Needed for Convolution implementation
############################################################################

# Extend filter to choosen size - not sure if after all that will be needed
extend(conv_filter::GraphNode, sizes, point) = BroadcastedOperator(extend, conv_filter, x, y, name="extend")
forward(::BroadcastedOperator{typeof(extend)}, conv_filter, sizes, point) = let
    channels = length(conv_filter[1,1,:])
    filter_width = length(conv_filter[:,1,1])
    filter_height = length(conv_filter[1,:,1])
    result = zeros(sizes[1],sizes[2], channels)
    for i in point[1]:(point[1]+filter_width)
        for j in point[2]:(point[2]+filter_height)
            for k in 1:channels
                result[i,j,k] = conv_filter[i-point[1]+1,j-point[2]+1,k]
            end
        end
    end
    return result
end
backward(node::BroadcastedOperator{typeof(extend)}, conv_filter, sizes, point, g) = let
    tmpResult = g .* node.output

    channels = length(conv_filter[1,1,:])
    filter_width = length(conv_filter[:,1,1])
    filter_height = length(conv_filter[1,:,1])
    
    result = zeros(filter_width, filter_height, channels) # TODO: Prawdopodobnie można zamienić rozwiązanie na wycięcie wartości z tensora i będzie pewnie lepsze
    for i in point[1]:(point[1]+filter_width)
        for j in point[2]:(point[2]+filter_height)
            for k in 1:channels
                result[i-point[1]+1,j-point[2]+1,k] = tmpResult[i,j,k]
            end
        end
    end
    return tuple(result,0,0)
end

# Convolution
conv(image::GraphNode, filters::GraphNode) = BroadcastedOperator(conv, image, filters, name="Convolution")
forward(::BroadcastedOperator{typeof(conv)}, image, filters) = let
    # filters is an array of filters
    # image is an entry array
    filterWidth = length(filters[1][:,1,1])
    filterHeight = length(filters[1][1,:,1])

    targetWidth = length(image[:,1,1]) - filterWidth + 1
    targetHeight = length(image[1,:,1]) - filterHeight + 1
    targetChannels = length(filters)
    
    result = zeros(targetWidth, targetHeight, targetChannels)
    for i in 1:targetChannels
        filter = filters[i]
        for j in 1:targetWidth
            for k in 1:targetHeight
                result[j,k,i] = sum(image[j:(j+filterWidth-1),k:(k+filterHeight-1),:].*filter)
            end
        end
    end
    return result
end
backward(node::BroadcastedOperator{typeof(conv)}, image, filters, g) = let
    # Calculating backward of filters
    filtersResult = [zeros(size(filters[1])) for i in 1:length(filters)]

    filterWidth = length(filters[1][:,1,1])
    filterHeight = length(filters[1][1,:,1])
    filterChannels = length(filters[1][1,1,:])
    numberOfFilters = length(filters)
    
    outputWidth = length(node.output[:,1,1])
    outputHeight = length(node.output[1,:,1])
    outputChannels = length(node.output[1,1,:])

    for n in 1:numberOfFilters
        g_layer = g[:,:,n]
        for i in 1:filterChannels
            for j in 1:filterWidth
                for k in 1:filterHeight
                    filtersResult[n][j,k,i]= sum(image[j:(j+outputWidth - 1),k:(k+outputHeight-1), i].*g_layer)
                end
            end
        end
    end

    reversedFilters = [filters[i][end:-1:1, end:-1:1, :] for i in 1:length(filters)]
    g_extended = zeros(2*(filterWidth-1)+outputWidth, 2*(filterHeight-1)+outputHeight, numberOfFilters)
    g_extended[filterWidth:(filterWidth+outputWidth-1), filterHeight:(filterHeight+outputHeight-1),:] = g
    
    inputWidth = length(image[:,1,1])
    inputHeight = length(image[1,:,1])

    # Prepare refersed filters matrices
    filtersToCalculate = [Array{Float64,3}(undef,filterWidth, filterHeight, outputChannels) for i in 1:filterChannels]
    for i in 1:filterChannels
        for j in 1:outputChannels
            filtersToCalculate[i][:,:,j] = reversedFilters[j][:,:,i]
        end
    end

    inputResult = zeros(size(image))   
    # Tensors multiplication and addition for each element in image
    for i in 1:filterChannels
        for j in 1:inputWidth
            for k in 1:inputHeight
                inputResult[j,k,i] = sum(g_extended[j:(j+filterWidth-1), k:(k+filterHeight-1), :].*filtersToCalculate[i])
            end
        end
    end

    return tuple(inputResult, filtersResult)
end
