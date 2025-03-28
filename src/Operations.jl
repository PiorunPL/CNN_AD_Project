# Scalar Operators
import Base: ^, sin
import Base: *
import LinearAlgebra: mul!

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, [x, n], name="^")
forward(::ScalarOperator{typeof(^)}, output, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, output, x, n, g) = tuple(g * n * x^(n-1), g * log(abs(x)) * x^n)

sin(x::GraphNode) = ScalarOperator(sin, x, name="sin")
forward(::ScalarOperator{typeof(sin)}, output, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, output, x, g) = return tuple(g * cos(x))

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, [x, y], name="*")
forward(::ScalarOperator{typeof(*)}, output, x, y) = return x * y
backward(::ScalarOperator{typeof(*)}, output, x, y, g) = return tuple(y * g, x * g)

# Broadcasted Operators


# Multiplication
mul(A::GraphNode, x::GraphNode, preallocated_A::Constant, preallocated_x::Constant) = BroadcastedOperator(mul!,size(x.output,2) == 1 ? (size(A.output,1)) : (size(A.output,1), size(x.output,2)), [A, x, preallocated_A, preallocated_x], name="mul!")
forward(::BroadcastedOperator{typeof(mul!)}, output, A, x, preallocated_A, preallocated_x) = let 
    mul!(output, A, x)
    # if isa(result, Float64)
    #     result = convert(Float32, result)
    # end
    return output
end
backward(::BroadcastedOperator{typeof(mul!)}, output, A, x, preallocated_A, preallocated_x, g) = let
    # println("size of g: ", size(g))
    # println("size of A: ", size(A))
    # println("size of x: ", size(x))
    mul!(preallocated_A, g, x')
    mul!(preallocated_x, A', g)
    tuple(preallocated_A, preallocated_x, 0.0f0, 0.0f0, 0.0f0)
end

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, (size(x.output)), [x::GraphNode, y::GraphNode], name="*")
forward(::BroadcastedOperator{typeof(*)}, output, x::Vector{Float32}, y::Vector{Float32}) = let 
    @inbounds @simd for i in eachindex(output)
        output[i] = x[i] * y[i]
    end
    return output
end
backward(node::BroadcastedOperator{typeof(*)}, output, x::Vector{Float32}, y::Vector{Float32}, g) = let
    𝟏 = ones(length(output))
    𝟏 = convert(Vector{Float32}, 𝟏)
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

# Subtraction
Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, (size(x.output)), [x::GraphNode, y::GraphNode], name="-")
forward(::BroadcastedOperator{typeof(-)}, output, x, y) = let
    @inbounds @simd for i in eachindex(output)
        output[i] = x[i] - y[i]
    end
    # return x .- y
    return output
end
backward(::BroadcastedOperator{typeof(-)}, output, x, y, g) = return tuple(g, -g)

# Addition
Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, (size(x.output)), [x::GraphNode, y::GraphNode], name="++")
forward(node::BroadcastedOperator{typeof(+)}, output, x, y) = let

    @inbounds @simd for i in eachindex(output)
        output[i] = x[i] + y[i]
    end
    # return x .+ y
    return output
end
backward(::BroadcastedOperator{typeof(+)}, output, x, y, g) = let
    # println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    return tuple(g, g)
end

# Division
# Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x::GraphNode, y::GraphNode, name="/")
# forward(::BroadcastedOperator{typeof(/)}, x, y) = let 
#     return x ./ y
# end
# backward(node::BroadcastedOperator{typeof(/)}, x, y, g) = let
#     𝟏 = ones(length(node.output))
#     Jx = diagm(𝟏 ./ y)
#     Jy = (-x ./ y .^ 2)
#     tuple(Jx' * g, Jy' * g)
# end


import Base: sum
sum(x::GraphNode) = ScalarOperator(sum, [x::GraphNode], name="sum")
forward(::ScalarOperator{typeof(sum)}, output, x::Vector{Float32}) = let
    # output = sum(x)
    return sum(x)
end
backward(::ScalarOperator{typeof(sum)}, output, x::Vector{Float32}, g) = let
    g = Float32(g)
    𝟏 = ones(size(x))
    𝟏 = convert(Vector{Float32}, 𝟏)
    tuple(𝟏 .* g)
end
# Sum
# import Base: sum
# sum(x::GraphNode) = BroadcastedOperator(sum, x::GraphNode, name="sum")
# forward(::BroadcastedOperator{typeof(sum)}, x::Vector{Float32}) = let
#     return sum(x)
# end
# backward(::BroadcastedOperator{typeof(sum)}, x::Vector{Float32}, g) = let
#     g = Float32(g)
#     𝟏 = ones(size(x))
#     𝟏 = convert(Vector{Float32}, 𝟏)
#     tuple(𝟏 .* g)
# end

import Base: max
max(x::GraphNode, y::GraphNode, preallocated_backward_result::Constant) = BroadcastedOperator(max, (size(x.output)), [x::GraphNode, y::GraphNode, preallocated_backward_result::Constant], name="max")
forward(::BroadcastedOperator{typeof(max)}, output, x, y, preallocated_backward_result) = let 
    # if isa(y, Float64)
    #     y = convert(Float32, y)
    # end
    @inbounds @simd for i in eachindex(output)
        output[i] = max(x[i], y)
    end
    return output
end
backward(::BroadcastedOperator{typeof(max)}, output, x, y, preallocated_backward_result, g) = let
    # if isa(y, Float64)
    #     y = convert(Float32, y)
    # end
    @inbounds @simd for i in eachindex(preallocated_backward_result)
        preallocated_backward_result[i] = isless(y, x[i])*g[i]
    end
    # Jx = isless.(y,x)
    tuple(preallocated_backward_result, 0.0f0)
end

# Max
# import Base: max
# Base.Broadcast.Broadcast(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x::GraphNode , y::GraphNode, name="max")
# forward(::BroadcastedOperator{typeof(max)}, x, y) = let 
#     if isa(y, Float64)
#         y = convert(Float32, y)
#     end
#     return max.(x, y)
# end
# backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
#     if isa(y, Float64)
#         y = convert(Float32, y)
#     end
#     Jx = isless.(y,x)
#     Jy = isless.(x,y)
#     tuple(Jx .* g, Jy .* g)
# end

# # Power
# Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y, name="^")
# forward(::BroadcastedOperator{typeof(^)}, x, y) = let
#     return x .^ y
# end
# backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
#     𝟏 = ones(length(node.output))
#     Jx = y .* x .^ (y .- 1)
#     Jy = x .^ y .* log.(abs.(x))
#     tuple(Jx .* g, Jy .* g)
# end

# log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, (size(x.output)), [x::GraphNode], name="log")
forward(::BroadcastedOperator{typeof(log)}, output, x::Vector{Float32}) = let 
    # return log.(x)
    @inbounds @simd for i in eachindex(output)
        output[i] = log(x[i])
    end
    return output
end
backward(::BroadcastedOperator{typeof(log)}, output, x::Vector{Float32}, g) = let
    # println("typeof g", typeof(g))
    # println("size of g", size(g))
    # if isa(g, Matrix)
    #     g = g[1,:]
    # end
    logDerivative = 1.0f0 ./ x
    return tuple(logDerivative .* g)
end

############################################################################
# Needed for Convolution implementation
############################################################################
# Convolution
conv(image::GraphNode, filters::GraphNode, inputResult::GraphNode, filtersResult::GraphNode) = BroadcastedOperator(conv, (size(image.output,1) - size(filters.output,1)+1, size(image.output,2) - size(filters.output,2)+1, size(filters.output, 4)), [image::GraphNode, filters::GraphNode, inputResult::GraphNode, filtersResult::GraphNode], name="Convolution")
forward(node::BroadcastedOperator{typeof(conv)}, output, image::Array{Float32, 3}, filters::Array{Float32, 4}, inputResult::Array{Float32, 3}, filtersResult::Array{Float32, 4}) = let
    filterHeight, filterWidth, filterChannels, targetChannels = size(filters)
    imageHeight, imageWidth, imageChannels = size(image)

    targetWidth = imageWidth - filterWidth + 1
    targetHeight = imageHeight - filterHeight + 1
    
    # println("targetWidth: ", targetWidth, " targetHeight: ", targetHeight, " targetChannels: ", targetChannels)
    fill!(output, 0.0f0)
    # result = zeros(Float32, targetWidth, targetHeight, targetChannels)
    # println("before filllllll")
    # fill!(node.output, 0.0f0)
    # @time result = node.output
    @inbounds for i in 1:targetChannels,
        j in 1:imageChannels,
        filterCol in 1:filterWidth,
        k in 1:targetWidth,
        l in 1:targetHeight,
        m in 1:filterHeight
        # println("8====================================================================>")
        # @time a = image[l+m-1,filterCol+k-1,j]
        # @time b = filters[m,filterCol,j,i]
        # @time c = a*b
        # println("typeof a ", typeof(a))
        # println("typeof b ", typeof(b))
        # println("typeof c ", typeof(c))
        # @time node.output[l,k,i] += c
        # node.output[l,k,i] += image[l+m-1,filterCol+k-1,j]*filters[m,filterCol,j,i]
        # result[l,k,i] += image[l+m-1,filterCol+k-1,j]*filters[m,filterCol,j,i]
        output[l,k,i] += image[l+m-1,filterCol+k-1,j]*filters[m,filterCol,j,i]
        # @time result[l,k,i] = result[l,k,i] + c
    end
    # return node.output
    # return result
    return output
end
backward(node::BroadcastedOperator{typeof(conv)}, output, image::Array{Float32, 3}, filters::Array{Float32, 4}, inputResult::Array{Float32, 3}, filtersResult::Array{Float32, 4}, g::Array{Float32, 3}) = let

    filterHeight, filterWidth, filterChannels, numberOfFilters = size(filters)
    outputHeight, outputWidth, outputChannels = size(output)

    fill!(filtersResult, 0.0f0)
    # @inbounds for l in 1:numberOfFilters,
    #    k in 1:filterChannels,
    #    i in 1:filterWidth,
    #    m in 1:outputWidth,
    #    j in 1:filterHeight,
    #    n in 1:outputHeight
    #        filtersResult[j,i,k,l] += image[j+n-1,i+m-1,k]*g[n,m,l]
    # end

    fill!(inputResult, 0.0f0)
    # inputResult = zeros(Float32, size(image))

    @inbounds for i in 1:outputChannels,
        l in 1:filterChannels,
        filterCol in 1:filterWidth,
        k in 1:outputWidth,
        m in 1:outputHeight,
        n in 1:filterHeight
            filtersResult[n,filterCol,l,i] += image[m+n-1,filterCol+k-1,l]*g[m,k,i]
            inputResult[m+n-1,k+filterCol-1,l] += filters[n,filterCol,l,i] * g[m,k,i]    
    end

    return tuple(inputResult, filtersResult, 0.0f0, 0.0f0)
end

#MaxPool
maxPool(input::GraphNode, poolSize::GraphNode, res::Constant) = BroadcastedOperator(maxPool, (floor(Int32, size(input.output,1)/poolSize.output[1]), floor(Int32, size(input.output,2)/poolSize.output[2]), size(input.output,3)), [input::GraphNode, poolSize::Constant, res::Constant], name="Max Pool")
forward(node::BroadcastedOperator{typeof(maxPool)}, output, input::Array{Float32, 3}, poolSize::Vector{Int64}, res::Array{Float32, 3}) = let
    inputHeight, inputWidth, inputChannels = size(input)

    outputWidth = floor(Int32, inputWidth/poolSize[1])
    outputHeight = floor(Int32, inputHeight/poolSize[2])
    fill!(output, -Inf)

    @inbounds for i in 1:inputChannels,
        k in 1:outputWidth*2,
        j in 1:outputHeight*2
            row = trunc(Int32,(j+1)/2)
            col = trunc(Int32,(k+1)/2)
            if(output[row,col,i] < input[j,k,i])
                output[row,col,i] = input[j,k,i]
            end
    end
    return output
end
backward(node::BroadcastedOperator{typeof(maxPool)}, output, input::Array{Float32, 3}, poolSize::Vector{Int64}, res::Array{Float32, 3}, g::Array{Float32, 3}) = let
    # result = zeros(Float32, size(input))
    res = fill!(res, 0.0f0)
    # result = convert(Array{Float32, 3}, result)
    inputWidth,inputHeight,inputChannels = size(input)
    
    # output = node.output
    outputWidth,outputHeight,outputChannels = size(output)
    
    # copy = @view node.output[:,:,:]
    @inbounds for i in 1:inputChannels
        for k in 1:(outputHeight*2)
            for j in 1:(outputWidth*2)
                # println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                # println("rows")
                row_number = trunc(Int32,(j-1)/2)+1
                col_number = trunc(Int32,(k-1)/2)+1
                # println("iffffffffffffffffff")
                # println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                # @time a = 
                if input[j,k,i] == output[row_number, col_number, i]
                    # println("insideeeeeeeeeeeeeeeeeeee iffffffffffffffffffff")
                    res[j,k,i] = g[row_number, col_number, i]
                    # println("LETS GOOOOOOOOOOOOOOOOOOO")
                end
                # println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            end
        end
    end
    return tuple(res, 0.0f0, 0.0f0)
end

#Flatten
flatten(input::GraphNode) = BroadcastedOperator(flatten, (size(input.output,1)*size(input.output,2)*size(input.output,3)), [input::GraphNode], name="Flatten")
forward(::BroadcastedOperator{typeof(flatten)}, output, input::Array{Float32, 3}) = let
    return reshape(input, length(input))
end
backward(node::BroadcastedOperator{typeof(flatten)}, output, input::Array{Float32, 3}, g::Vector{Float32}) = let
    return tuple(reshape(g, size(input)))
end