function glorot_uniform(a::Int64,b::Int64,c::Int64,d::Int64,n::Int32)
    # limit_value = sqrt(6/n)
    limit_value = Float32(sqrt(6/n))
    # println("typeof limit_value: ", typeof(limit_value))
    result = -limit_value .+ rand(Float32,a,b,c,d) .* (2*limit_value)
    # println("typeof result: ", typeof(result))
    # result = reshape(result, a, b, c, d)
    return result
    # println("typeof result: ", typeof(result))
    # return convert(Array{Float32, 4}, result)
end

function glorot_uniform(x::Int64,y::Int64,z::Int64,n::Int32)
    # limit_value = Float32(sqrt(6/n))
    limit_value = Float32(sqrt(6/n))
    result = -limit_value .+ rand(Float32, x,y,z) .* (2*limit_value)
    # result = reshape(result,x,y,z)
    # println("typeof result: ", typeof(result))
    # return convert(Array{Float32, 3}, result)
    return result
end

function glorot_uniform(x::Int64,y::Int64,n::Int32)
    # limit_value = Float32(sqrt(6/n))
    limit_value = Float32(sqrt(6/n))
    # println("typeof limit_value: ", typeof(limit_value))
    result = -limit_value .+ rand(Float32, x,y) .* (2*limit_value)
    # println("typeof reshape(result,x,y): ", typeof(reshape(result,x,y)))
    # result = reshape(result,x,y)
    # println("typeof result: ", typeof(result))
    # return convert(Array{Float32, 2}, result)
    return result
end

function glorot_uniform(x::Int64,n::Int32)
    # limit_value = Float32(sqrt(6/n))
    limit_value = Float32(sqrt(6/n))
    result = -limit_value .+ rand(Float32, x) .* (2*limit_value)
    # println("typeof result: ", typeof(result))
    # return reshape(result,x,1,1)
    # return convert(Array{Float32, 1}, result)
    return result
end
