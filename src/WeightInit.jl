function glorot_uniform(a::Int64,b::Int64,c::Int64,d::Int64,n::Int32)
    limit_value = Float32(sqrt(6/n))
    result = -limit_value .+ rand(Float32,a,b,c,d) .* (2*limit_value)
    return result
end

function glorot_uniform(x::Int64,y::Int64,z::Int64,n::Int32)
    limit_value = Float32(sqrt(6/n))
    result = -limit_value .+ rand(Float32, x,y,z) .* (2*limit_value)
    return result
end

function glorot_uniform(x::Int64,y::Int64,n::Int32)
    limit_value = Float32(sqrt(6/n))
    result = -limit_value .+ rand(Float32, x,y) .* (2*limit_value)
    return result
end

function glorot_uniform(x::Int64,n::Int32)
    limit_value = Float32(sqrt(6/n))
    result = -limit_value .+ rand(Float32, x) .* (2*limit_value)
    return result
end
