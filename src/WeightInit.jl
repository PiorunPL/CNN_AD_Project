function glorot_uniform(a,b,c,d,n)
    limit_value = sqrt(6/n)
    result = -limit_value .+ rand(Float64,a,b,c,d) .* (2*limit_value)
    return result
end
function glorot_uniform(x,y,z,n)
    limit_value = sqrt(6/n)
    result = -limit_value .+ rand(Float64, x,y,z) .* (2*limit_value)
    return result
end
function glorot_uniform(x,y,n)
    limit_value = sqrt(6/n)
    result = -limit_value .+ rand(Float64, x,y) .* (2*limit_value)
    return result
end
function glorot_uniform(x,n)
    limit_value = sqrt(6/n)
    result = -limit_value .+ rand(Float64, x) .* (2*limit_value)
    return result
end
