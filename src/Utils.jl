function convert_to_float32(x,y)
    if isa(x, Array{Float64, 3})
        x = convert(Array{Float32, 3}, x)
    end
    if isa(y, Array{Float64, 3})
        y = convert(Array{Float32, 3}, y)
    end
    # if isa(x, Vector{Float64})
    #     x = convert(Vector{Float32}, x)
    # end
    # if isa(y, Vector{Float64})
    #     y = convert(Vector{Float32}, y)
    # end
    return x, y
end