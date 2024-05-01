function mean_squared_loss(y, ŷ, n)
    return sum( (y .- ŷ) .^ Constant(2) ) ./ Constant(n)
end
