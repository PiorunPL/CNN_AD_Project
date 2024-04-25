function mean_squared_loss(y, ŷ)
    return Constant(length(y.output)) .* (y .- ŷ) .^ Constant(2)
end
