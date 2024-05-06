function mean_squared_loss(y, ŷ, n)
    return sum( (y .- ŷ) .^ Constant(2) ) ./ Constant(n)
end

function cross_entropy(y, ŷ)
    return Constant(-1)*sum(y.*log.(ŷ))
end
