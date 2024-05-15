function mean_squared_loss(y, ŷ, n::Int64)
    return sum( (y .- ŷ) .^ Constant(2) ) ./ Constant(n)
end

function cross_entropy(y, ŷ)
    return Constant(-1.0)*sum(y.*log.(ŷ))
end