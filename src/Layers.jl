# Dense Layer
function dense(w, x, b, activation)
    return activation(w * x .+ b) 
end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

# Bias
function bias(x, w)
    return x .+ w
end