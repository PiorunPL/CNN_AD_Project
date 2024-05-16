function batch_process(net::Vector{GraphNode}, data_arr::Vector{Tuple{Array{Float32, 3}, Int64}}, input_node::Variable, output_node::Variable, expected_output::Vector{Float32})
    loss = 0.0
    for data in data_arr
        input_node.output = data[1]
        fill!(expected_output, 0.0f0)
        expected_output[data[2]+1] = 1
        output_node.output = expected_output

        # println("forward")

        # @time loss += forward!(net)
        # println("backward")
        # @time backward!(net)
        
        loss += forward!(net)
        backward!(net)
    end
    return loss
end

function batch_update!(node_arr::Vector{Variable}, step::Float32, batchsize::Int64)
    for node in node_arr
        # println(node.gradient)
        node.output -= step.*(node.gradient./batchsize)
        # println("typeof node.output ", typeof(node.output), "; name: ", node.name)
    end
end

function testNetwork(testData::Vector{Tuple{Array{Float32, 3}, Int64}}, graph::Vector{GraphNode}, batchsize::Int64, image::Variable, y::Variable)
    shuffle!(testData)
    accuracy = 0
    for i in 1:batchsize
        input, expectedOutput = testData[i]
        image.output = input
        # y.output = expectedOutput
        result = forward!(graph)
        if netResult(result) == expectedOutput
            accuracy += 1
        end
    end
    return accuracy/batchsize*100
end

function netResult(x)
    maxValue, index = findmax(x)
    return index-1
end
