function batch_process(net, data_arr, input_node, output_node, expected_output)
    loss = 0.0
    for data in data_arr
        input_node.output = data[1]
        fill!(expected_output, 0.0)
        expected_output[data[2]+1] = 1
        output_node.output = expected_output
        loss += forward!(net)
        backward!(net)
    end
    return loss
end

function batch_update!(node_arr, step, batchsize)
    for node in node_arr
        node.output -= step.*(node.gradient./batchsize)
    end
end

function testNetwork(testData, graph, batchsize, image, y)
    shuffle!(testData)
    accuracy = 0
    for i in 1:batchsize
        input, expectedOutput = testData[i]
        image.output = input
        y.output = expectedOutput
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
