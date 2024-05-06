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
