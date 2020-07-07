import numpy as np
#begining
start_inputs = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]])
start_outputs = np.array([0, 0, 1, 1])
mixed_start_outputs = start_outputs.reshape(4, 1)
np.random.seed(3)

def generate_weights(inputs):
    while True:
        weights = np.random.randint(-4, 4, len(inputs))
        if weights.mean() == 0:
            break
    return weights

def calc_output(input, weight, bias,):
    output = 0
    for i in range(len(weight)):
        output += sum(input[i] * weight[i])
    output += bias
    return output

def sigmoid_calc(input):
    sigmoid = 1/(1 + np.exp(-input))
    return sigmoid

def sigmoid_der_calc(input):
    sigmoid_der = sigmoid_calc(input) * (1 - sigmoid_calc(input))
    return sigmoid_der

bias = 3
learning_rate = 0.05
weights = generate_weights(start_inputs)
output = calc_output(start_inputs,weights, bias)
#print(output)
a = sigmoid_calc(output)
#print(a)
b = sigmoid_der_calc(output)
#print(b)
