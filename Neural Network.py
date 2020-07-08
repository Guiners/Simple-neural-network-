import numpy as np
#data
start_inputs = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]])
#print(start_inputs)
start_outputs = np.array([0, 0, 1, 1])
#print(start_outputs)
start_outputs = start_outputs.reshape(4, 1)
#print(mixed_start_outputs)
#np.random.seed()

def generate_weights(inputs):
    while True:
        weights = np.random.randint(-10.0, 10.0, (len(inputs) - 1))
        weights = weights * 0.11111111111111 #wait
        if weights.mean() == 0:
            #print(weights)
            break
    weights = weights.reshape(3, 1)
    return weights

def output_calc(input, weight, bias,):
    output = np.dot(input, weight) + bias
    return output

def sigmoid_calc(input):
    sigmoid = 1/(1 + np.exp(-input))
    return sigmoid

def sigmoid_der_calc(input):
    sigmoid_der = sigmoid_calc(input) * (1 - sigmoid_calc(input))
    return sigmoid_der

def test(point):
    result = sigmoid_calc(output_calc(point, weights, bias))
    if result > 0.5:
        output = 1
    else:
        output = 0
    print("network thinks that is", output, "with", result, "probability")
    return result, output
#parameters
bias = 3
learning_rate = 0.05
weights = generate_weights(start_inputs)
#print(weights)

for i in range(50001):
    inputs = start_inputs

    #feedforward
    xw = output_calc(inputs, weights, bias) #change name
    z = sigmoid_calc(xw)    #change name
    #print(z)

    #backpropagation
    error = z - start_outputs
    error_sum = error.sum()
    #print(error_sum)
    dcost_dpred = error #change name
    dpred_dz = sigmoid_der_calc(z) #change name

    #slope = inputs * dcost_dpred * dpred_dz
    z_delta = dcost_dpred * dpred_dz #change name

    inputs = start_inputs.transpose()
    #print('a', np.dot(inputs, z_delta))
    #print('b', np.dot(inputs, z_delta)* learning_rate)
    #print('c', weights)
    #print(type(np.dot(inputs, z_delta)))
    weights = weights - (learning_rate * np.dot(inputs, z_delta)) 
    #print(weights)
    for x in z_delta:
        bias -= learning_rate * x

    if i%10000 == 0:
        print("value of error", error_sum)

#test
point = np.random.randint(0, 2, 3)
print(point)
a, b = test(point)
