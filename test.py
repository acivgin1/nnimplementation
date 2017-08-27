from operations import *

# Activation
# FullyConnectedLayer
# ConvolutionalLayer
# MaxPoolLayer
# FinalLayer


print_shape = True
# testing FullyConnectedLayer class
input = np.ones((5, 4, 1))
weights = np.ndarray((5, 4))
biases = np.ndarray((5, 1))
activation = Activation.Relu

fcl = FullyConnectedLayer(weights, biases, activation)
output = fcl.feedforward(input)


a_l_minus = np.ndarray((5, 4, 1))
weights_l_plus = np.ndarray((3, 5))
delta_l_plus = np.ndarray((5, 3, 1))
learning_rate = 0.01
n = 10
delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate, n)

if print_shape:
    print(a_l_minus.shape)
    print(weights_l_plus.shape)
    print(delta_l_plus.shape)
    print(delta_l.shape)
    print(weights_l.shape)

