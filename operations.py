import numpy as np
from math import sqrt
from scipy import signal

class FullyConnectedLayer(object):
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.preactivate = np.zeros(np.shape(biases))
        self.activation = activation.activation
        self.d_activation = activation.derivative

    def feedforward(self, ff_input):
        self.preactivate = np.add(np.matmul(self.weights, ff_input), self.biases)
        output = self.activation(self.preactivate)
        return output  # for use in next layers and backpropagation

    def backpropagation(self, a_l_minus, weights_l_plus, delta_l_plus, learning_rate, n):
        sigma_z_l = np.diagflat(self.d_activation(self.preactivate))
        delta_l = np.matmul(sigma_z_l, np.matmul(np.transpose(weights_l_plus), delta_l_plus))
        delta_b = delta_l
        delta_w = np.matmul(delta_l, np.transpose(a_l_minus))
        weights_l = np.copy(self.weights)

        self.weights = self.weights - learning_rate/n * delta_w
        self.biases = self.biases - learning_rate/n * delta_b

        return delta_l, weights_l  # for use in previous layers


class FinalLayer(object):
    def __init__(self, weights, biases, activation, cost):
        self.weights = weights
        self.biases = biases
        self.output = np.zeros(np.shape(biases))
        self.preactivate = np.zeros(np.shape(biases))
        self.activation = activation.activation
        self.d_activation = activation.derivative
        self.cost = cost.cost
        self.d_cost = cost.derivative
        if cost.name == 'cross_entropy':
            self.activation = activation_function.softmax.activation
            self.d_activation = activation_function.softmax.derivative

    def feedforward(self, ff_input):
        self.preactivate = np.add(np.matmul(self.weights, ff_input), self.biases)
        self.output = self.activation(self.preactivate)
        return self.output

    def backpropagation(self, y):
        delta_l = np.matmul(np.diagflat(self.d_activation(self.preactivate)), self.d_cost(y, self.output))
        weights_l = np.copy(self.weights)
        return delta_l, weights_l


class ConvolutionalLayer(object):
    def __init__(self, window, biases, activation):
        self.window = window
        self.biases = biases
        self.activation = activation.activation
        self.d_activation = activation.derivative
        self.preactivate = np.array()

    def feedforward(self, ff_input):
        self.preactivate = signal.correlate2d(ff_input, self.window)
        output = self.activation(self.preactivate)
        return output

    def backpropagation(self, a_l_minus, weights_l_plus, delta_l_plus, learning_rate, n):

        if a_l_minus.ndim > 2:
            w = a_l_minus.shape[1]
            l = a_l_minus.shape[2]
        else:
            w = int(sqrt(a_l_minus.shape[1]))
            l = w
        a_l_minus = np.reshape(a_l_minus, (-1, w, l))
        delta_w = signal.correlate2d(a_l_minus, )
    def backpropagation(self, ):

if __name__ == '__main__':
    print('Operations')