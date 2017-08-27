import numpy as np
from math import sqrt
from scipy import signal


class Activation(object):
    class Relu(object):
        @staticmethod
        def activation(preact):
            return np.where(preact > 0, preact, np.zeros(preact.shape))

        @staticmethod
        def derivative(preact):
            return np.where(preact > 0, np.ones(preact.shape), np.zeros(preact.shape))

    class Sigmoid(object):
        @staticmethod
        def activation(preact):
            d = np.exp(-preact)
            return 1/(1+d)

        @staticmethod
        def derivative(preact):
            d = np.exp(-preact)
            return d/(1 - d)**2

    class Softmax(object):
        @staticmethod
        def activation(preact):
            d = np.exp(preact)
            return d/np.transpose(np.sum(d, axis=1).reshape([1, -1]))

        @staticmethod
        def derivative(preact):
            d = np.exp(preact)
            d = d/np.transpose(np.sum(d, axis=1).reshape([1, -1]))
            return d*(1-d)

# istestirano, radi


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
        sigma_z_l = np.zeros((1, self.preactivate.shape[1], self.preactivate.shape[1]))
        for i in range(self.preactivate.shape[0]):
            if i == 0:
                sigma_z_l = np.diagflat(self.d_activation(self.preactivate[i, :, :]))
                sigma_z_l = sigma_z_l.reshape((1, self.preactivate.shape[1], -1))
            else:
                temp = np.diagflat(self.d_activation(self.preactivate[i, :, :]))
                temp = temp.reshape((1, sigma_z_l.shape[1], -1))
                sigma_z_l = np.concatenate((sigma_z_l, temp), 0)
        # sigma_z_l = np.diagflat(self.d_activation(self.preactivate[:,]))
        delta_l = np.matmul(sigma_z_l, np.matmul(np.transpose(weights_l_plus), delta_l_plus))
        delta_b = delta_l
        delta_w = np.matmul(delta_l, np.transpose(a_l_minus, (0, 2, 1)))
        weights_l = np.copy(self.weights)

        self.weights = self.weights - learning_rate * delta_w/n
        self.biases = self.biases - learning_rate * delta_b/n

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
            self.activation = Activation.Softmax.activation
            self.d_activation = Activation.Softmax.derivative

    def feedforward(self, ff_input):
        self.preactivate = np.add(np.matmul(self.weights, ff_input), self.biases)
        self.output = self.activation(self.preactivate)
        return self.output

    def backpropagation(self, y, a_l_minus, learning_rate, n):
        delta_l = np.matmul(np.diagflat(self.d_activation(self.preactivate)), self.d_cost(y, self.output))
        delta_b = delta_l
        delta_w = np.matmul(delta_l, np.transpose(a_l_minus))
        weights_l = np.copy(self.weights)

        self.weights = self.weights - learning_rate*delta_w/n
        self.biases = self.biases - learning_rate*delta_b/n

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
            out_channels = a_l_minus.shape[3]
        else:
            w = int(sqrt(a_l_minus.shape[1]))
            l = w
            out_channels = 1
        a_l_minus = np.reshape(a_l_minus, (-1, w, l, out_channels))
        delta_l = np.multiply(signal.correlate2d(delta_l_plus, weights_l_plus), self.d_activation(self.preactivate))
        delta_w = signal.correlate2d(a_l_minus, delta_l)
        delta_b = np.sum(delta_l, axis = (1, 2))

        window_l = np.copy(self.window)

        self.window = self.window - learning_rate*delta_w/n
        self.biases = self.biases - learning_rate*delta_b/n
        return delta_l, window_l


class MaxPoolLayer(object):
    def __init__(self, stride, ksize):
        self.stride = stride
        self.ksize = ksize

    def feedforward(self, ff_input):
        print('feedforward method in class MaxPoolLayer not implemented.')

    def backpropagation(self, ff_input):
        print('backpropagation method in class MaxPoolLayer not implemented.')

if __name__ == '__main__':
    print('Operations intended only for importing into other files')
else:
    print('Successfuly imported operations.py')