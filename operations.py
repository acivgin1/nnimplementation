import numpy as np
from math import sqrt
from scipy import signal

np.seterr(divide='ignore', invalid='ignore', over='ignore')

print('\033[91m' + 'WARNING: Probably could sum a_l_minus, delta_l along axis = 0, for speed up and memory saving' + '\033[0m')


class Cost(object):
    class LMS(object):
        @staticmethod
        def cost(labels, output):
            return 0.5 * np.square(np.subtract(labels, output)).sum()

        @staticmethod
        def derivative(preact, labels, output, d_activation):
            sigma_z_l = d_activation(preact)
            sigma_z_l = np.matmul(sigma_z_l, np.ones((1, sigma_z_l.shape[1])))
            return np.matmul(sigma_z_l, np.subtract(output, labels).sum(0))

    class CrossEntropy(object):
        @staticmethod
        def cost(labels, output):
            return -np.add(np.multiply(labels, np.log(output)), np.multiply((1-labels), np.log(1-output))).sum()

        @staticmethod
        def derivative(preact, labels, output, d_activation):
            return np.subtract(labels, output).sum(0)


class Activation(object):
    class Relu(object):
        @staticmethod
        def activation(preact):
            return np.where(preact > 0.0, preact, np.zeros(preact.shape))

        @staticmethod
        def derivative(preact):
            return np.where(preact > 0.0, np.ones(preact.shape), np.zeros(preact.shape))

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
            return d/np.transpose(np.sum(d, axis=1).reshape([1, -1, 1]), (1, 0, 2))

        @staticmethod
        def derivative(preact):
            d = np.exp(preact)
            d = d/np.transpose(np.sum(d, axis=1).reshape([1, -1, 1]), (1, 0, 2))
            return d*(1-d)


class FullyConnectedLayer(object):  # istestirano, radi
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

    def backpropagation(self, a_l_minus, delta_l_plus, weights_l_plus, learning_rate, batch_size):
        sigma_z_l = self.d_activation(self.preactivate)
        sigma_z_l = np.matmul(sigma_z_l, np.ones((1, sigma_z_l.shape[1])))

        delta_l = np.matmul(sigma_z_l, np.matmul(np.transpose(weights_l_plus), delta_l_plus))
        delta_b = delta_l.sum(0)

        delta_w = np.matmul(delta_l, np.transpose(a_l_minus, (0, 2, 1))).sum(0)

        weights_l = np.copy(self.weights)
        self.weights = self.weights - learning_rate * delta_w/batch_size
        self.biases = self.biases - learning_rate * delta_b/batch_size

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
        if cost == Cost.CrossEntropy:
            self.activation = Activation.Softmax.activation
            self.d_activation = Activation.Softmax.derivative

    def feedforward(self, ff_input):
        self.preactivate = np.add(np.matmul(self.weights, ff_input), self.biases)
        self.output = self.activation(self.preactivate)
        return self.output

    def backpropagation(self, y, a_l_minus, learning_rate, batch_size):
        delta_l = self.d_cost(self.preactivate, y, self.output, self.d_activation)
        delta_b = delta_l.sum(0)
        delta_w = np.matmul(delta_l, np.transpose(a_l_minus, (0, 2, 1))).sum(0)
        weights_l = np.copy(self.weights)

        self.weights = self.weights - learning_rate*delta_w/batch_size
        self.biases = self.biases - learning_rate*delta_b/batch_size
        return delta_l, weights_l

    def cost(self, labels, ff_output):
        return self.cost(labels=labels, output=ff_output)

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
        delta_b = np.sum(delta_l, axis=(1, 2))

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
    print('\033[91m' + 'WARNING: operations.py intended only for use in other files' + '\033[0m')
else:
    print('\033[94m' + 'OK: Successfuly imported operations.py' + '\033[0m')