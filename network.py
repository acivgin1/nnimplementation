import numpy as np
from operations import *


class Network(object):
    def __init__(self, op_list, learning_rate, batch_size):
        self.op_list = []
        self.activations = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        for v in op_list:
            if v[0] == 'fc':
                print('fully connected {}'.format(v[1]))
                in_nodes = v[1][0]
                out_nodes = v[1][1]
                weights = np.random.rand(out_nodes, in_nodes) - 0.5
                biases = np.random.rand(out_nodes, 1) - 0.5
                if len(v) == 3:
                    fl = FullyConnectedLayer(weights=weights, biases=biases, activation=v[2])
                elif len(v) == 2:
                    fl = FullyConnectedLayer(weights=weights, biases=biases, activation=Activation.Relu)
                else:
                    print('ERROR op_list has a tuple of length more than 3 or less than 2')
                self.op_list.append(fl)
            if v[0] == 'fl':
                print('final layer {}'.format(v[1]))
                in_nodes = v[1][0]
                out_nodes = v[1][1]
                weights = np.random.rand(out_nodes, in_nodes)
                biases = np.random.rand(out_nodes, 1)
                if len(v) == 4:
                    fc = FinalLayer(weights=weights, biases=biases, activation=v[2], cost=v[3])
                elif len(v) == 3:
                    fc = FinalLayer(weights=weights, biases=biases, activation=v[2], cost=Cost.CrossEntropy)
                elif len(v) == 2:
                    fc = FinalLayer(weights=weights, biases=biases, activation=Activation.Softmax, cost=Cost.CrossEntropy)
                else:
                    print('ERROR op_list has a tuple of length more than 4 or less than 2')
                self.op_list.append(fc)

    def feedforward(self, ff_input):
        self.activations.append(ff_input)
        self.activations.append(self.op_list[0].feedforward(ff_input=ff_input))
        for ops in self.op_list[1:]:
            self.activations.append(ops.feedforward(ff_input=self.activations[-1]))
        return self.activations[-1]

    def backpropagation(self, labels):
        current_layer = len(self.activations) - 2
        delta_l, weights_l = self.op_list[-1].backpropagation(y=labels,
                                                              a_l_minus=self.activations[current_layer],
                                                              learning_rate=self.learning_rate,
                                                              batch_size=self.batch_size)
        for ops in self.op_list[::-1][1:]:
            current_layer = current_layer - 1

            delta_l, weights_l = ops.backpropagation(a_l_minus=self.activations[current_layer],
                                                     delta_l_plus=delta_l,
                                                     weights_l_plus=weights_l,
                                                     learning_rate=self.learning_rate,
                                                     batch_size=self.batch_size)

    def run(self, ff_input, labels):
        self.feedforward(ff_input=ff_input)
        self.backpropagation(labels=labels)

        results = np.equal(np.argmax(self.activations[-1], axis=1), np.argmax(labels, axis=1)).sum(0)
        accuracy = results/self.batch_size
        cost = self.op_list[-1].cost(labels=labels, output=self.activations[-1])
        return cost, accuracy, results
