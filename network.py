from operations import *


class Network(object):
    def __init__(self, op_list, learning_rate, batch_size, beta):
        self.op_list = []
        self.activations = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        for v in op_list:
            if v[0] == 'fc':
                print('fully connected {}'.format(v[1]))
                in_nodes = v[1][0]
                out_nodes = v[1][1]
                weights = 0.5*np.random.rand(out_nodes, in_nodes)-0.25
                biases = 0.5*np.random.rand(out_nodes, 1)-0.25
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
                weights = 0.5*np.random.rand(out_nodes, in_nodes)-0.25
                biases = 0.5*np.random.rand(out_nodes, 1)-0.25
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
                                                              batch_size=self.batch_size,
                                                              beta=self.beta)
        for ops in self.op_list[::-1][1:]:
            current_layer = current_layer - 1

            delta_l, weights_l = ops.backpropagation(a_l_minus=self.activations[current_layer],
                                                     delta_l_plus=delta_l,
                                                     weights_l_plus=weights_l,
                                                     learning_rate=self.learning_rate,
                                                     batch_size=self.batch_size,
                                                     beta=self.beta)

    def save(self, filename):
        weights_list = []
        biases_list = []
        for v in self.op_list:
            weights_list.append(v.weights)
            biases_list.append(v.biases)
        np.savez(filename + 'weights', weights_list)
        np.savez(filename + 'biases', biases_list)

    def load(self, filename):
        npz_weights = np.load(filename + 'weights.npz')
        npz_biases = np.load(filename + 'biases.npz')
        weights = npz_weights['arr_0']
        biases = npz_biases['arr_0']
        i = 0
        for v in self.op_list:
            v.weights = weights[i]
            v.biases = biases[i]
            i = i + 1

    def run(self, ff_input, labels):
        self.feedforward(ff_input=ff_input)
        self.backpropagation(labels=labels)

        results = np.equal(np.argmax(self.activations[-1], axis=1), np.argmax(labels, axis=1)).sum(0)
        accuracy = results/self.batch_size
        cost = self.op_list[-1].cost(labels=labels, output=self.activations[-1])/self.batch_size
        costL2 = 0
        for v in self.op_list:
            costL2 = costL2 + self.beta*np.square(v.weights).sum()
        self.activations = []
        weights_data = []
        biases_data = []
        for v in self.op_list:
            weights_data.append(np.array([v.weights.max(), v.weights.min(), v.weights.mean(), v.weights.std()]))
            biases_data.append(np.array([v.biases.max(), v.biases.min(), v.biases.mean(), v.biases.std()]))

        return cost, costL2, accuracy, results, weights_data, biases_data
