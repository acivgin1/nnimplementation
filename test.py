from datetime import datetime
from operations import *
from network import *
start_time = datetime.now()


# Activation:               OK
# Cost Functions:           OK
# FullyConnectedLayer:      OK
# ConvolutionalLayer:       NT
# MaxPoolLayer:             NT
# FinalLayer:               OK
# Network:                  OK

# testing FullyConnectedLayer class

activation = Activation.Relu
cost = Cost.CrossEntropy
learning_rate = 0.01
batch_size = 501


testFullyConnectedLayer = False
testFinalLayer = False
testNetworkFF = False
print_shapes = False

if testFullyConnectedLayer:
    for i in range(38, 40):
        for j in range(38, 40):
            for k in range(28, 32):
                ff_input = np.random.rand(batch_size, i, 1)
                weights = np.random.rand(j, i)
                biases = np.random.rand(j, 1)
                fcl = FullyConnectedLayer(weights, biases, activation)
                for _ in range(5):
                    output = fcl.feedforward(ff_input)

                    a_l_minus = np.random.rand(batch_size, i, 1)
                    weights_l_plus = np.random.rand(k, j)
                    delta_l_plus = np.random.rand(batch_size, k, 1)

                    delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate,
                                                             batch_size)
    print('FullyConnected: OK')
    ff_input = np.ones((2048, 4, 1))
    weights = np.ndarray((5, 4))
    biases = np.ndarray((5, 1))

    fcl = FullyConnectedLayer(weights, biases, activation)
    output = fcl.feedforward(ff_input)

    if print_shapes:
        print('input:          ', ff_input.shape)
        print('weights:        ', weights.shape)
        print('biases:         ', biases.shape)
        print(' - output:      ', output.shape)
        print('')
    a_l_minus = np.ndarray((2048, 4, 1))
    weights_l_plus = np.ndarray((3, 5))
    delta_l_plus = np.ndarray((2048, 3, 1))

    delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate, batch_size)

    if print_shapes:
        print('a_l_minus:      ', a_l_minus.shape)
        print('weights_l_plus: ', weights_l_plus.shape)
        print('delta_l_plus:   ', delta_l_plus.shape)
        print(' - delta_l:     ', delta_l.shape)
        print(' - weights_l:   ', weights_l.shape)
        print('')

    output = fcl.feedforward(ff_input)
    if print_shapes:
        print('input:          ', ff_input.shape)
        print('weights:        ', weights.shape)
        print('biases:         ', biases.shape)
        print(' - output:      ', output.shape)
        print('')

    delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate, batch_size)

if testFinalLayer:
    for i in range(200, 250, 10):
        for j in range(10, 15):
            weights = np.random.rand(j, i)
            biases = np.random.rand(j, 1)
            fl = FinalLayer(weights, biases, activation, cost)

            ff_input = np.random.rand(batch_size, i, 1)
            for _ in range(5):
                output = fl.feedforward(ff_input)

                y = np.random.rand(batch_size, j, 1)
                a_l_minus = np.random.rand(batch_size, i, 1)

                delta_l, weights_l = fl.backpropagation(y, a_l_minus, learning_rate, batch_size)
    print('FinalLayer: OK')
    weights = np.random.rand(10, 200)
    biases = np.random.rand(10, 1)
    fl = FinalLayer(weights, biases, activation, cost)

    ff_input = np.random.rand(batch_size, 200, 1)
    output = fl.feedforward(ff_input)

    y = np.random.rand(batch_size, 10, 1)
    a_l_minus = np.random.rand(batch_size, 200, 1)

    delta_l, weights_l = fl.backpropagation(y, a_l_minus, learning_rate, batch_size)
    if print_shapes:
        print('input:          ', ff_input.shape)
        print('weights:        ', weights.shape)
        print('biases:         ', biases.shape)
        print(' - output:      ', output.shape)
        print('')

        print('a_l_minus:      ', a_l_minus.shape)
        print(' - delta_l:     ', delta_l.shape)
        print(' - weights_l:   ', weights_l.shape)
        print('')

if testNetworkFF:
    op_list = [('fc', (28 * 28, 250), Activation.Sigmoid),
               ('fc', (250, 500), Activation.Relu),
               ('fl', (500, 10), Activation.Softmax, Cost.CrossEntropy)]
    net = Network(op_list=op_list,
                  learning_rate=learning_rate,
                  batch_size=batch_size)
    net.run(ff_input=np.random.rand(batch_size, 28*28, 1), labels=np.random.rand(batch_size, 10, 1))

import loader
training_data, validation_data, test_data = loader.load_data_wrapper()

print(training_data.shape)

print("--- {} seconds ---".format(datetime.now() - start_time))
