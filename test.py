from datetime import datetime
from network import *

import loader

start_time = datetime.now()

cost = Cost.CrossEntropy
learning_rate = .15
batch_size = 6000
hm_epoch = 100
beta = 0.0003

op_list = [('fc', (28 * 28, 70), Activation.Sigmoid),
           ('fl', (70, 10), Activation.Sigmoid, cost)]

net = Network(op_list=op_list,
              learning_rate=learning_rate,
              batch_size=batch_size,
              beta=beta)

net.load('saves/save')


def test():
    n = int(10000 / 5000)
    acc_cost = 0
    acc_results = 0
    current = 0

    for i in range(n):
        images, labels, current = loader.next_batch(batch_size=5000, current=current, dataset='testing')
        images = images.reshape(5000, -1, 1)
        cost, _, accuracy, results, _, _ = net.run(ff_input=images, labels=labels)
        acc_cost = acc_cost + cost
        acc_results = acc_results + results
    print('Test dataset cost: {}'.format(acc_cost))
    print('Test dataset accuracy: {}'.format(acc_results / 10000))
    n = int(60000 / 6000)
    current = 0
    for i in range(n):
        images, labels, current = loader.next_batch(batch_size=6000, current=current, dataset='training')
        images = images.reshape(6000, -1, 1)
        cost, _, accuracy, results, _, _ = net.run(ff_input=images, labels=labels)
        acc_cost = acc_cost + cost
        acc_results = acc_results + results
    print('Test dataset cost: {}'.format(acc_cost))
    print('Test dataset accuracy: {}'.format(acc_results /70000))


def train():
    # w1 = np.loadtxt('data/w1', skiprows=1).reshape((-1, 4))
    # w2 = np.loadtxt('data/w2', skiprows=1).reshape((-1, 4))
    # b1 = np.loadtxt('data/b1', skiprows=1).reshape((-1, 4))
    # b2 = np.loadtxt('data/b2', skiprows=1).reshape((-1, 4))
    # cost_and_acc = np.loadtxt('data/cac', skiprows=1).reshape((-1, 3))
    for epoch in range(hm_epoch):
        current = 0
        acc_cost = 0
        acc_costL2 = 0
        acc_results = 0

        n = int(60000/batch_size)

        for _ in range(n):
            images, labels, current = loader.next_batch(batch_size=batch_size, current=current)
            images = images.reshape(batch_size, -1, 1)

            cost, costL2, accuracy, results, w_data, b_data = net.run(ff_input=images, labels=labels)
            acc_costL2 = acc_costL2 + costL2
            acc_cost = acc_cost + cost
            acc_results = acc_results + results

            # BEGIN Save data
            # w1 = np.concatenate((w1, w_data[0].reshape(1, 4)))
            # w2 = np.concatenate((w2, w_data[1].reshape(1, 4)))
            #
            # b1 = np.concatenate((b1, b_data[0].reshape(1, 4)))
            # b2 = np.concatenate((b2, b_data[1].reshape(1, 4)))
            #
            # cost_and_acc = np.concatenate((cost_and_acc, np.array((cost, costL2, accuracy)).reshape(1, 3)))
            # END Save data
        print('Epoch {} of {}, cost: {}, L2 cost: {}'.format(epoch,
                                                             hm_epoch,
                                                             round(acc_cost, 3),
                                                             round(acc_costL2, 3)),
              end=' ')

        print('Accuracy: {}'.format(round(int(acc_results)/(n*batch_size), 5)))
        net.save('saves/save')
        net.save('saves/save{}'.format(epoch))
        # np.savetxt('data/w1',   w1,             fmt='%3.7f',    header='Weights 1')
        # np.savetxt('data/w2',   w2,             fmt='%3.7f',    header='Weights 2')
        # np.savetxt('data/b1',   b1,             fmt='%3.7f',    header='Biases 1')
        # np.savetxt('data/b2',   b2,             fmt='%3.7f',    header='Biases 2')
        # np.savetxt('data/cac',  cost_and_acc,   fmt='%3.7f',    header='Cost and Accuracy')

    print("--- {} seconds ---".format(datetime.now() - start_time))

test()
train()

# testFullyConnectedLayer = False
# testFinalLayer = False
# testNetworkFF = False
# print_shapes = False
#
# if testFullyConnectedLayer:
#     for i in range(38, 40):
#         for j in range(38, 40):
#             for k in range(28, 32):
#                 ff_input = np.random.rand(batch_size, i, 1)
#                 weights = np.random.rand(j, i)
#                 biases = np.random.rand(j, 1)
#                 fcl = FullyConnectedLayer(weights, biases, activation)
#                 for _ in range(5):
#                     output = fcl.feedforward(ff_input)
#
#                     a_l_minus = np.random.rand(batch_size, i, 1)
#                     weights_l_plus = np.random.rand(k, j)
#                     delta_l_plus = np.random.rand(batch_size, k, 1)
#
#                     delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate,
#                                                              batch_size)
#     print('FullyConnected: OK')
#     ff_input = np.ones((2048, 4, 1))
#     weights = np.ndarray((5, 4))
#     biases = np.ndarray((5, 1))
#
#     fcl = FullyConnectedLayer(weights, biases, activation)
#     output = fcl.feedforward(ff_input)
#
#     if print_shapes:
#         print('input:          ', ff_input.shape)
#         print('weights:        ', weights.shape)
#         print('biases:         ', biases.shape)
#         print(' - output:      ', output.shape)
#         print('')
#     a_l_minus = np.ndarray((2048, 4, 1))
#     weights_l_plus = np.ndarray((3, 5))
#     delta_l_plus = np.ndarray((2048, 3, 1))
#
#     delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate, batch_size)
#
#     if print_shapes:
#         print('a_l_minus:      ', a_l_minus.shape)
#         print('weights_l_plus: ', weights_l_plus.shape)
#         print('delta_l_plus:   ', delta_l_plus.shape)
#         print(' - delta_l:     ', delta_l.shape)
#         print(' - weights_l:   ', weights_l.shape)
#         print('')
#
#     output = fcl.feedforward(ff_input)
#     if print_shapes:
#         print('input:          ', ff_input.shape)
#         print('weights:        ', weights.shape)
#         print('biases:         ', biases.shape)
#         print(' - output:      ', output.shape)
#         print('')
#
#     delta_l, weights_l = fcl.backpropagation(a_l_minus, weights_l_plus, delta_l_plus, learning_rate, batch_size)
#
# if testFinalLayer:
#     for i in range(200, 250, 10):
#         for j in range(10, 15):
#             weights = np.random.rand(j, i)
#             biases = np.random.rand(j, 1)
#             fl = FinalLayer(weights, biases, activation, cost)
#
#             ff_input = np.random.rand(batch_size, i, 1)
#             for _ in range(5):
#                 output = fl.feedforward(ff_input)
#
#                 y = np.random.rand(batch_size, j, 1)
#                 a_l_minus = np.random.rand(batch_size, i, 1)
#
#                 delta_l, weights_l = fl.backpropagation(y, a_l_minus, learning_rate, batch_size)
#     print('FinalLayer: OK')
#     weights = np.random.rand(10, 200)
#     biases = np.random.rand(10, 1)
#     fl = FinalLayer(weights, biases, activation, cost)
#
#     ff_input = np.random.rand(batch_size, 200, 1)
#     output = fl.feedforward(ff_input)
#
#     y = np.random.rand(batch_size, 10, 1)
#     a_l_minus = np.random.rand(batch_size, 200, 1)
#
#     delta_l, weights_l = fl.backpropagation(y, a_l_minus, learning_rate, batch_size)
#     if print_shapes:
#         print('input:          ', ff_input.shape)
#         print('weights:        ', weights.shape)
#         print('biases:         ', biases.shape)
#         print(' - output:      ', output.shape)
#         print('')
#
#         print('a_l_minus:      ', a_l_minus.shape)
#         print(' - delta_l:     ', delta_l.shape)
#         print(' - weights_l:   ', weights_l.shape)
#         print('')
#
# if testNetworkFF:
#     op_list = [('fc', (28 * 28, 250), Activation.Sigmoid),
#                ('fc', (250, 500), Activation.Relu),
#                ('fl', (500, 10), Activation.Softmax, Cost.CrossEntropy)]
#     net = Network(op_list=op_list,
#                   learning_rate=learning_rate,
#                   batch_size=batch_size)
#     net.run(ff_input=np.random.rand(batch_size, 28*28, 1), labels=np.random.rand(batch_size, 10, 1))
