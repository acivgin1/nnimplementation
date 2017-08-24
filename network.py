import numpy as np


class Network(object):
    def __init__(self, op_list):
        self.num_ops = len(op_list)
        self.op_list = op_list

