# actually Leaky relu
from numpy import exp

import numpy as np

def ReLU(x):
    return max(0.1 * x, x)


def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(self, x):
    # compute the derivative of the sigmoid function ASSUMING
    # that x has already been passed through the 'sigmoid'
    # function
    return x * (1 - x)


def softmax(x):
    return exp(x) / exp(x).sum()
