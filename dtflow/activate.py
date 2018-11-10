import numpy as np

from . import Operation, Tensor


class Sigmoid(Operation):
    def __init__(self):
        """
        the sigmoid activation function
        """

    def __call__(self, X):
        """
        applies the sigmoid function to the input
        the function is 1 / (1 + e^(-x))

        :type X: np.ndarray
        :desc X: input matrix [m x n]
        """
        self.__output = 1 / (1 + np.exp(-X))
        self.__previous = getattr(X, 'previous', None)

        return Tensor(self.__output, self)

    def backprop(self, sigma, optim):
        """
        calculates the derivative of
        sigmoid which is simply s*(1-s)
        and passes on the further backprop

        :type sigma: np.ndarray
        :desc sigma: the backpropagated error

        :type optim: Optimizer
        :desc optim: the optimizer for updating
        """
        if self.__previous:
            change = self.__output * (1 - self.__output)
            next_sigma = change * sigma
            self.__previous.backprop(next_sigma, optim)


def sigmoid(X):
    return Sigmoid()(X)
