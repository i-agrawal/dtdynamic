import numpy as np

from . import Operation, Tensor


class Sigmoid(Operation):
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

        operation = self if self.__previous else None
        return Tensor(self.__output, operation)

    def backprop(self, sigma, optim):
        """
        the sigmoid function is:
            1 / (1 + e^(-x))

        after some math, the derivative
        with respect to x simply is:
            sigmoid(x) * (1 - sigmoid(x))

        and we multiply this by the backpropagated error
        """
        if self.__previous:
            change = self.__output * (1 - self.__output)
            next_sigma = change * sigma
            self.__previous.backprop(next_sigma, optim)


def sigmoid(X):
    """
    a helper function for creating a
    sigmoid so that it looks nicer when
    we call it later
    """
    return Sigmoid()(X)


class ReLU(Operation):
    def __init__(self, leak=0):
        """
        the ReLU activation function

        :type leak: float
        :desc leak: amount to let small in
        """
        self.leak = 0

    def __call__(self, X):
        """
        applies the relu function to the input
        the function is:
            |  x > 0     x
            |  x <= 0    leak * x

        :type X: np.ndarray
        :desc X: input matrix [m x n]
        """
        self.__mask = np.ones(X.shape)
        self.__mask[X < 0] = self.leak
        self.__previous = getattr(X, 'previous', None)

        operation = self if self.__previous else None
        return Tensor(X * self.__mask, operation)

    def backprop(self, sigma, optim):
        """
        the relu function is:
            |  x > 0     x
            |  x <= 0    leak * x

        the derivative with respect to x simply is:
            |  x > 0     1
            |  x <= 0    leak

        and we multiply this by the backpropagated error
        """
        if self.__previous:
            next_sigma = self.__mask * sigma
            self.__previous.backprop(next_sigma, optim)


def relu(X, leak=0):
    """
    a helper function for creating a
    ReLU so that it looks nicer when
    we call it later
    """
    return ReLU(leak)(X)
