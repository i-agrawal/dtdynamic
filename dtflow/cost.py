import numpy as np

from . import Operation, Tensor


class CrossEntropy(Operation):
    def __init__(self):
        """
        the cross entropy cost function
        """

    def __call__(self, h, y, eps):
        """
        given the hypothesis and the correct
        outputs, calculate the log loss between
        h and y assuming one-hot encoded

        :type h: np.ndarray
        :desc h: predicted classes [m x k]

        :type y: np.ndarray
        :desc y: actual classes [m x k]
        """
        self.__previous = getattr(h, 'previous', None)
        h = np.clip(h, eps, 1 - eps)
        self.__input = y, h

        cost = y * np.log(h) + (1 - y) * np.log(1 - h)
        operation = self if self.__previous else None
        return Tensor(-np.mean(cost), operation)

    def backprop(self, sigma, optim):
        """
        calculates the derivative of
        cross entropy which is:
        -y / h + (1 - y) / (1 - h)

        :type sigma: np.ndarray
        :desc sigma: the backpropagated error

        :type optim: Optimizer
        :desc optim: the optimizer for updating
        """
        if self.__previous:
            y, h = self.__input
            change = (1 - y) / (1 - h) - y / h
            next_sigma = change * sigma
            self.__previous.backprop(next_sigma, optim)


def cross_entropy(h, y, eps=1e-9):
    return CrossEntropy()(h, y, eps)
