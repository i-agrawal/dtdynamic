import abc

import numpy as np

from . import Operation, Tensor


class Layer(Operation, metaclass=abc.ABCMeta):
    def __init__(self):
        """
        the layer class defines a wrapper
        around network layers who have
        weights that need to update
        """

    @abc.abstractmethod
    def gradients(self, sigma, indices):
        """
        calculates the gradients for any
        value that needs to be updated
        and returns them
        """


class Linear(Layer):
    def __init__(self, n, k):
        """
        the commonly used 'fully connected'
        layer which is just a weight matrix
        that multiplies the input

        :type n: int
        :desc n: the number of input features

        :type k: int
        :desc k: the number of output features
        """
        self.w = np.random.randn(n, k)
        self.b = np.random.randn(1, k)

    def __call__(self, X):
        """
        multiplies the input by the current
        saved weights of this layer

        :type X: np.ndarray
        :desc X: input matrix [m x n]
        """
        self.__input = X
        self.__previous = getattr(X, 'previous', None)

        output = np.dot(X, self.w) + self.b
        return Tensor(output, self)

    def gradients(self, sigma, indices):
        """
        finds the gradient with respect
        to w, b and returns them
        """
        gradw = np.dot(self.__input[indices].T, sigma[indices])
        gradb = np.sum(sigma[indices], axis=0)
        return gradw, gradb

    def backprop(self, sigma, optim):
        """
        further backprops the sigma and
        then updates the weights using the
        optimizer

        :type sigma: np.ndarray
        :desc sigma: the backpropagated error

        :type optim: Optimizer
        :desc optim: the optimizer for updating
        """
        if self.__previous:
            next_sigma = np.dot(sigma, self.w.T)
            self.__previous.backprop(next_sigma, optim)

        optim.update(self, sigma)
