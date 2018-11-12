import abc

import numpy as np

from . import Operation, Tensor


class Layer(Operation, metaclass=abc.ABCMeta):
    def __init__(self):
        """
        the layer class defines a wrapper
        around network layers who have
        weights that need to update

        :type prevw: np.ndarray
        :desc prevw: the previous gradient of w
                     held for the optimizer

        :type prevb: np.ndarray
        :desc prevb: the previous gradient of b
                     held for the optimizer
        """
        self.prevw = 0
        self.prevb = 0

    @abc.abstractmethod
    def gradients(self, sigma, indices):
        """
        calculates the gradients for
        the weights and the bias and
        then returns them

        :type sigma: np.ndarray
        :desc sigma: the backpropagated error

        :type indices: List[int]
        :desc indices: the samples to use
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
        super().__init__()
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
        the fully connected layer is simply:
            x * w + b
        so the derivative with respect to w is:
            x
        the derivative with respect to b is:
            1
        and we multiply these by the backpropagated error
        """
        input_ = self.__input[indices]
        error = sigma[indices]

        gradw = np.dot(input_.T, error)
        gradb = np.sum(error, axis=0)
        return gradw, gradb

    def backprop(self, sigma, optim):
        """
        the fully connected layer is simply:
            x * w + b
        so the derivative with respect to x is:
            w
        and we multiply this by the backpropagated error
        """
        if self.__previous:
            next_sigma = np.dot(sigma, self.w.T)
            self.__previous.backprop(next_sigma, optim)

        optim.update(self, sigma)
