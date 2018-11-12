import abc

import numpy as np


class Tensor(np.ndarray):
    def __new__(self, data, previous):
        """
        the tensor class is an extremely
        thin wrapper around the np.array
        class that just stores the operation
        that created this tensor

        :type previous: Operation
        :desc previous: the operation that created it
        """
        array = np.asarray(data).view(self)
        array.previous = previous
        return array


class Operation(metaclass=abc.ABCMeta):
    """
    the operation class defines a wrapper
    around commonly used operations in
    machine learning (i.e. addition,
    convolution, cost functions, etc.)
    """

    @abc.abstractmethod
    def __call__(self):
        """
        the operation takes in input here,
        applies the operation, saves any info
        it needs for backprop, and outputs
        the results of the operation
        """

    @abc.abstractmethod
    def backprop(self, sigma, optim, indices=None):
        """
        given the back propagated change,
        calculate the gradient and update
        the internal weights and then
        feed the next sigma to the next op

        :type sigma: np.ndarray
        :desc sigma: the backpropagated error

        :type optim: Optimizer
        :desc optim: the optimizer for updating

        :type indices: np.ndarray
        :desc indices: the samples being used
        """
