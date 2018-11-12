import abc

import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        """
        the optimizer class defines a wrapper
        around any optimizer usually involving
        variants of gradient descent
        """
        self.report = False

    def step(self, outputted, report=False):
        """
        given the output of a network start
        a backpropagation to update weights
        of any operation that helped create it

        :type outputted: np.ndarray
        :desc outputted: the output to backprop from

        :type report: bool
        :desc report: whether to output weight change
        """
        self.report = report
        self.change = 0
        if getattr(outputted, 'previous', None):
            outputted.previous.backprop(1, self)
        return self.change

    @abc.abstractmethod
    def update(self, operation, sigma):
        """
        given and operation and the current
        sigma update the weights in the
        operation

        :type operation: np.ndarray
        :desc weights: the weights to be update

        :type sigma: np.ndarray
        :desc sigma: the current sigma
        """


class GradientDescent(Optimizer):
    def __init__(self, eta, l1=0, l2=0, momentum=0, nesterov=False):
        """
        the gradient descent abstract class
        defines a helpful function for gradient
        descent optimizers

        :type eta: float
        :desc eta: the learning rate

        :type l1: float
        :desc l1: the l1 regularization rate

        :type l2: float
        :desc l2: the l2 regularization rate

        :type momentum: float
        :desc momentum: previous weight update worth

        :type nesterov: float
        :desc nesterov: previous weight update worth
        """
        super().__init__()
        self.eta = eta
        self.l1 = l1
        self.l2 = l2
        self.momentum = momentum
        self.nesterov = nesterov

    def modify_grads(self, layer, gradw, gradb, samples):
        """
        helper function for all gradient descents
        to apply regularization, momentum, and
        nesterov accelerate gradient

        :type layer: Layer
        :desc layer: the layer we are applying this to

        :type gradw: np.ndarray
        :desc gradw: the weight gradient

        :type gradb: np.ndarray
        :desc gradb: the bias gradient

        :type samples: int
        :desc samples: the number of samples
                       used for the gradient
        """
        if self.l1:
            gradw += self.l1 * np.sign(layer.w)
        if self.l2:
            gradw += self.l2 * layer.w

        eta = self.eta / samples
        if self.momentum:
            prevw = self.momentum * layer.prevw
            prevb = self.momentum * layer.prevb
            if self.nesterov:
                gradw -= prevw
                gradb -= prevb
            gradw = prevw + eta * gradw
            gradb = prevb + eta * gradb
            layer.prevw = gradw
            layer.prevb = gradb
        else:
            gradw = eta * gradw
            gradb = eta * gradb

        if self.report:
            self.change += np.sum(gradw**2)
            self.change += np.sum(gradb**2)
        return gradw, gradb


class BatchGD(GradientDescent):
    """
    the batch gradient descent class
    defines the batch/vanilla gradient
    descent optimizer
    """

    def update(self, layer, sigma):
        """
        find the gradients over the entire
        batch and then average it over the
        the number of examples
        """
        gradw, gradb = layer.gradients(sigma, np.arange(len(sigma)))
        gradw, gradb = self.modify_grads(layer, gradw, gradb, len(sigma))

        layer.w -= gradw
        layer.b -= gradb


class StochasticGD(GradientDescent):
    """
    the stochastic gradient descent class
    defines the gradient descent to update
    one sample at a time
    """

    def update(self, layer, sigma):
        """
        find the gradients over an example
        in the batch and then apply it and
        repeat for all examples in the batch
        """
        indices = np.arange(len(sigma))
        np.random.shuffle(indices)
        for ind in indices:
            gradw, gradb = layer.gradients(sigma, [ind])
            gradw, gradb = self.modify_grads(layer, gradw, gradb, 1)

            layer.w -= gradw
            layer.b -= gradb


class MiniBatchGD(GradientDescent):
    def __init__(self, batch, *args, **kwargs):
        """
        the mini batch gradient descent class
        defines the gradient descent to update
        batch size sample at a time

        :type batch: int
        :desc batch: mini batch size
        """
        super().__init__(*args, **kwargs)
        self.batch = batch

    def update(self, layer, sigma):
        """
        find the gradients over a small subset
        of examples in the batch and then apply
        it and repeat for all subsets in the batch
        """
        n = len(sigma)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i in range(0, n, self.batch):
            inds = indices[i:min(i + self.batch, n)]
            gradw, gradb = layer.gradients(sigma, inds)
            gradw, gradb = self.modify_grads(layer, gradw, gradb, len(inds))

            layer.w -= gradw
            layer.b -= gradb
