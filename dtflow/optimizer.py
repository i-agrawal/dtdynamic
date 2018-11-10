import abc

import numpy as np

from . import Tensor, Operation


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        """
        the optimizer class defines a wrapper
        around any optimizer usually involving
        variants of gradient descent
        """

    def step(self, outputted):
        """
        given the output of a network start
        a backpropagation to update weights
        of any operation that helped create it

        :type outputted: np.ndarray
        :desc outputted: the output to backprop from
        """
        if getattr(outputted, 'previous', None):
            outputted.previous.backprop(1, self)

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
        self.eta = eta
        self.l1 = l1
        self.l2 = l2
        self.momentum = momentum
        self.nesterov = nesterov
        self.prev = {}

    def modify_grads(self, orig, grad, apply_reg):
        """
        helper function for all gradient descents
        to apply regularization, momentum, and
        nesterov accelerate gradient

        :type orig: np.ndarray
        :desc orig: original weights

        :type grad: np.ndarray
        :desc grad: the weight gradient

        :type apply_reg: bool
        :desc apply_reg: whether to regularize these
        """
        if apply_reg and self.l1:
            grad += self.l1 * np.sign(orig)
        if apply_reg and self.l2:
            grad += self.l2 * orig
        if self.momentum:
            key, _ = orig.__array_interface__['data']
            prev = self.momentum * self.prev.get(key, 0)
            if self.nesterov:
                grad -= prev
            grad = prev + self.eta * grad
            self.prev[key] = grad
        else:
            grad = self.eta * grad
        return grad


class BatchGD(GradientDescent):
    """
    the batch gradient descent class
    defines the batch/vanilla gradient
    descent optimizer
    """

    def update(self, layer, sigma):
        """
        given and layer and the current
        sigma update the weights in the
        layer

        :type layer: Layer
        :desc layer: the layer with weights

        :type sigma: np.ndarray
        :desc sigma: the current sigma
        """
        gradw, gradb = layer.gradients(sigma, np.arange(len(sigma)))
        layer.w -= self.modify_grads(layer.w, gradw, True) / len(sigma)
        layer.b -= self.modify_grads(layer.b, gradb, False) / len(sigma)


class StochasticGD(GradientDescent):
    """
    the stochastic gradient descent class
    defines the gradient descent to update
    one sample at a time
    """

    def update(self, layer, sigma):
        """
        given and layer and the current
        sigma update the weights in the
        layer

        :type layer: Layer
        :desc layer: the layer with weights

        :type sigma: np.ndarray
        :desc sigma: the current sigma
        """
        indices = np.arange(len(sigma))
        np.random.shuffle(indices)
        for ind in indices:
            gradw, gradb = layer.gradients(sigma, [ind])
            layer.w -= self.modify_grads(layer.w, gradw, True)
            layer.b -= self.modify_grads(layer.b, gradb, False)


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
        given and layer and the current
        sigma update the weights in the
        layer

        :type layer: Layer
        :desc layer: the layer with weights

        :type sigma: np.ndarray
        :desc sigma: the current sigma
        """
        n = len(sigma)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i in range(0, n, self.batch):
            inds = indices[i:min(i + self.batch, n)]
            gradw, gradb = layer.gradients(sigma, inds)
            layer.w -= self.modify_grads(layer.w, gradw, True) / len(inds)
            layer.b -= self.modify_grads(layer.b, gradb, False) / len(inds)
