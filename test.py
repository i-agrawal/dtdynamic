import numpy as np
from sklearn import datasets

from dtflow import layer, activate, cost, optimizer


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y = np.eye(3)[y]

    optim = optimizer.BatchGD(eta=0.1, momentum=0.05, nesterov=True)
    linear = layer.Linear(4, 3)

    for _ in range(1000):
        h = activate.sigmoid(linear(X))
        error = cost.cross_entropy(h, y)
        optim.step(error)

    y = iris.target
    h = activate.sigmoid(linear(X))
    guess = np.argmax(h, axis=1)
    print(np.mean(guess == y))
