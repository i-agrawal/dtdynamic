import numpy as np

from dtflow import layer, activate, cost, optimizer


if __name__ == '__main__':
    mnist = np.load('data/mnist.npz')
    X = mnist['x_train']
    y = mnist['y_train']
    y = np.eye(10)[y]

    himg, wimg = X.shape[1:]
    X = np.reshape(X, (-1, himg*wimg))

    layer1 = layer.Linear(784, 64)
    layer2 = layer.Linear(64, 10)
    optim = optimizer.BatchGD(0.1, momentum=0.1, nesterov=True)

    max_epochs = 1000
    for i in range(max_epochs):
        a = activate.sigmoid(layer1(X))
        h = activate.sigmoid(layer2(a))
        error = cost.cross_entropy(h, y)
        print(i, error)

        change = optim.step(error, report=True)
        if change < 1e-3:
            break

    import matplotlib.pyplot as plt

    while True:
        sample = X[np.random.randint(len(X))]

        a = activate.sigmoid(layer1([sample]))
        h = activate.sigmoid(layer2(a))
        pred = np.argmax(h[0])
        print(pred)

        img = sample.reshape((himg, wimg))
        plt.imshow(img, cmap='Greys', interpolation='nearest')
        plt.show()
