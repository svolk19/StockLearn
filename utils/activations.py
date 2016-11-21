import numpy as np


def sigmoid(X, deriv=False):
    # activation function at each neuron: tanh
    # assumes X is a numpy array
    if deriv:
        return 1.0 - X ** 2
    else:
        for i, column in enumerate(X):
            for j, elem in enumerate(column):
                X[i][j] = np.tanh(elem)
        return X


def softmax(X, predict=False):
    # activation for last set of neurons: the probabilistic normalizer
    # assumes X is a numpy array
    exp = np.exp(X)
    exp_sum = np.sum(exp, axis=0)
    results = exp / exp_sum

    if predict:
        return np.argmax(results, axis=1)
    else:
        return results


