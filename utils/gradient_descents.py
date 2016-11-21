import numpy as np


def mini_batch_generate(X, y, batch_size, randomize=True):
    # yields mini batches of training data to perform stochastic gradient descent

    index_array = np.arange(len(X))

    if randomize:
        np.random.shuffle(index_array)

    indices = []
    for start_index in range(0, len(index_array) - batch_size + 1, batch_size):
        indices.append(index_array[start_index:start_index + batch_size])

    indices = np.array(indices)

    X_batches = np.empty((len(indices), batch_size, len(X[0])))
    y_batches = np.empty((len(indices), batch_size, len(y[0])))

    for i, indexes in enumerate(indices):
        for j, index in enumerate(indexes):
            X_batches[i][j] = X[index]
            y_batches[i][j] = y[index]

    return X_batches, y_batches


