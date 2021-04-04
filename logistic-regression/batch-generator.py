# python code that generates batches of a given size
import numpy as np


def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- feature matrix
    param y: np.array[n_objects] --- vector of target values
    """

    assert len(X) == len(y)

    np.random.seed(42)
    
    X = np.array(X)
    y = np.array(y)

    perm = np.random.permutation(len(X))

    length = len(X)
    while length // batch_size != 0:
        yield (X[perm[:batch_size]], y[perm[:batch_size]])
        length -= batch_size
        perm = perm[:batch_size]

