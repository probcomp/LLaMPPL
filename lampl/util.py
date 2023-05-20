import numpy as np

def logsumexp(arr):
    # Numerically stable implementation
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))

def softmax(arr):
    # Numerically stable implementation
    return np.exp(arr - logsumexp(arr))