from .distribution import Distribution
import numpy as np
from ..util import softmax

class LogCategorical(Distribution):
    def __init__(self, logits):
        self.probs = softmax(logits)

    def sample(self):
        n = np.random.choice(len(self.probs), p=(self.probs))
        return n, self.log_prob(n)

    def log_prob(self, value):
        return np.log(self.probs[value])
    
    def argmax(self, idx):
        return np.argsort(self.probs)[-idx]