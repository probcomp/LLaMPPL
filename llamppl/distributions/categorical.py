from .distribution import Distribution
import numpy as np

class Categorical(Distribution):
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        x = np.random.choice(len(self.probs), p=self.probs)
        return x, self.log_prob(x)

    def argmax(self, idx):
        return np.argsort(self.probs)[-idx]

    def log_prob(self, value):
        return np.log(self.probs[value])