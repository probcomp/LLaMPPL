from .distribution import Distribution
import numpy as np
from ..util import softmax

class TokenCategorical(Distribution):
    vocab = None

    @classmethod
    def set_vocab(cls, vocab):
        cls.vocab = vocab

    def __init__(self, logits):
        # Error if class variable vocab is not set
        if TokenCategorical.vocab is None:
            raise Exception("TokenCategorical.vocab is not set")
        
        self.probs = softmax(logits)

    def sample(self):
        n = np.random.choice(len(self.probs), p=(self.probs))
        return TokenCategorical.vocab[n], np.log(self.probs[n])

    def log_prob(self, value):
        return np.log(self.probs[value.token_id])
    
    def argmax(self, idx):
        tok = np.argsort(self.probs)[-idx]
        return TokenCategorical.vocab[tok], np.log(self.probs[tok])