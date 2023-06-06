from .distribution import Distribution
from ..util import softmax
from ..token import Token
import numpy as np

class Transformer(Distribution):

    # TODO: support custom temperatures
    def __init__(self, ctx):
        self.ctx = ctx

    def log_prob(self, x):
        # Check if x is a token or an int
        if isinstance(x, Token):
            return self.ctx.observe_token(x.token_id)
        else:
            return self.ctx.observe_token(x)
        
    def sample(self):
        probs = softmax(self.ctx.logits())
        token_id = np.random.choice(len(probs), p=(probs))
        logprob = self.ctx.observe_token(token_id)
        return self.ctx.vocab[token_id], logprob
    
    def argmax(self, idx):
        token_id = np.argsort(self.ctx.logits())[-idx]
        logprob = self.ctx.observe_token(token_id)
        return self.ctx.vocab[token_id], logprob