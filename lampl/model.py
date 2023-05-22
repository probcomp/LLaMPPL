import numpy as np
from .util import softmax
from .llama import LlamaContext, CachedLlama

class Model:
    def __init__(self):
        self.weight = 0.0
        self.finished = False
        self.ctx = LlamaContext()

    def new_llm(self):
        return CachedLlama(self.ctx)

    def finish(self):
        self.finished = True
    
    def done_stepping(self):
        return self.finished

    def step(self):
        if not self.done_stepping():
            assert False, "step not implemented"
    
    def __str__(self):
        return "Particle with weight " + str(self.weight)
    
    def start(self):
        pass
    
    def score(self, score):
        self.weight += score
    
    def observe_token(self, llm, token_id):
        self.score(llm.observe_token(token_id))
        return token_id

    def sample_token(self, llm, q=None):
        probs = softmax(llm.logits()) if q is None else np.exp(q)
        token_id = np.random.choice(range(len(llm.vocab)), p=probs)
        llm_score = llm.observe_token(token_id)
        if q is not None:
            self.score(llm_score - q[token_id])
        return token_id