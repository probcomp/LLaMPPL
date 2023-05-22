import lampl
import numpy as np

def can_follow(str_so_far, s):
    if len(s.strip()) > 5:
        return False
    if len(s.strip()) == 0:
        return True
    if not s[0].isalpha():
        return True
    if len(str_so_far) == 0:
        return True # First token, can be alphanumeric
    if str_so_far[-1].isalpha() and len(str_so_far.split()[-1]) + len(s) <= 5:
        return True
    else:
        return False


class ConstraintModel(lampl.Model):
    # Constructs the model. Should not make any
    # random choices, as it will only be executed
    # one time, before inference begins.
    def __init__(self, ctx, prompt, can_follow):
        self.llm = lampl.CachedLlama(ctx).prompt(prompt)
        self.generated = ""
        self.can_follow = can_follow
        super().__init__()

    # String for display
    def __str__(self):
        return self.generated

    def step(self):
        # Generate proposed token.
        # The locally optimal proposal guarantees that the
        # constraint is satisfied.
        token_id = self.sample_token(self.llm, q=self.locally_optimal_proposal())

        # Check if done
        if token_id == lampl.EOS:
            self.finish()
            return

        # Update generated string
        self.generated += self.llm.vocab[token_id]
    
    def locally_optimal_proposal(self):
        # Get next token logits
        logits = self.llm.logits()
        # Compute locally optimal proposal
        mask = np.array([0.0 if self.can_follow(self.generated, v) else float('-inf') for v in self.llm.vocab])
        q_logprobs = lampl.lognormalize(logits + mask)
        return q_logprobs

# Create the model
ctx = lampl.LlamaContext()
model = ConstraintModel(ctx, b" THE ADEVENTURES OF PRINCE TIM. By John McInany. Kirkus Reviews, starred pick. This is a story about a boy named Tim, who", can_follow)
particles = lampl.smc(model, 10)