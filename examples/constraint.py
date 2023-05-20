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


class ConstraintModel:
    # Constructs the model. Should not make any
    # random choices, as it will only be executed
    # one time, before inference begins.
    def __init__(self, llm, prompt, can_follow):
        self.llm = llm
        self.llm.prompt(prompt)
        self.generated = ""
        self.can_follow = can_follow
        self.finished = False

    # Returns True if the particle is done.
    def done_stepping(self):
        return self.finished

    # String for display
    def __str__(self):
        return self.generated

    def step(self):
        # Get logits
        logits = self.llm.logits()
        # Mask by constraint
        mask            = np.array([0.0 if self.can_follow(self.generated, v) else float('-inf') for v in self.llm.vocab])
        q_probs         = lampl.softmax(logits + mask)
        # Generate proposed token
        token_id        = np.random.choice(range(len(self.llm.vocab)), p=q_probs)
        # Update weight
        q_score           = -np.log(q_probs[token_id])
        p_score         = self.llm.observe_token(token_id)

        # Check if done
        if token_id == lampl.llama_cpp.llama_token_eos():
            self.finished = True
        else:
            # Update generated string
            self.generated += self.llm.vocab[token_id]

        # Return the score
        return q_score + p_score

# Create the model
ctx = lampl.LlamaContext()
llm = lampl.CachedLlama(ctx)
model = ConstraintModel(llm, b" THE ADEVENTURES OF PRINCE TIM. By John McInany. Kirkus Reviews, starred pick. This is a story about a boy named Tim, who", can_follow)
particles, weights = lampl.smc(model, 10)