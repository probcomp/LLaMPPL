import llamppl as llp
import numpy as np

def can_follow(str_so_far, s):
    if isinstance(s, llp.Token):
        s = str(s)
    if len(s.strip()) > 5:
        return False
    if len(s.strip()) == 0:
        return True
    if not s[0].isalpha():
        return True
    if len(str_so_far) == 0:
        return True # First token, can be alphanumeric
    words = str_so_far.split()
    if len(words) >= 1 and len(words[-1]) + len(s) <= 5:
        return True
    else:
        return False

class ConstraintModel(llp.Model):
    # Constructs the model. Should not make any
    # random choices, as it will only be executed
    # one time, before inference begins.
    def __init__(self, prompt, can_follow):
        super().__init__()
        self.context = self.new_context(prompt)
        self.can_follow = can_follow

    def step(self):
        # Generate proposed token.
        token = self.sample(llp.Transformer(self.context), 
                            proposal=self.locally_optimal_proposal())

        # Condition on constraint
        self.condition(self.can_follow(self.s, token))

        # Check if done
        if token == llp.EOS:
            self.finish()
            return

        # Update generated string
        self.s += token
    
    def locally_optimal_proposal(self):
        # Get next token logits
        logits = self.context.logits()
        # Compute locally optimal proposal
        mask = np.array([0.0 if self.can_follow(self.s, v) else float('-inf') for v in self.vocab()])
        q_logprobs = llp.lognormalize(logits + mask)
        return llp.TokenCategorical(q_logprobs)

# Create the model
llp.LLaMAConfig.set_model_path(input("Path to GGML LLaMA model weights: "))
prompt = " The Fed says"
model = ConstraintModel(prompt, can_follow)
for i, p in enumerate(llp.fearnhead_clifford_smc(model, 4, 4)):
    print(f"Particle {i}: {p} (weight {p.weight})")


