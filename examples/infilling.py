import llamppl as llp
import numpy as np

class Infilling(llp.Model):
    def __init__(self, words):
        super().__init__()
        self.s = words.pop(0)
        self.ctx = self.new_context(self.s)
        self.remaining_segments = [self.llama.tokenize(w) for w in words]
    
    def start(self):
        self.step()

    def step(self):
        # Generate a token
        n = self.sample(llp.Geometric(0.5)) + 1
        for _ in range(n):
            self.s += self.sample(llp.Transformer(self.ctx))
        # Observe the next tokens
        for token in self.remaining_segments.pop(0):
            self.s += self.observe(llp.Transformer(self.ctx), token)
        # Check if done
        if len(self.remaining_segments) == 0:
            self.observe(llp.Transformer(self.ctx), llp.EOS)
            self.finish()

# Create the model
llp.LLaMAConfig.set_model_path(input("Path to GGML LLaMA model weights: "))
model = Infilling(["Well, you see, every", " he", " to", " another", "!"])
# Run SMC
for i,p in enumerate(llp.smc_steer(model, 4,4)):
    print(f"Particle {i}: {p} (weight {p.weight})")
