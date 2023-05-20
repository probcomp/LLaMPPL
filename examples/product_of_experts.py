import lampl
import numpy as np
import copy

class ProductOfExpertsModel:
    # Constructs the model. Should not make any
    # random choices, as it will only be executed
    # one time, before inference begins.
    def __init__(self, llms):
        self.llms = llms
        self.generated = ""
        self.finished = False

    # Returns True if the particle is done.
    def done_stepping(self):
        return self.finished

    # String for display
    def __str__(self):
        return self.generated

    def step(self):
        # Get logits from each LLM
        logits = [llm.logits() for llm in self.llms]
        # Convert to logprobs
        logprobs = [l - lampl.logsumexp(l) for l in logits]
        # Get unnormalized model probabilities by summing
        p_scores = sum(logprobs)
        # Get proposal probabilities
        q_logprobs = p_scores - lampl.logsumexp(p_scores)
        q_probs = np.exp(q_logprobs)
        # Generate proposed token
        token_id = np.random.choice(range(len(self.llms[0].vocab)), p=q_probs)
        # Update weight
        q_score = -q_logprobs[token_id]
        p_score = sum([llm.observe_token(token_id) for llm in self.llms])

        # Check if done -- eos or sentence-ending punctuation
        if token_id == lampl.llama_cpp.llama_token_eos():
            self.finished = True
        elif self.llms[0].vocab[token_id] in [".", "?", "!"] and not self.generated.split()[-1][0].isupper():
            self.finished = True
            self.generated += self.llms[0].vocab[token_id]
        else:
            # Update generated string
            self.generated += self.llms[0].vocab[token_id]

        # Return the score
        return q_score + p_score

# Create the model
ctx = lampl.LlamaContext()
prompts = [b" My favorite writer is probably", b" My favorite physicist is probably"]
llms = [lampl.CachedLlama(ctx) for prompt in prompts]
for llm, prompt in zip(llms, prompts):
    llm.prompt(prompt)
model = ProductOfExpertsModel(llms)
particles, weights = lampl.smc(model, 20)
for i, p in enumerate(particles):
    print(f"Particle {i}: {p} (weight {weights[i]})")