import lampl

class ProductOfExpertsModel(lampl.Model):
    # Constructs the model. Should not make any
    # random choices, as it will only be executed
    # one time, before inference begins.
    def __init__(self, prompts):
        super().__init__()
        self.llms = [self.new_llm().prompt(prompt) for prompt in prompts]
        self.generated = ""

    # String for display
    def __str__(self):
        return self.generated

    def step(self):
        # Generate proposed token
        token_id = self.sample_token(self.llms[0], q=self.locally_optimal_proposal())

        # Observe from other LLMs
        for llm in self.llms[1:]:
            self.observe_token(llm, token_id)

        # Check for eos 
        if token_id == lampl.EOS:
            self.finish()
            return

        # Update generated string
        self.generated += self.llms[0].vocab[token_id]

        # Check for sentence-ending punctuation
        if self.llms[0].vocab[token_id] in [".", "?", "!"] and not self.generated.split()[-1][0].isupper():
            self.finish()
        
    # Greedy / locally optimal proposal for next token
    def locally_optimal_proposal(self):
        logits = [llm.logits() for llm in self.llms]
        logprobs = [l - lampl.logsumexp(l) for l in logits]
        p_scores = sum(logprobs)
        q_logprobs = p_scores - lampl.logsumexp(p_scores)
        return q_logprobs

# Create the model
prompts = [b" My favorite writer is probably", b" My favorite physicist is probably"]
model = ProductOfExpertsModel(prompts)
# Run SMC
for i, p in enumerate(lampl.smc(model, 20)):
    print(f"Particle {i}: {p} (weight {p.weight})")