import lampl

class EveryOtherWord(lampl.Model):
    def __init__(self, llm, words):
        self.sentence = words.pop(0)
        self.llm = llm.prompt(self.sentence.encode('utf-8'))
        # TODO: don't assume one token per word
        self.remaining_tokens = [self.llm.vocab.index(w) for w in words]
        super().__init__()
    
    def __str__(self):
        return self.sentence
    
    def step(self):
        # Generate a token
        self.sentence += self.llm.vocab[self.sample_token(self.llm)]
        # Observe the next token
        self.sentence += self.llm.vocab[self.observe_token(self.llm, self.remaining_tokens.pop(0))]
        # Check if done
        if len(self.remaining_tokens) == 0:
            self.finish()


# Create the model
llm = lampl.CachedLlama(lampl.LlamaContext())
words = [" Every", " he", " to", " another", "."]
model = EveryOtherWord(llm, words)
particles = lampl.smc(model, 50)
for i, p in enumerate(particles):
    print(f"Particle {i}: {p} (weight {p.weight})")