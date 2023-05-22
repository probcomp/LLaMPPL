import lampl

class EveryOtherWord(lampl.Model):
    def __init__(self, llm, words):
        self.llm = llm.prompt(words[0].encode('utf-8'))
        self.words = words[1:]
        self.tokens = [self.llm.vocab.index(w) for w in self.words]
        self.generated = words[0]
        super().__init__()
    
    def __str__(self):
        return self.generated
    
    def step(self):
        # Generate a token
        self.generated += self.llm.vocab[self.sample_token(self.llm)]
        # Observe the next token
        self.observe_token(self.llm, self.tokens.pop(0))
        self.generated += self.words.pop(0)
        # Check if done
        if len(self.words) == 0:
            self.finish()


# Create the model
llm = lampl.CachedLlama(lampl.LlamaContext())
words = [" Every", " he", " to", " another", "."]
model = EveryOtherWord(llm, words)
particles = lampl.smc(model, 50)
for i, p in enumerate(particles):
    print(f"Particle {i}: {p} (weight {p.weight})")