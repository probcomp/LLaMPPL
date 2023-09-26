# LLaMPPL: A Large Language Model Probabilistic Programming Language

LLaMPPL is a research prototype for _language model probabilistic programming_:
specifying language generation tasks by writing probabilistic programs that combine
calls to LLMs, symbolic program logic, and probabilistic conditioning. 
To solve these tasks, LLaMPPL uses a specialized sequential Monte Carlo inference
algorithm. This technique, _SMC steering_, is described in our paper: https://arxiv.org/abs/2306.03081.

**Note: A new version of this library is available at [https://github.com/probcomp/hfppl](https://github.com/probcomp/hfppl) that integrates with HuggingFace language models and supports GPU acceleration.**

## Installation

Clone this repository and run `pip install -e .` in the root directory, or `python setup.py develop` to install in development mode. Then run `python examples/{example}.py`, for one of our examples (`constraints.py`, `infilling.py`, or `prompt_intersection.py`) to
test the installation. You will be prompted for a path to the weights, in GGML format, a pretrained LLaMA model. If you have access to Meta's LLaMA weights, you can follow the instructions [here](https://github.com/alex-lew/llama.cpp/tree/068a0a9c36f4c3a6e8ec58de569e93d47d5b85a1#prepare-data--run) to convert them to the proper format.

## Usage

A LLaMPPL program is a subclass of the `llamppl.Model` class.

```python
from llamppl import Model, Transformer, EOS, TokenCategorical

# A LLaMPPL model subclasses the Model class
class MyModel(Model):

    # The __init__ method is used to process arguments
    # and initialize instance variables.
    def __init__(self, prompt, forbidden_letter):
        super().__init__()

        # The string we will be generating
        self.s         = ""
        # A stateful context object for the LLM, initialized with the prompt
        self.context   = self.new_context(prompt)
        # The forbidden letter
        self.forbidden = forbidden_letter
    
    # The step method is used to perform a single 'step' of generation.
    # This might be a single token, a single phrase, or any other division.
    # Here, we generate one token at a time.
    def step(self):
        # Sample a token from the LLM -- automatically extends `self.context`
        token = self.sample(Transformer(self.context), proposal=self.proposal())

        # Condition on the token not having the forbidden letter
        self.condition(self.forbidden not in str(token).lower())

        # Update the string
        self.s += token

        # Check for EOS or end of sentence
        if token == EOS or str(token) in ['.', '!', '?']:
            # Finish generation
            self.finish()
    
    # Helper method to define a custom proposal
    def proposal(self):
        logits = self.context.logits().copy()
        forbidden_token_ids = [i for (i, v) in enumerate(self.vocab()) if self.forbidden in str(v).lower()]
        logits[forbidden_token_ids] = -float('inf')
        return TokenCategorical(logits)
```

The `Model` class provides a number of useful methods for specifying a LLaMPPL program:

- `self.sample(dist[, proposal])` samples from the given distribution. Providing a proposal does not modify the task description, but can improve inference. Here, for example, we use a proposal that pre-emptively avoids the forbidden letter.
- `self.condition(cond)` conditions on the given Boolean expression.
- `self.new_context(prompt)` creates a new context object, initialized with the given prompt.
- `self.finish()` indicates that generation is complete.
- `self.observe(dist, obs)` performs a form of 'soft conditioning' on the given distribution. It is equivalent to (but more efficient than) sampling a value `v` from `dist` and then immediately running `condition(v == obs)`.

To run inference, we use the `smc_steer` method:
    
```python
from llamppl import smc_steer, LLaMAConfig
# Initialize the model with weights
LLaMAConfig.set_model_path("path/to/weights.ggml")
# Create a model instance
model = MyModel("The weather today is expected to be", "e")
# Run inference
particles = smc_steer(model, 5, 3) # number of particles N, and beam factor K
```

Sample output:
```
sunny.
sunny and cool.
34° (81°F) in Chicago with winds at 5mph.
34° (81°F) in Chicago with winds at 2-9 mph.
```
