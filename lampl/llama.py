import numpy as np
from . import llama_cpp
import multiprocessing
from .util import *

N_THREADS = multiprocessing.cpu_count()
HOME_PATH = "/Users/alexlew"
MODEL_PATH = f"{HOME_PATH}/gg-llama.cpp/models/7B/ggml-model-q4_0.bin"


class TokenTrie:
    # Trie of tokens. At each node, we store the token and its absolute position in the KV cache.
    # We also may store the logprob or logits of the token, if they have been evaluated.

    # A *particle* points to a node in the Trie, and maintains various statistics for
    # querying the language model.

    def __init__(self, kv_index, parent=None, logprob=None, logits=None):
        self.kv_index = kv_index
        self.children = {} # maps token ID to child
        self.logprob = logprob # of this token, given previous
        self.logits = logits # for next token
        if parent is None:
            self.mask_fragment = [0.0] # BOS token attends to itself
        else:
            num_intervening_tokens = kv_index - parent.kv_index - 1
            self.mask_fragment = [-float('inf')] * num_intervening_tokens + [0.0]
    
    def has_token(self, token_id):
        return token_id in self.children
    
    def get_token(self, token_id):
        return self.children[token_id]
    
    def add_token(self, token_id, kv_index, logprob = None, logits = None):
        self.children[token_id] = TokenTrie(kv_index, self, logprob, logits)
        return self.children[token_id]

class LlamaContext:
    def __init__(self):
        self.ctx = llama_cpp.llama_init_from_file(MODEL_PATH.encode('utf-8'), llama_cpp.llama_context_default_params())
        self.kv_index = -1 # Index of last token in KV cache
        self.vocab = [llama_cpp.llama_token_to_str(self.ctx, token).decode('utf-8', errors='ignore') for token in range(llama_cpp.llama_n_vocab(self.ctx))]

        # We store the root node of a TokenTrie, but the LlamaContext object is not
        # itself responsible for maintaining the cache.
        self.trie = TokenTrie(0)

        # Evaluate beginning-of-sequence token
        self.eval([llama_cpp.llama_token_bos()], [0], [0.0])
    
    def __deepcopy__(self, memo):
        return self

    def eval(self, tokens, indices, attention_mask):
        n_new = len(tokens)
        if self.kv_index + n_new >= 512:
            assert False, "Warning: cache has more than 512 tokens. This violates memory assumptions."

        tokens = (llama_cpp.llama_token * len(tokens))(*tokens)
        indices = (llama_cpp.c_int * len(tokens))(*indices)
        attention_mask = (llama_cpp.c_float * (len(attention_mask)))(*attention_mask)
        llama_cpp.llama_eval_multi(self.ctx, tokens, indices, attention_mask, n_new, self.kv_index+1, N_THREADS)
        self.kv_index += n_new

    def get_last_token_logits(self):
        return np.array(llama_cpp.llama_get_logits(self.ctx)[0:llama_cpp.llama_n_vocab(self.ctx)]) # [-1] -- currently we do not have logits_all = True

    def __str__(self):
        # Using recursion, render a Trie in a visually natural way
        def render(node, indent):
            s = ""
            for token_id, child in node.children.items():
                s += " " * indent + f"{llama_cpp.llama_token_to_str(self.ctx, token_id).decode('utf-8', errors='ignore')} ({child.kv_index})\n"
                s += render(child, indent + 2)
            return s
        return render(self.trie, 0)

def autoregressive_mask(n_tokens):
    return [[llama_cpp.c_float(0.0)] * (i + 1) + [llama_cpp.c_float(float('-inf'))] * (n_tokens - i - 1) for i in range(n_tokens)]


# LLaMA interface used by a particular particle during inference.
# The particle holds a reference to a Trie of tokens, which is shared
# among many particles. Where possible, it reuses results from this
# Trie. When it needs to evaluate a new token, it adds it to the Trie.
class CachedLlama:

    def __init__(self, ctx, trie=None, index=1, mask=None, kv_index=0):
        self.ctx = ctx
        self.vocab = self.ctx.vocab
        self.trie = trie if trie is not None else ctx.trie
        self.current_index = index # BOS is token 0, already included in context
        self.current_mask = mask if mask is not None else [0.0] # BOS token is attended to
        self.kv_index = kv_index # Equal to self.trie.kv_index... so maybe can delete?

    def extend_mask(self):
        if self.kv_index < self.ctx.kv_index:
            self.current_mask.extend([-float('inf')] * (self.ctx.kv_index - self.kv_index))
            self.kv_index = self.ctx.kv_index
    
    def prompt(self, prompt):
        # Tokenize the prompt
        tokens = (llama_cpp.llama_token * (len(prompt) + 1))()
        num_tokens = llama_cpp.llama_tokenize(self.ctx.ctx, prompt, tokens, len(tokens), False)
        tokens = tokens[:num_tokens]

        # Advance in the trie as far as possible
        consumed = 0
        while consumed < num_tokens:
            token = tokens[consumed]
            if self.trie.has_token(token):
                self.trie           = self.trie.get_token(token)
                
                consumed           += 1
                self.current_index += 1
                self.current_mask  += self.trie.mask_fragment
                self.kv_index       = self.trie.kv_index
            else:
                break
        
        num_tokens -= consumed
        tokens      = tokens[consumed:]

        if num_tokens == 0:
            return self

        # Update mask and kv_index
        self.extend_mask()

        # Compute indices and attention mask
        indices = range(self.current_index, self.current_index + num_tokens)
        attention_mask = sum([self.current_mask + m for m in autoregressive_mask(num_tokens)], [])

        # Evaluate
        self.ctx.eval(tokens, indices, attention_mask)

        # Update stats
        for token in tokens:
            self.kv_index += 1
            self.current_index += 1
            self.current_mask.append(0.0)
            self.trie = self.trie.add_token(token, self.kv_index)
        
        # Save logits for end of prompt
        self.trie.logits = self.ctx.get_last_token_logits()

        return self

    def logits(self):
        if self.trie.logits is None and self.trie.kv_index == self.ctx.kv_index:
            self.trie.logits = self.ctx.get_last_token_logits()
        # TODO: error if still None?
        return self.trie.logits
    
    def observe_token(self, token_id):
        # Check if token is in trie
        if self.trie.has_token(token_id):
            # If so, update trie and return logprob
            self.trie = self.trie.get_token(token_id)
            self.current_mask += self.trie.mask_fragment
            self.current_index += 1
            self.kv_index = self.trie.kv_index

            logprob = self.trie.logprob
            if logprob is None and self.trie.parent.logits is not None:
                logprob = np.log(softmax(self.trie.parent.logits)[token_id])
                self.trie.logprob = logprob
            # TODO: error if still None?
            return logprob

        # If not, extend mask and evaluate
        logits = self.logits()
        logprob = (logits - logsumexp(logits))[token_id]
        self.extend_mask()
        self.current_mask.append(0.0)
        self.ctx.eval([token_id], [self.current_index], self.current_mask)
        self.current_index += 1
        self.kv_index += 1
        self.trie = self.trie.add_token(token_id, self.kv_index, logprob, self.ctx.get_last_token_logits())

        return logprob
    
    def observe_tokens(self, tokens):
        score = 0.0
        for token in tokens:
            score += self.observe_token(token)
        
        return score

    def observe_text(self, s):
        # Tokenize string
        tokens = (llama_cpp.llama_token * (len(s) + 1))()
        num_tokens = llama_cpp.llama_tokenize(self.ctx.ctx, s, tokens, len(tokens), False)
        tokens = tokens[:num_tokens]

        # Observe tokens
        return self.observe_tokens(tokens)

    def __deepcopy__(self, memo):
        # Does not copy the context or trie, which are shared across copies.
        # The mask is just a list of floats and can be copied shallowly.
        return CachedLlama(self.ctx, self.trie, self.current_index, self.current_mask.copy(), self.kv_index)

# class LlamaModel:
#     def __init__(self, ctx, index = 0, mask = []):
#         self.ctx = ctx
#         self.tokens = []
#         self.indices = []
#         self.current_index = index
#         self.current_mask = mask
#         self.vocab = self.ctx.vocab
#         self.last_n_tokens = self.ctx.n_tokens
#         self.most_recent_logits = None

#     def extend_mask(self):
#         if self.last_n_tokens < self.ctx.n_tokens:
#             self.current_mask.extend([-float('inf')] * (self.ctx.n_tokens - self.last_n_tokens))
#             self.last_n_tokens = self.ctx.n_tokens
    
#     def prompt(self, prompt):
#         # Extend mask if necessary
#         self.extend_mask()
#         # Tokenize the prompt
#         tokens = (llama_cpp.llama_token * (len(prompt) + 1))()
#         num_tokens = llama_cpp.llama_tokenize(self.ctx, prompt, tokens, len(tokens), True)
#         tokens = tokens[:num_tokens]
#         # Compute indices and attention mask
#         indices = range(self.current_index, self.current_index + num_tokens)
#         attention_mask = [self.current_mask + m for m in autoregressive_mask(num_tokens)]
#         # Evaluate
#         self.ctx.eval(tokens, indices, attention_mask)
#         # Update stats
#         self.current_index += num_tokens
#         self.current_mask = attention_mask[-1]
#         self.tokens.extend(tokens)
#         self.indices.extend(indices)
#         self.last_n_tokens += num_tokens
#         self.most_recent_logits = self.ctx.get_last_token_logits()

#     def logits(self):
#         return self.most_recent_logits
    
#     def observe_token(self, token_id):
#         # Extend mask if necessary
#         self.extend_mask()
#         # Update stats
#         self.current_index += 1
#         self.current_mask.append(0.0)
#         # Compute logprob
#         logprob = softmax(self.most_recent_logits)[token_id]
#         # Evaluate
#         self.ctx.eval([token_id], [self.current_index - 1], [self.current_mask])
#         self.most_recent_logits = self.ctx.get_last_token_logits()
#         self.last_n_tokens += 1
#         # Return the logprob
#         return logprob
    
#     def observe_tokens(self, tokens):
#         score = 0.0
#         for token in tokens:
#             score += self.observe_token(token)
        
#         return score
    
#     def observe_text(self, s):
#         # Tokenize string
#         tokens = (llama_cpp.llama_token * (len(s) + 1))()
#         num_tokens = llama_cpp.llama_tokenize(self.ctx, s, tokens, len(tokens), True)
#         tokens = tokens[:num_tokens]

#         # Observe tokens
#         return self.observe_tokens(tokens)
    
#     def split(self, n):
#         [LlamaModel(self.ctx, self.current_index, self.current_mask.copy()) for _ in range(n)]
    