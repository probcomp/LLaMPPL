import sys
import os
import ctypes
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_void_p,
    c_bool,
    POINTER,
    Structure,
    c_uint8,
    c_size_t,
)
import pathlib

# Load the LLaMA library -- code copied from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama_cpp.py
def _load_shared_library(lib_base_name):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".so"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    if "LLAMA_CPP_LIB" in os.environ:
        lib_base_name = os.environ["LLAMA_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict() # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "llama"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# C types
LLAMA_FILE_VERSION = c_int(2)
LLAMA_FILE_MAGIC = b"ggjt"
LLAMA_FILE_MAGIC_UNVERSIONED = b"ggml"
LLAMA_SESSION_MAGIC = b"ggsn"
LLAMA_SESSION_VERSION = c_int(1)

llama_context_p = c_void_p

llama_token = c_int
llama_token_p = POINTER(llama_token)

llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)

class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_int),  # text context
        ("n_parts", c_int),  # -1 for default
        ("n_gpu_layers", c_int),  # number of layers to store in VRAM
        ("seed", c_int),  # RNG seed, 0 for random
        ("f16_kv", c_bool),  # use fp16 for KV cache
        (
            "logits_all",
            c_bool,
        ),  # the llama_eval() call computes all logits, not just the last one
        ("vocab_only", c_bool),  # only load the vocabulary, no weights
        ("use_mmap", c_bool),  # use mmap if possible
        ("use_mlock", c_bool),  # force system to keep model in RAM
        ("embedding", c_bool),  # embedding mode only
        # called with a progress value between 0 and 1, pass NULL to disable
        ("progress_callback", llama_progress_callback),
        # context pointer passed to the progress callback
        ("progress_callback_user_data", c_void_p),
    ]

llama_context_params_p = POINTER(llama_context_params)


LLAMA_FTYPE_ALL_F32 = c_int(0)
LLAMA_FTYPE_MOSTLY_F16 = c_int(1)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q4_0 = c_int(2)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q4_1 = c_int(3)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(
    4
)  # tok_embeddings.weight and output.weight are F16
# LLAMA_FTYPE_MOSTLY_Q4_2 = c_int(5)  # except 1d tensors
# LLAMA_FTYPE_MOSTYL_Q4_3 = c_int(6)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q8_0 = c_int(7)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q5_0 = c_int(8)  # except 1d tensors
LLAMA_FTYPE_MOSTLY_Q5_1 = c_int(9)  # except 1d tensors

# Misc
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

def llama_context_default_params():
    return _lib.llama_context_default_params()
_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params

def llama_init_from_file(path_model, params):
    return _lib.llama_init_from_file(path_model, params)
_lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
_lib.llama_init_from_file.restype = llama_context_p

# Frees all allocated memory
def llama_free(ctx):
    _lib.llama_free(ctx)
_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None

# Returns the number of tokens in the KV cache
def llama_get_kv_cache_token_count(ctx):
    return _lib.llama_get_kv_cache_token_count(ctx)
_lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_token_count.restype = c_int

# Sets the current rng seed.
def llama_set_rng_seed(ctx, seed):
    return _lib.llama_set_rng_seed(ctx, seed)
_lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
_lib.llama_set_rng_seed.restype = None

# Convert the provided text into tokens.
# The tokens pointer must be large enough to hold the resulting tokens.
# Returns the number of tokens on success, no more than n_max_tokens
# Returns a negative number on failure - the number of tokens that would have been returned
# TODO: not sure if correct
def llama_tokenize(
    ctx,
    text,
    tokens,
    n_max_tokens,
    add_bos):
    return _lib.llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)
_lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
_lib.llama_tokenize.restype = c_int

def llama_n_vocab(ctx: llama_context_p) -> c_int:
    return _lib.llama_n_vocab(ctx)
_lib.llama_n_vocab.argtypes = [llama_context_p]
_lib.llama_n_vocab.restype = c_int

# Token logits obtained from the last call to llama_eval()
# The logits for the last token are stored in the last row
# Can be mutated in order to change the probabilities of the next token
# Rows: n_tokens
# Cols: n_vocab
def llama_get_logits(ctx):
    return _lib.llama_get_logits(ctx)
_lib.llama_get_logits.argtypes = [llama_context_p]
_lib.llama_get_logits.restype = c_float_p

# Token Id -> String. Uses the vocabulary in the provided context
def llama_token_to_str(ctx, token):
    return _lib.llama_token_to_str(ctx, token)
_lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
_lib.llama_token_to_str.restype = c_char_p


# Special tokens
def llama_token_bos():
    return _lib.llama_token_bos()
_lib.llama_token_bos.argtypes = []
_lib.llama_token_bos.restype = llama_token


def llama_token_eos():
    return _lib.llama_token_eos()
_lib.llama_token_eos.argtypes = []
_lib.llama_token_eos.restype = llama_token


def llama_token_nl():
    return _lib.llama_token_nl()
_lib.llama_token_nl.argtypes = []
_lib.llama_token_nl.restype = llama_token


# Run the llama inference to obtain the logits and probabilities for the next token.
# tokens + n_tokens is the provided batch of new tokens to process
# n_past is the number of tokens to use from previous eval calls
# Returns 0 on success
def llama_eval_multi(
    ctx,
    tokens,
    token_indices,
    attn_mask,
    n_tokens,
    n_past,
    n_threads
):
    return _lib.llama_eval_multi(ctx, tokens, token_indices, attn_mask, n_tokens, n_past, n_threads)


_lib.llama_eval_multi.argtypes = [llama_context_p, llama_token_p, POINTER(c_int), c_float_p, c_int, c_int, c_int]
_lib.llama_eval_multi.restype = c_int