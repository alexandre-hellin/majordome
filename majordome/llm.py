from llama_cpp import Llama, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0
from pathlib import Path

from .config import config
from .persona import get_persona

import os
import sys
import random
import torch

# LLM – Large Language Model Configuration
MODEL_PATH = str(Path(__file__).parent.parent / "models" / config.get("llm", {}).get("model", ""))
CONTEXT_SIZE = 131072 >> 3

llm = None

def ask_llm(history: list, max_tokens=512, temperature=1.0, seed=None):
    """Stream tokens out of the LLM."""
    random.seed(seed)

    messages = [{"role": "system", "content": get_persona().render_prompt()}] + history

    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=64,
        seed=random.randint(~sys.maxsize, sys.maxsize),
        stream=True
    )


def preload():
    """Preload the LLM at startup."""
    _init_llm()
    _warmup()


def _init_llm():
    """Initialize the LLM if not already loaded."""
    global llm
    if llm is None:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=CONTEXT_SIZE,
            n_threads=os.cpu_count() // 2,
            n_batch=512,
            n_ubatch=512,
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            type_k=GGML_TYPE_Q8_0,
            type_v=GGML_TYPE_Q4_0,
            flash_attn=True,
            verbose=False
        )


def _warmup():
    """Warm up the LLM by generating a single token."""
    messages = [{"role": "system", "content": get_persona().render_prompt()}]

    llm.create_chat_completion(
        messages=messages,
        max_tokens=1  # Only prefill, no generation
    )