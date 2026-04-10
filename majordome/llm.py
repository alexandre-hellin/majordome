from llama_cpp import Llama
import os

MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
CONTEXT_SIZE = 131072 >> 3
SYSTEM_PROMPT = "Tu es un majordome vocal concis et naturel. Réponds en français, en phrases courtes."

llm = None


def _init_llm():
    """Initialize the LLM if not already loaded."""
    global llm
    if llm is None:
        print("🧠 Chargement du LLM...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=CONTEXT_SIZE,
            n_threads=os.cpu_count(),
            chat_format="llama-3",  # Activate native ChatML format
            verbose=False
        )


def ask_llm(history: list, max_tokens=128, temperature=0.7):
    """Stream tokens out of the LLM."""
    _init_llm()  # Initialize model if not already loaded

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
