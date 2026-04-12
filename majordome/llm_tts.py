from .llm import ask_llm, maybe_summarize_history
from .tts import speak_interruptible
from . import shared


def llm_tts_thread():
    """Interact with the LLM and TTS in a loop."""
    while True:
        text = shared.text_queue.get()
        print(f"👤 {text}")

        with shared.conv_lock:
            shared.conversation.append({"role": "user", "content": text})
            shared.conversation = maybe_summarize_history(shared.conversation)
            history = list(shared.conversation)

        shared.stop_event.clear()
        shared.is_speaking.set()

        stream = ask_llm(history)
        full_response = speak_interruptible(stream)

        shared.is_speaking.clear()

        with shared.conv_lock:
            shared.conversation.append({"role": "assistant", "content": full_response})
