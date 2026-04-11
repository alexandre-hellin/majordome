import threading
import queue

# ── Shared State ───────────────────────────────────────────────
audio_queue = queue.Queue()  # Raw audio chunks → thread VAD/ASR
text_queue = queue.Queue()  # Text utterances → thread LLM/TTS
stop_event = threading.Event()  # Trigger to cut TTS speaking
is_speaking = threading.Event()  # Triggered when TTS is active
conversation = []  # Shared history (lock protected)
conv_lock = threading.Lock()
