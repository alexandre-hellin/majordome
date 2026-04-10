import threading
import queue

# ── Configuration ──────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESH = 0.015
SILENCE_SECS = 0.8  # Silence after speaking → end of utterance
PRE_ROLL_SECS = 0.3  # Chunks kept before voice detection

# ── Shared State ───────────────────────────────────────────────
audio_queue = queue.Queue()  # Raw audio chunks → thread VAD/ASR
text_queue = queue.Queue()  # Text utterances → thread LLM/TTS
stop_event = threading.Event()  # Trigger to cut TTS speaking
is_speaking = threading.Event()  # Triggered when TTS is active
conversation = []  # Shared history (lock protected)
conv_lock = threading.Lock()
