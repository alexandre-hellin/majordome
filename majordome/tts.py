import io
import queue
import threading
import wave
import numpy as np
import sounddevice as sd

from piper import PiperVoice, SynthesisConfig
from .shared import stop_event

MODEL_PATH = "models/fr_FR-upmc-medium.onnx"

voice = None
config = SynthesisConfig(
    length_scale=1.1,
    noise_scale=0.6,
    noise_w_scale=0.8,
    normalize_audio=True
)


def speak_interruptible(stream) -> str:
    """
    Streams LLM tokens and feeds sentences to a TTS worker thread so that
    LLM generation and audio playback run concurrently.
    """
    sentence_queue = queue.Queue()

    def tts_worker():
        while True:
            _sentence = sentence_queue.get()
            if _sentence is None:
                break
            if not stop_event.is_set():
                _speak_text(_sentence)

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    buffer = ""
    full_text = ""

    for chunk in stream:
        if stop_event.is_set():
            break

        token = chunk["choices"][0]["delta"].get("content", "")
        if not token:
            continue

        print(token, end="", flush=True)
        buffer += token
        full_text += token

        if any(buffer.rstrip().endswith(p) for p in (",", ".", "!", "?", "…", "\n")):
            sentence = buffer.strip()
            buffer = ""
            if sentence:
                sentence_queue.put(sentence)

    if buffer.strip() and not stop_event.is_set():
        sentence_queue.put(buffer.strip())

    sentence_queue.put(None)  # sentinel: signal worker to stop
    tts_thread.join()

    print()
    return full_text


def _init_voice():
    """Initialize the Piper voice if not already loaded."""
    global voice
    if voice is None:
        voice = PiperVoice.load(MODEL_PATH)


def _speak_text(text: str):
    """Speak the given text using the Piper voice."""
    try:
        _init_voice()  # Initialize model if not already loaded

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            # Parameters set before piper writes anything
            wav_file.setnchannels(1)  # mono (upmc-medium is mono)
            wav_file.setsampwidth(2)  # 16 bits
            wav_file.setframerate(22050)  # upmc-medium model frequency
            voice.synthesize_wav(text, wav_file, syn_config=config)

        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            framerate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        audio = np.frombuffer(frames, dtype=dtype_map[sampwidth])

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        sd.play(audio, samplerate=framerate)
        sd.wait()

    except Exception as e:
        print("❌ Erreur TTS :", e)
