import queue
import threading
import sounddevice as sd
import torch

from omnivoice import OmniVoice

from .config import config
from .persona import get_persona
from .shared import stop_event

# TTS – Output Audio Configuration
SAMPLE_RATE = 24000
AUDIO_BUFFER_SIZE = 4  # Max sentences pre-generated in advance
SENTENCE_END_TOKENS = (",", ".", "!", "?", "…", "\n")

model = None
tts_audio_queue = queue.Queue(maxsize=AUDIO_BUFFER_SIZE)

def speak_interruptible(stream) -> str:
    """Stream LLM tokens, synthesize sentences via TTS, and play them back."""
    sentence_queue = queue.Queue()

    tts_thread = threading.Thread(target=_tts_worker, args=(sentence_queue, tts_audio_queue), daemon=True)
    playback_thread = threading.Thread(target=_playback_worker, args=(tts_audio_queue,), daemon=True)
    tts_thread.start()
    playback_thread.start()

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

        if buffer.rstrip().endswith(SENTENCE_END_TOKENS):
            buffer = _flush_buffer(buffer, sentence_queue)

    # Send any remaining text
    if buffer.strip() and not stop_event.is_set():
        _flush_buffer(buffer, sentence_queue)

    sentence_queue.put(None)  # Signal end of stream to TTS worker
    tts_thread.join()
    playback_thread.join()

    print()
    return full_text


def preload():
    """Preload the OmniVoice model at startup."""
    _init_model()
    _warmup_model()


def shutdown():
    """Shutdown the TTS thread gracefully."""
    stop_event.set()
    tts_audio_queue.put(None)


def _tts_worker(sentence_queue: queue.Queue, audio_queue: queue.Queue) -> None:
    """Generate TTS audio tensors ahead of playback."""
    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            audio_queue.put(None)  # Propagate sentinel to playback
            break
        if stop_event.is_set():
            continue  # Drain queue without processing
        try:
            for tensor in model.generate(text=sentence, ref_audio=get_persona().audio):
                audio_queue.put(tensor.squeeze().cpu().numpy())
        except Exception as e:
            print("❌ TTS error:", e)


def _playback_worker(audio_queue: queue.Queue) -> None:
    """Write audio chunks to the output stream block by block."""
    try:
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while True:
                audio = audio_queue.get()
                if audio is None:
                    break
                if not stop_event.is_set():
                    try:
                        stream.write(audio.astype("float32"))
                    except sd.PortAudioError:
                        break
    except Exception:
        pass  # Avoid unexpected exception from happening when exiting the thread with SIGINT

def _flush_buffer(buffer: str, sentence_queue: queue.Queue) -> str:
    """Send a completed sentence to the TTS queue and reset the buffer."""
    sentence = buffer.strip()
    if sentence:
        sentence_queue.put(sentence)
    return ""


def _init_model():
    """Initialize the OmniVoice model if not already loaded."""
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = OmniVoice.from_pretrained(
            config.get("tts", {}).get("model", ""),
            device_map=device,
            dtype=dtype,
        )


def _warmup_model():
    """
    Warm up the OmniVoice model by running a silent inference pass.
    This pre-allocates memory and improves performance.
    """
    warmup_text = "Bonjour !"

    try:
        audio_tensors = model.generate(
            text=warmup_text,
            ref_audio=get_persona().audio,
        )

        # Consume the tensors to trigger full computation without playing audio
        for tensor in audio_tensors:
            _ = tensor.squeeze().cpu().numpy()

    except Exception as e:
        print("❌ Model warmup failed:", e)
