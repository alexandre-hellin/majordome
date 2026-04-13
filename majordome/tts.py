import queue
import threading
import sounddevice as sd
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from omnivoice import OmniVoice

from majordome.config import config
from majordome.persona import get_persona
from majordome.shared import stop_event
from majordome.dsp import trim_silence, apply_crossfade

# TTS – Output Audio Configuration
SAMPLE_RATE = 24000
AUDIO_BUFFER_SIZE = 3  # Max sentences pre-generated in advance
CROSSFADE_MS = 10  # Crossfade length between chunks (in milliseconds)
SILENCE_RMS_THRESHOLD = 0.01 # RMS threshold below which audio is considered silence and trimmed
SILENCE_WINDOW_MS = 5 # Analysis window size for RMS calculation (in milliseconds)
MAX_TTS_WORKERS = 2
LANGUAGE="fr"

model: OmniVoice | None = None
tts_audio_queue = queue.Queue(maxsize=AUDIO_BUFFER_SIZE)
_voice_clone_prompt = None


def speak_interruptible(stream) -> str:
    """Stream LLM tokens, synthesize sentences via TTS, and play them back."""
    sentence_queue = queue.Queue()

    tts_thread = threading.Thread(target=_tts_orchestrator, args=(sentence_queue, tts_audio_queue), daemon=True)
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

        ends_with_newline = "\n" in token
        if ends_with_newline:
            buffer = _flush_buffer(buffer, sentence_queue)

    # Send any remaining text
    if buffer.strip() and not stop_event.is_set():
        _flush_buffer(buffer, sentence_queue)

    sentence_queue.put(None)  # Signal end of stream to TTS worker
    tts_thread.join()
    playback_thread.join()

    print()
    return full_text


def preload() -> None:
    """Preload the OmniVoice model at startup."""
    _init_model()
    _warmup_model()


def shutdown() -> None:
    """Shutdown the TTS thread gracefully."""
    stop_event.set()
    tts_audio_queue.put(None)


def _tts_orchestrator(sentence_queue: queue.Queue, audio_queue: queue.Queue) -> None:
    """Generate TTS audio tensors with a pool of concurrent workers, preserving order."""
    with ThreadPoolExecutor(max_workers=MAX_TTS_WORKERS) as executor:
        pending: deque = deque()

        while True:
            sentence = sentence_queue.get()

            if sentence is None:
                break  # Sentinel received, stop accepting sentences

            if not stop_event.is_set():
                pending.append(executor.submit(_generate_tts_audio, sentence))

            # Unstack the completed futures at the top (to preserve the order)
            while pending and pending[0].done():
                _flush_tts_future(pending.popleft(), audio_queue)

        # Clear the remaining futures in order
        for future in pending:
            if stop_event.is_set():
                future.cancel()
            else:
                _flush_tts_future(future, audio_queue)

    audio_queue.put(None)  # Propagate sentinel to playback


def _generate_tts_audio(sentence: str) -> list:
    """Generate TTS audio tensors for a single sentence."""
    return [
        tensor.squeeze().cpu().numpy()
        for tensor in model.generate(text=sentence, language=LANGUAGE, voice_clone_prompt=_voice_clone_prompt)
    ]


def _flush_tts_future(future, audio_queue: queue.Queue) -> None:
    """Write audio chunks from a completed future to the audio queue."""
    try:
        for arr in future.result():
            audio_queue.put(arr)
    except Exception as e:
        print("❌ TTS error:", e)


def _playback_worker(audio_queue: queue.Queue) -> None:
    """Write audio chunks to the output stream with crossfade to avoid glitches."""
    crossfade_samples = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
    previous_tail = np.zeros(crossfade_samples, dtype="float32")

    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            latency="low",
        ) as stream:
            while True:
                audio = audio_queue.get()
                if audio is None:
                    break
                if stop_event.is_set():
                    break

                chunk = trim_silence(audio.astype("float32"), SAMPLE_RATE, SILENCE_WINDOW_MS, SILENCE_RMS_THRESHOLD)
                chunk = apply_crossfade(previous_tail, chunk)

                # Save the tail of this chunk for the next crossfade
                previous_tail = chunk[-crossfade_samples:].copy() if len(chunk) >= crossfade_samples else chunk.copy()

                try:
                    stream.write(chunk)
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


def _init_model() -> None:
    """Initialize the OmniVoice model if not already loaded."""
    global model, _voice_clone_prompt
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Initialize the TTS model
        model = OmniVoice.from_pretrained(
            config.get("tts", {}).get("model", ""),
            device_map=device,
            dtype=dtype,
        )

        # Compile model to optimize CUDA kernels
        if device == "cuda":
            model = torch.compile(model, mode="reduce-overhead")

        # Precalculate the voice clone prompt only once
        _voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=get_persona().audio,
            ref_text=get_persona().voice_transcription,
        )

        # Empty cache to free up precious GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()


def _warmup_model() -> None:
    """
    Warm up the OmniVoice model by running a silent inference pass.
    Pre-allocates memory and compiles kernels on the active device.
    """
    warmup_text = "Bonjour !"

    try:
        device = next(model.parameters()).device

        audio_tensors = model.generate(
            text=warmup_text,
            language=LANGUAGE,
            voice_clone_prompt=_voice_clone_prompt
        )

        # Consume the tensors to trigger full computation without playing audio
        for tensor in audio_tensors:
            _ = tensor.squeeze().cpu().numpy()

        # Synchronize CUDA kernels if running on GPU to ensure warmup is fully complete
        if device.type == "cuda":
            torch.cuda.synchronize()

    except Exception as e:
        print("❌ Model warmup failed:", e)
