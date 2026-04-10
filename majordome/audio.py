import sounddevice as sd
import threading
import numpy as np

from .stt import transcribe_audio
from . import shared
from collections import deque


def audio_capture_thread():
    """Capture audio input from the microphone and put it in the audio queue."""

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        shared.audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=shared.SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=shared.CHUNK_SIZE,
                        callback=callback):
        threading.Event().wait()  # runs indefinitely


def vad_asr_thread():
    """Detect voice activity and record audio chunks until silence."""
    pre_roll = deque(maxlen=int(shared.PRE_ROLL_SECS * shared.SAMPLE_RATE / shared.CHUNK_SIZE))
    utterance = []
    speaking = False
    silent_count = 0
    silence_limit = int(shared.SILENCE_SECS * shared.SAMPLE_RATE / shared.CHUNK_SIZE)

    while True:
        chunk = shared.audio_queue.get()
        rms = np.sqrt(np.mean(chunk ** 2))

        if rms > shared.SILENCE_THRESH:
            # Voice detected → cut the TTS if it speaks
            if shared.is_speaking.is_set():
                shared.stop_event.set()

            if not speaking:
                speaking = True
                utterance = list(pre_roll)  # Includes pre-roll
            silent_count = 0
            utterance.append(chunk)

        elif speaking:
            utterance.append(chunk)
            silent_count += 1
            if silent_count >= silence_limit:
                # Utterance ends → transcription
                audio_data = np.concatenate(utterance)
                text = transcribe_audio(audio_data)
                if text.strip():
                    shared.text_queue.put(text.strip())
                # Reset
                speaking = False
                utterance = []
                silent_count = 0
        else:
            pre_roll.append(chunk)
