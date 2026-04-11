import sounddevice as sd
import threading

from .stt import SAMPLE_RATE, CHUNK_SIZE
from .shared import audio_queue

def audio_capture_thread():
    """Capture audio input from the microphone and put it in the audio queue."""

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=CHUNK_SIZE,
                        callback=callback):
        threading.Event().wait()  # runs indefinitely
