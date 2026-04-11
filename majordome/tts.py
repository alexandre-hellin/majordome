import io
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
    Processes a stream of text data and allows interruption to dynamically handle streaming text generation.
    The function continuously reads chunks from the provided stream while checking for an external interruption
    event. It accumulates and outputs text token by token, grouping them into sentences based on predefined
    delimiters. Once a complete sentence is formed, it processes and outputs the sentence, while allowing
    external interruptions. Finally, any remaining buffered text is processed and returned.

    :param stream: An iterable stream of data chunks where each chunk contains a "choices" key holding
                   a sequence of delta outputs with optional "content" text.
    :type stream: iterable
    :return: The fully processed and concatenated text from the input stream, including all sentences
             generated before interruption or completion.
    :rtype: str
    """
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
            if sentence and not stop_event.is_set():
                _speak_text(sentence)

    if buffer.strip() and not stop_event.is_set():
        _speak_text(buffer.strip())
        full_text += buffer

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
