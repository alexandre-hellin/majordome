from faster_whisper import WhisperModel
from .config import config
import torch
import numpy as np

# STT – Input Audio Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESH = 0.005
SILENCE_SECS = 1.5  # Silence after speaking → end of utterance
PRE_ROLL_SECS = 0.4  # Chunks kept before voice detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = config.get("stt", {}).get("model", "small")
LANGUAGE = "fr"
BEAM_SIZE = 5

whisper = None


def transcribe_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Transcribes audio data using the Whisper model and returns the recognized text. The function processes provided
    audio input to ensure compatibility with the model, including format and dimensionality adjustments. It incorporates
    filters to handle short, empty, or unreliable segments of audio, improving transcription accuracy. The transcription
    outputs deterministic text by disabling sampling and applying thresholds for noise and abnormal repetitions.

    :param audio: Audio data represented as a 1D or 2D NumPy array. If 2D, the audio will be averaged across channels.
    :param sample_rate: Sample rate of the audio input in Hertz. Default is defined by the SAMPLE_RATE variable.
    :type audio: numpy.ndarray
    :type sample_rate: int
    :return: Transcribed text as a string, or an empty string for short, invalid, or low-confidence audio.
    :rtype: str
    """
    _init_whisper()  # Initialize model if not already loaded

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Avoid Whisper crashing on short audio
    if len(audio) < sample_rate * 0.3:  # 0.1 → 0.3s: avoid short false positives
        return ""

    # Avoid Whisper crashing on empty audio
    if not np.isfinite(audio).all():
        return ""

    segments, info = whisper.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=True,  # silence filter integrated in Whisper
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
        condition_on_previous_text=False,  # avoid looping hallucinations
        no_speech_threshold=0.6,  # reject low-confidence segments
        compression_ratio_threshold=2.4,  # reject abnormal repetitions
        temperature=0.0,  # disable sampling → more deterministic
    )

    # Filter segments with low speech probability
    texts = [
        seg.text for seg in segments
        if seg.no_speech_prob < 0.6
    ]

    return " ".join(texts).strip()


def preload():
    """Preload Whisper at startup."""
    _init_whisper()


def _init_whisper():
    """Initialize Whisper if not already loaded."""
    global whisper
    if whisper is None:
        whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="auto")
