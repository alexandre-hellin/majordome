import numpy as np

from .stt import transcribe_audio, SAMPLE_RATE, PRE_ROLL_SECS, SILENCE_SECS, CHUNK_SIZE, SILENCE_THRESH
from .shared import audio_queue, is_speaking, stop_event, text_queue
from collections import deque


def vad_asr_thread():
    """Detect voice activity and record audio chunks until silence."""
    pre_roll = deque(maxlen=int(PRE_ROLL_SECS * SAMPLE_RATE / CHUNK_SIZE))
    utterance = []
    speaking = False
    silent_count = 0
    silence_limit = int(SILENCE_SECS * SAMPLE_RATE / CHUNK_SIZE)

    while True:
        chunk = audio_queue.get()
        rms = np.sqrt(np.mean(chunk ** 2))

        if rms > SILENCE_THRESH:
            # Voice detected → cut the TTS if it speaks
            if is_speaking.is_set():
                stop_event.set()

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
                    text_queue.put(text.strip())
                # Reset
                speaking = False
                utterance = []
                silent_count = 0
        else:
            pre_roll.append(chunk)
