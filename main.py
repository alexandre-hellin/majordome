import concurrent.futures
import threading
import signal

from majordome import audio_capture_thread, vad_asr_thread, llm_tts_thread, get_persona
from majordome import preload_stt, preload_llm, preload_tts, preload_persona
from majordome import shutdown_tts

_stop_event = threading.Event()


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    _preload_all()
    print(f"🟢 Majordome démarré ({get_persona().display_name}). Ctrl+C pour quitter.")

    # Start STS pipeline
    threads = [
        threading.Thread(target=audio_capture_thread, daemon=True),
        threading.Thread(target=vad_asr_thread,       daemon=True),
        threading.Thread(target=llm_tts_thread,       daemon=True),
    ]
    for t in threads:
        t.start()

    # Wait for SIGINT signal
    _stop_event.wait()
    _shutdown_all()
    print("\n🛑 Arrêt.")


def _signal_handler(sig, frame):
    """Catches SIGINT before other Fortran libs (NumPy, sounddevice…)."""
    _stop_event.set()


def _preload_all():
    print("⏳ Un instant…")
    preload_persona()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(preload_stt),
            executor.submit(preload_llm),
            executor.submit(preload_tts),
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _shutdown_all():
    shutdown_tts()


if __name__ == "__main__":
    main()
