import concurrent.futures
import threading
import time

from majordome import audio_capture_thread, vad_asr_thread, llm_tts_thread, get_persona
from majordome import preload_stt, preload_llm, preload_tts, preload_persona


def main():
    _preload_all()
    print(f"🟢 Majordome démarré ({get_persona().display_name}). Ctrl+C pour quitter.")

    threads = [
        threading.Thread(target=audio_capture_thread, daemon=True),
        threading.Thread(target=vad_asr_thread,       daemon=True),
        threading.Thread(target=llm_tts_thread,       daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt.")


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


if __name__ == "__main__":
    main()
