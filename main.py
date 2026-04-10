import threading
import time
from majordome import audio_capture_thread, vad_asr_thread, llm_tts_thread

def main():
    print("🟢 Majordome démarré. Ctrl+C pour quitter.")
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

if __name__ == "__main__":
    main()