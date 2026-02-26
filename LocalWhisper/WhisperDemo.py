import sounddevice as sd
import numpy as np
from collections import deque
from faster_whisper import WhisperModel
import threading
import queue

# Load Faster-Whisper model
model = WhisperModel("small", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000   # ~0.25 sec
BUFFER_SECONDS = 5  # transcribe every 5 seconds

audio_queue = queue.Queue()
buffer = deque(maxlen=BUFFER_SECONDS * SAMPLE_RATE)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def transcribe_worker():
    while True:
        audio_block = audio_queue.get()
        if audio_block is None:  # stop signal
            break
        buffer.extend(audio_block[:, 0])

        # Only transcribe when buffer has enough audio (5 seconds)
        if len(buffer) >= BUFFER_SECONDS * SAMPLE_RATE:
            audio = np.array(buffer, dtype=np.float32)
            segments, _ = model.transcribe(audio, language="en")
            for seg in segments:
                print(f"You said: {seg.text}")

def run_assistant():
    print("🎤 Assistant started. Speak into the microphone.")
    print("Say something... (Ctrl+C or say stop assistant to exit)\n")

    worker = threading.Thread(target=transcribe_worker, daemon=True)
    worker.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 Assistant stopped.")
    finally:
        audio_queue.put(None)
        worker.join()

if __name__ == "__main__":
    run_assistant()
