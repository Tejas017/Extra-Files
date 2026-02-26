import sounddevice as sd
import numpy as np
from collections import deque
from faster_whisper import WhisperModel

# Load model
model = WhisperModel("small", device="cpu", compute_type="int8")

# Parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 1  # seconds
BUFFER_DURATION = 5  # seconds
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION

# Rolling audio buffer
audio_buffer = deque(maxlen=BUFFER_SIZE)
listening = False  # Wake word gate

def audio_callback(indata, frames, time, status):
    global listening
    if status:
        print("⚠️", status)
    audio_buffer.extend(indata[:, 0])  # Take single channel

def transcribe_buffer():
    global listening
    if len(audio_buffer) < SAMPLE_RATE:  # Need at least 1 sec
        return

    # Convert to numpy array
    audio_np = np.array(audio_buffer, dtype=np.float32)

    # Transcribe last BUFFER_DURATION seconds
    segments, _ = model.transcribe(audio_np, beam_size=1)

    text = " ".join([seg.text for seg in segments]).lower()

    # Wake word detection
    if not listening and "progo" in text:
        print("🚀 Wake word detected: PROGO")
        listening = True
        return

    if listening:
        if "stop" in text:
            print("🛑 Stopping assistant.")
            listening = False
        else:
            print("📝", text)


def main():
    print(" Starting stream... Say 'progo' to activate.")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
        try:
            while True:
                transcribe_buffer()
        except KeyboardInterrupt:
            print("Exiting.")

if __name__ == "__main__":
    main()
