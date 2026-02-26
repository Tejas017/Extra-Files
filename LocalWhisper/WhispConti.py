import time
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# Sliding window settings
WINDOW_SECONDS = 5.0       # analyze last 5s each pass
STEP_SECONDS = 0.5         # run inference every 0.5s
OVERLAP_SECONDS = 0.6      # keep this context so words at the edge aren't cut
MIN_SPEECH_SECONDS = 0.25  # skip empty/very short windows

# Model settings (CPU-friendly)
MODEL_NAME = "tiny.en"     # fast on CPU, English-only
COMPUTE_TYPE = "int8"      # int8 for CPU speed
BEAM_SIZE = 1              # 1=greedy; increase for accuracy (slower)

# Optional: talk back
ENABLE_TTS = False

try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 180)
except Exception:
    ENABLE_TTS = False
# ----------------------------

def speak(text: str):
    if ENABLE_TTS:
        tts_engine.say(text)
        tts_engine.runAndWait()

class AudioRing:
    """Thread-safe ring buffer keeping the last N seconds of audio."""
    def __init__(self, seconds: float, samplerate: int):
        self.max_samples = int(seconds * samplerate)
        self.buf = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
        self.total_samples = 0  # lifetime samples appended

    def append(self, data: np.ndarray):
        with self.lock:
            # flatten to 1D
            mono = data.reshape(-1)
            for x in mono:
                self.buf.append(x)
            self.total_samples += len(mono)

    def get_tail(self, seconds: float, samplerate: int) -> np.ndarray:
        with self.lock:
            n = int(seconds * samplerate)
            if n <= 0:
                return np.array([], dtype=np.float32)
            # take the last n samples
            n = min(n, len(self.buf))
            if n == 0:
                return np.array([], dtype=np.float32)
            arr = np.fromiter(list(self.buf)[-n:], dtype=np.float32)
            return arr

def input_callback(indata, frames, time_info, status):
    if status:
        # Buffer over/underrun notices etc.
        print(f"[sounddevice] {status}")
    audio_ring.append(indata.copy())

def soft_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

def main():
    print("Loading Faster-Whisper...")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

    # We’ll track how far we've already emitted (in seconds) relative to each window.
    # For each run, we know the window starts at (window_end - WINDOW_SECONDS).
    # We only print words whose end_time > (window_start + already_emitted_offset).
    already_emitted_global = 0.0  # total seconds of reliable audio we've emitted

    # Start microphone stream
    print("🎤 Streaming… Press Ctrl+C to stop")
    speak("Assistant started. You can speak now.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=input_callback,
        blocksize=int(SAMPLE_RATE * STEP_SECONDS / 2),  # small blocks -> low latency
    )
    stream.start()

    last_infer_time = 0.0
    start_time = time.time()

    try:
        while True:
            now = time.time()
            if now - last_infer_time < STEP_SECONDS:
                time.sleep(0.01)
                continue

            # Take last WINDOW_SECONDS of audio
            audio_win = audio_ring.get_tail(WINDOW_SECONDS, SAMPLE_RATE)

            # Simple silence gate to skip useless runs
            if len(audio_win) < int(MIN_SPEECH_SECONDS * SAMPLE_RATE) or soft_rms(audio_win) < 1e-3:
                last_infer_time = now
                continue

            # Run transcription on the window
            # NOTE: supply numpy audio directly; word timestamps for fine-grained emission
            segments, info = model.transcribe(
                audio_win,
                beam_size=BEAM_SIZE,
                word_timestamps=True,
                vad_filter=False,          # we keep it simple; you can enable if you like
                language="en"              # tiny.en expects English
            )

            # The window's *absolute* start time in our runtime clock:
            # We treat stream time as (now - duration_of_window)
            window_duration = len(audio_win) / SAMPLE_RATE
            window_start_abs = now - window_duration

            # Emit only "new" words (those that occur after what we've already trusted/emitted).
            # We'll compute a threshold in absolute time.
            new_words = []
            threshold_abs = already_emitted_global  # absolute time we reached last time
            # (we track in "absolute wall-clock seconds since start()", using start_time as zero)
            threshold_abs_clock = start_time + threshold_abs

            for seg in segments:
                if not hasattr(seg, "words") or not seg.words:
                    continue
                for w in seg.words:
                    # Word times are relative to the *window*, not wall clock. Convert:
                    w_start_abs = window_start_abs + (w.start or 0.0)
                    w_end_abs = window_start_abs + (w.end or 0.0)
                    # Emit only if this word ends after the threshold (new word)
                    if w_end_abs > threshold_abs_clock:
                        txt = (w.word or "").strip()
                        if txt:
                            new_words.append((w_end_abs, txt))

            # Sort by time and print contiguous words as a line
            if new_words:
                new_words.sort(key=lambda t: t[0])
                line = " ".join(w for _, w in new_words).strip()
                if line:
                    print(line)
                    if ENABLE_TTS:
                        speak(line)

                # Advance the emitted frontier, but keep an overlap cushion
                latest_word_end_abs = new_words[-1][0]
                # Pull back by OVERLAP_SECONDS to allow corrections at boundary
                new_frontier_abs = max(threshold_abs_clock, latest_word_end_abs - OVERLAP_SECONDS)
                # Convert from clock to "seconds since start_time"
                already_emitted_global = max(already_emitted_global, new_frontier_abs - start_time)

            last_infer_time = now

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        speak("Assistant stopped. Goodbye!")
    finally:
        stream.stop()
        stream.close()

# Global ring buffer (created before main so callback can use it)
audio_ring = AudioRing(seconds=WINDOW_SECONDS * 2.0, samplerate=SAMPLE_RATE)

if __name__ == "__main__":
    main()
