import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os
import scipy.io.wavfile as wav

model = WhisperModel("small", device="cpu", compute_type="int8")

def record_and_transcribe(duration=5, samplerate=16000):
    print(f"🎙️ Recording {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    # Save to a secure temporary file (auto-deletes after use)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
        wav.write(tmpfile.name, samplerate, audio)
        segments, info = model.transcribe(tmpfile.name, beam_size=5)
        text = " ".join([segment.text for segment in segments])
    return text

print("📝 Transcription:", record_and_transcribe())
