import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

# Load Faster-Whisper small model
model = WhisperModel("small", device="cpu", compute_type="int8")

def record_audio(filename="mic.wav", duration=5, samplerate=16000):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    wav.write(filename, samplerate, audio)
    print(f"✅ Saved recording as {filename}")
    return filename

def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text

if __name__ == "__main__":
    file_path = record_audio(duration=5)  # Record 5 seconds
    print("🔎 Transcribing...")
    result = transcribe_audio(file_path)
    print("📝 Transcription:", result)
