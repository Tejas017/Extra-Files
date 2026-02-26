import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import subprocess
import json

# ---- Ollama helper ----
def query_ollama(prompt, model="gemma3:1b"):
    """Send prompt to Ollama and return text output."""
    try:
        process = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True
        )
        return process.stdout.strip()
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# ---- Whisper setup ----
model = WhisperModel("small.en", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
BLOCK_SIZE = 5  # seconds per transcription block
OVERLAP = 1     # seconds overlap

audio_queue = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def correct_transcription(raw_text):
    """Ask Ollama to correct and verify transcription."""
    prompt = f"""
    You are a speech transcription corrector.
    The speech recognizer output is: "{raw_text}".
    If it contains mistakes or wrong words due to pronunciation, correct it.
    Respond in JSON:
    {{
      "original": "{raw_text}",
      "corrected": "<corrected sentence>",
      "changed": true or false
    }}
    """
    response = query_ollama(prompt)
    try:
        data = json.loads(response)
        return data.get("corrected", raw_text)
    except json.JSONDecodeError:
        # fallback if Ollama doesn't return JSON
        return response.strip()

def stream_transcribe():
    print("🎙️ Listening continuously... Press Ctrl+C to stop.")
    buffer = np.zeros((0,), dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.5),
        callback=callback
    ):
        try:
            while True:
                block = audio_queue.get()
                buffer = np.concatenate([buffer, block.flatten()])

                if len(buffer) >= SAMPLE_RATE * BLOCK_SIZE:
                    chunk = buffer[-(SAMPLE_RATE * (BLOCK_SIZE + OVERLAP)):]
                    segments, _ = model.transcribe(chunk, language="en", beam_size=1)
                    text = " ".join([seg.text for seg in segments]).strip()

                    if text:
                        print(f"\nRaw Transcription: {text}")
                        corrected = correct_transcription(text)
                        print(f"✅ Corrected: {corrected}")

                    buffer = buffer[-(SAMPLE_RATE * OVERLAP):]

        except KeyboardInterrupt:
            print("\n🛑 Stopped listening.")

if __name__ == "__main__":
    stream_transcribe()
