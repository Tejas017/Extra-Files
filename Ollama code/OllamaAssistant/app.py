import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel
from ollama_handler import query_ollama
import subprocess, webbrowser, json, re

# Whisper setup
model = WhisperModel("small.en", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000
BLOCK_SIZE = 5
OVERLAP = 1
audio_queue = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def execute_command(intent, target):
    intent = intent.lower()
    target = target.lower()

    if intent in ["open", "open_application"]:
        if "chrome" in target or "browser" in target:
            subprocess.Popen(["start", "chrome"], shell=True)
        elif "notepad" in target:
            subprocess.Popen(["notepad.exe"])
        elif "youtube" in target:
            webbrowser.open("https://www.youtube.com")
        else:
            print(f"Unknown open target: {target}")
    elif intent == "search":
        webbrowser.open(f"https://www.google.com/search?q={target}")
    else:
        print(f"No defined action for intent '{intent}'")

def parse_ollama_response(response):
    """
    Normalize Ollama's wrapped JSON into a real dict.
    """
    if isinstance(response, dict):
        response_text = response.get("text", "")
    else:
        response_text = str(response)

    # Extract JSON block
    match = re.search(r'\{[\s\S]*\}', response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"type": "non-command", "text": response_text}

def stream_assistant():
    print("🎙️ Voice Assistant listening... (Ctrl+C to stop)")
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

                    if not text:
                        buffer = buffer[-(SAMPLE_RATE * OVERLAP):]
                        continue

                    print(f"Heard: {text}")

                    prompt = f"""
                    The user said: "{text}".
                    Determine if it's a command or normal speech.
                    If it's a command, return structured JSON:
                    {{
                      "type": "command" or "non-command",
                      "intent": "<intent>",
                      "target": "<target or detail>"
                    }}
                    """

                    response = query_ollama(prompt)
                    print("Ollama raw:", response)

                    parsed = parse_ollama_response(response)
                    print("Parsed:", parsed)

                    if parsed.get("type") == "command":
                        execute_command(parsed.get("intent", ""), parsed.get("target", ""))
                    else:
                        print("Ollama:", parsed.get("text", ""))

                    buffer = buffer[-(SAMPLE_RATE * OVERLAP):]

        except KeyboardInterrupt:
            print("\n🛑 Assistant stopped.")

if __name__ == "__main__":
    stream_assistant()
