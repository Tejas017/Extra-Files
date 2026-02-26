import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import os
from faster_whisper import WhisperModel
from ollama_handler import query_ollama
import webbrowser
import subprocess

model = WhisperModel("small.en",device="cpu")

def record_audio(duration=4, samplerate=16000):
    print("Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, samplerate)
        return tmp.name

def transcribe_audio(path):
    segments, _ = model.transcribe(path)
    text = " ".join([s.text for s in segments]).strip()
    return text

def classify_text(text):
    prompt = f"""
    The user said: "{text}".
    Determine if this is a command or normal talk.
    If command, identify its intent (open, search, run, etc.).
    Respond in JSON like:
    {{
      "type": "command" or "non-command",
      "intent": "<intent>",
      "target": "<target or detail>"
    }}
    """
    return query_ollama(prompt)

def execute_command(intent, target):
    if intent == "open":
        if "browser" in target:
            webbrowser.open("https://google.com")
        elif "notepad" in target:
            subprocess.Popen(["notepad.exe"])
        else:
            print(f"Unknown open target: {target}")
    elif intent == "search":
        webbrowser.open(f"https://www.google.com/search?q={target}")
    else:
        print(f"No action for intent: {intent}")

def main():
    while True:
        audio_path = record_audio()
        text = transcribe_audio(audio_path)
        if not text:
            continue

        print(f"Heard: {text}")
        response = classify_text(text)
        print("Ollama Response:", response)

        # crude extraction
        if '"type": "command"' in response:
            try:
                import json
                data = json.loads(response)
                if data["type"] == "command":
                    execute_command(data["intent"], data["target"])
            except Exception as e:
                print("Failed to parse Ollama output:", e)
        else:
            print("Non-command speech.")

if __name__ == "__main__":
    main()