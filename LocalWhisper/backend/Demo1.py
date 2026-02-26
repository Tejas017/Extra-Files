import sounddevice as sd
import numpy as np
import pyttsx3
from openwakeword.model import Model
from faster_whisper import WhisperModel


# ----------------------------
# Init Models
# ----------------------------
oww_model = Model(wakeword_models=["alexa.onnx"])
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# TTS
tts_engine = pyttsx3.init()

wake_word_detected = False
sample_rate = 16000
current_wake_word = "prognosis"  # default wake word

def record_audio(duration=3, samplerate=16000):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# This function will allow changing wake word dynamically
def change_wake_word(new_wake_word):
    global current_wake_word
    current_wake_word = new_wake_word.strip().lower()
    speak(f"Wake word changed to {current_wake_word}")
    print(f"🔑 Wake word updated: {current_wake_word}")

# Main assistant loop
def assistant():
    global wake_word_detected
    print(f"👂 Listening for wake word: '{current_wake_word}'...")

    while True:
        audio_chunk = record_audio(duration=1)
        prob = oww_model.predict(audio_chunk)

        if prob.max() > 0.9:
            wake_word_detected = True
            speak("Hello user, how can I help you?")

        while wake_word_detected:
            audio_data = record_audio(duration=5)
            segments, _ = whisper_model.transcribe(audio_data, language="en")

            for seg in segments:
                text = seg.text.strip().lower()
                print("📝 You said:", text)

                # Change wake word dynamically
                if "change wake word to" in text:
                    new_word = text.split("change wake word to")[-1].strip()
                    change_wake_word(new_word)

                if "stop" in text:
                    print("🛑 Stopping, waiting for wake word again...")
                    wake_word_detected = False
                    break


if __name__ == "__main__":
    assistant()
