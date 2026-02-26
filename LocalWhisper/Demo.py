import sounddevice as sd
import numpy as np
import pyttsx3
from openwakeword.model import Model
from faster_whisper import WhisperModel
import librosa

# ----------------------------
# Helper: Convert audio → mel spectrogram
# ----------------------------
def audio_to_mel(audio, sr=16000, n_mels=40, n_fft=400, hop_length=160):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return np.expand_dims(mel_spec.T, axis=0)  # (1, time, n_mels)

# ----------------------------
# Init Models
# ----------------------------
oww_model = Model(
    wakeword_models=["train_model/models/prognosis.onnx"],
    inference_framework="onnx"
)
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

# Change wake word dynamically
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
        mel = audio_to_mel(audio_chunk, sample_rate)

        prob = oww_model.predict(mel)
        print(prob)

        # if prob.max() > 0.9:
        #     wake_word_detected = True
        #     speak("Hello user, how can I help you?")

        if max(prob.values()) > 0.7:
            detected = max(prob, key=prob.get)  # keyword with highest probability
            print(f"🔑 Detected wake word: {detected} (prob={prob[detected]:.2f})")
            # print(f"🔑 Detected wake word: {detected}")
            speak("Hello user, how can I help you?")
            if detected == current_wake_word:
                wake_word_detected = True
                speak("Hello user, how can I help you?")

        while wake_word_detected:
            audio_data = record_audio(duration=5)
            segments, _ = whisper_model.transcribe(audio_data, language="en")

            for seg in segments:
                text = seg.text.strip().lower()
                print("📝 You said:", text)

                if "change wake word to" in text:
                    new_word = text.split("change wake word to")[-1].strip()
                    change_wake_word(new_word)

                if "stop" in text:
                    print("🛑 Stopping, waiting for wake word again...")
                    wake_word_detected = False
                    break

if __name__ == "__main__":
    assistant()
