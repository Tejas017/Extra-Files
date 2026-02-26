from faster_whisper import WhisperModel

model = WhisperModel("small")  # or "base", "medium", "large-v2"

segments, info = model.transcribe("audio.wav")
text = " ".join([segment.text for segment in segments])
print(text)
