from flask import Flask, request, jsonify
from stt_handler import transcribe_audio
from ollama_handler import query_ollama

app = Flask(__name__)

@app.route("/process_audio", methods=["POST"])
def process_audio():
    audio_file = request.files["file"]
    path = f"/tmp/{audio_file.filename}"
    audio_file.save(path)

    text = transcribe_audio(path)
    response = query_ollama(f"User said: '{text}'. Identify intent.")
    return jsonify({"text": text, "ollama_response": response})

if __name__ == "__main__":
    app.run(debug=True)
