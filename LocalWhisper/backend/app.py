from flask import Flask
from threading import Thread
import Demo1  # your assistant.py file

app = Flask(__name__)
assistant_thread = None

@app.route("/api/start_assistant", methods=["POST"])
def start_assistant():
    global assistant_thread
    if assistant_thread is None or not assistant_thread.is_alive():
        assistant_thread = Thread(target=Demo1.assistant)
        assistant_thread.start()
    return "Assistant started"

@app.route("/api/stop_assistant", methods=["POST"])
def stop_assistant():
    Demo1.wake_word_detected = False  # stops the loop
    return "Assistant stopped"

if __name__ == "__main__":
    app.run(port=5000)
