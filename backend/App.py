from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

DATA_FILE = "data.json"

@app.route("/save", methods=["POST"])
def save_text():
    data = request.get_json()
    print("Saving:", data)

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)

    with open(DATA_FILE, 'r+') as f:
        existing = json.load(f)
        existing.append(data)
        f.seek(0)
        json.dump(existing, f, indent=4)

    return jsonify({"message": "Data saved!"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
