import subprocess
import json

def query_ollama(prompt, model="gemma3:1b"):
    """
    Sends text to a local Ollama model and returns JSON response.
    """
    cmd = ["ollama", "run", model]
    process = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    output = process.stdout.strip()

    if not output:
        return {"error": "No response from Ollama"}

    # Try to parse JSON safely
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        # In case model returns text, not JSON
        return {"text": output}
