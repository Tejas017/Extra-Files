import subprocess, json

def query_ollama(prompt: str):
    cmd = ["ollama", "run", "mistral"]
    process = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    output = process.stdout.strip()

    # Optional: print stderr for debugging
    if process.stderr:
        print("Ollama stderr:", process.stderr)

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print("Invalid JSON from Ollama:", output)
        data = {"response": output}

    return data