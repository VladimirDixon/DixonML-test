from flask import Flask, request, render_template, jsonify
import requests
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("ERROR: GROQ_API_KEY is not set. Please check your .env file.")

app = Flask(__name__)

# Home route (renders the HTML chat page)
@app.route("/")
def home():
    return render_template("index.html")

# Default user-configurable settings
app.config['constant_prompt'] = ''
app.config['temperature'] = 0.7
app.config['max_tokens'] = 100

@app.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.get_json()
    app.config['constant_prompt'] = data.get('constant_prompt', '')
    app.config['temperature'] = float(data.get('temperature', 0.7))
    app.config['max_tokens'] = int(data.get('max_tokens', 100))
    return jsonify({'status': 'success'})

# Route to handle chat
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    chat_history = data.get('history', [])

    # Insert the constant prompt at the beginning as a system message
    constant_prompt = app.config.get('constant_prompt', '').strip()
    if constant_prompt:
        chat_history.insert(0, {"role": "system", "content": constant_prompt})

    # Prepare payload for Groq API
    payload = {
        "messages": chat_history,
        "model": "llama-3.3-70b-versatile",
        "temperature": app.config.get('temperature', 0.7),
        "max_tokens": app.config.get('max_tokens', 100)
    }

    # Make request to Groq API
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )

        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
