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

# Store user settings (constant prompt, temperature, max tokens)
user_settings = {
    "constant_prompt": "",
    "temperature": 0.7,  # Default value
    "max_tokens": 100  # Default value
}

@app.route("/set_settings", methods=["POST"])
def set_settings():
    """Stores the constant prompt, temperature, and max tokens when the user clicks 'Save'."""
    data = request.json

    # Update settings only when explicitly saved
    if "constant_prompt" in data:
        user_settings["constant_prompt"] = data["constant_prompt"]
    if "temperature" in data:
        user_settings["temperature"] = float(data["temperature"])
    if "max_tokens" in data:
        user_settings["max_tokens"] = int(data["max_tokens"])

    return jsonify({
        "message": "Settings saved successfully.",
        "current_settings": user_settings
    })

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat interaction with Groq API."""
    data = request.json
    user_message = data.get("message", "")

    # Retrieve stored settings
    constant_prompt = user_settings["constant_prompt"]
    temperature = user_settings["temperature"]
    max_tokens = user_settings["max_tokens"]

    # Construct the full prompt
    full_prompt = f"{constant_prompt}\nUser: {user_message}\nAI:"

    # Prepare request payload for Groq API
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "system", "content": constant_prompt}, 
                     {"role": "user", "content": user_message}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Call Groq API
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    print("Groq API Response:", response.status_code, response.text)  # Debugging output

    # Handle API response
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
    else:
        reply = "Error: Could not get response from AI."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
