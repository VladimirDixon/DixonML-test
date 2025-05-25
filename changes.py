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

# Default user-configurable settings
app.config['model'] = 'llama3-8b-8192' # Default model
app.config['ai_name'] = 'DixonLM'
app.config['constant_prompt'] = '' 
app.config['temperature'] = 0.7
app.config['max_tokens'] = 250
app.config['top_p'] = 1.0
app.config['stop_sequences'] = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/get_settings', methods=['GET'])
def get_settings():
    settings = {
        'model': app.config.get('model'), # New
        'ai_name': app.config.get('ai_name'),
        'constant_prompt': app.config.get('constant_prompt'),
        'temperature': app.config.get('temperature'),
        'max_tokens': app.config.get('max_tokens'),
        'top_p': app.config.get('top_p'),
        'stop_sequences': app.config.get('stop_sequences')
    }
    if settings['stop_sequences'] and isinstance(settings['stop_sequences'], list):
        settings['stop_sequences_str'] = ",".join(settings['stop_sequences'])
    else:
        settings['stop_sequences_str'] = ""
    return jsonify(settings)

@app.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.get_json()
    app.config['model'] = data.get('model', app.config['model']) # New
    app.config['ai_name'] = data.get('ai_name', app.config['ai_name']).strip()
    app.config['constant_prompt'] = data.get('constant_prompt', app.config['constant_prompt']).strip()
    app.config['temperature'] = float(data.get('temperature', app.config['temperature']))
    app.config['max_tokens'] = int(data.get('max_tokens', app.config['max_tokens']))
    app.config['top_p'] = float(data.get('top_p', app.config['top_p']))
    
    stop_sequences_str = data.get('stop_sequences', '').strip()
    if stop_sequences_str:
        app.config['stop_sequences'] = [s.strip() for s in stop_sequences_str.split(',') if s.strip()]
    else:
        app.config['stop_sequences'] = None
        
    return jsonify({'status': 'success', 'message': 'Settings saved!'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    chat_history = data.get('history', [])

    ai_model_name = app.config.get('ai_name', 'AI').strip()
    user_defined_system_prompt = app.config.get('constant_prompt', '').strip()

    full_system_prompt_parts = []
    if ai_model_name: 
        full_system_prompt_parts.append(f"You are {ai_model_name}.")
    if user_defined_system_prompt:
        full_system_prompt_parts.append(user_defined_system_prompt)
    
    final_system_prompt = "\n\n".join(full_system_prompt_parts).strip() 

    if final_system_prompt:
        if not chat_history or chat_history[0].get("role") != "system":
            chat_history.insert(0, {"role": "system", "content": final_system_prompt})
        elif chat_history[0].get("role") == "system":
            chat_history[0]["content"] = final_system_prompt 
            
    payload = {
        "messages": chat_history,
        "model": app.config.get('model'), # Use the selected model from config
        "temperature": app.config.get('temperature', 0.7),
        "max_tokens": app.config.get('max_tokens', 250),
        "top_p": app.config.get('top_p', 1.0)
    }

    if app.config.get('stop_sequences'):
        payload["stop"] = app.config.get('stop_sequences')

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
        reply_data = response.json()
        
        reply = "Error: No valid response from API."
        if reply_data.get("choices") and len(reply_data["choices"]) > 0:
            message = reply_data["choices"][0].get("message")
            if message and "content" in message:
                reply = message["content"]
            elif message: 
                reply = "Received an empty message content from API."
        else: 
            reply = "API did not return any 'choices'."
            if reply_data.get("error") and reply_data["error"].get("message"):
                reply = f"API Error: {reply_data['error']['message']}"

        return jsonify({"reply": reply})

    except requests.exceptions.RequestException as e:
        error_message = f"API Request Error: {str(e)}"
        if e.response is not None:
            try:
                error_detail = e.response.json()
                if isinstance(error_detail.get("error"), dict) and error_detail["error"].get("message"):
                     error_message += f" - Details: {error_detail['error']['message']}"
                else:
                    error_message += f" - Details: {error_detail}"
            except ValueError: 
                error_message += f" - Server Response: {e.response.text}"
        return jsonify({"error": error_message}), 500
    except Exception as e: 
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))