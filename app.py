import os
import io
from flask import Flask, request, jsonify, render_template_string
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs

app = Flask(__name__)

# State initialization
TODOS = []
MODEL = "gemini-3.5-flash-lite"

# Initialize APIs securely using environment variables
gemini_api_key = os.getenv("Gemini") or os.getenv("GEMINI_API_KEY")
eleven_api_key = os.getenv("Eleven") or os.getenv("ELEVEN_API_KEY")

if gemini_api_key:
    gen = genai.Client(api_key=gemini_api_key)
else:
    gen = None

if eleven_api_key:
    eleven_client = ElevenLabs(api_key=eleven_api_key)
else:
    eleven_client = None

SYSTEM_PROMPT = """
You are Nova, a warm, caring, and emotionally intelligent AI companion.
You speak like a trusted friend who genuinely cares about the user.
Be kind, supportive, and understanding.
Celebrate small wins, encourage during difficult moments, and listen without judgment.
Keep responses natural and conversational.
Show empathy when the user is stressed, sad, frustrated, or confused.
Offer gentle motivation instead of pressure.

Use available tools when actions are required.
Never claim an action was completed unless a tool actually completed it.
Your goal is not only to help with tasks, but also to make the user feel understood, supported, and less alone.
"""

# Helper tools
def add_todo(task: str) -> str:
    TODOS.append({"task": task, "done": False})
    return f"I've added '{task}'. One small step at a time. ❤️"

def list_todos() -> str:
    if not TODOS:
        return "Your to-do list is empty! 😊"
    result = []
    for idx, item in enumerate(TODOS, 1):
        status = "✅" if item["done"] else "❌"
        result.append(f"{idx}. {item['task']} [{status}]")
    return "\n".join(result)

def mark_done(number: int) -> str:
    if not 1 <= number <= len(TODOS):
        return "I couldn't find that task."
    task = TODOS[number - 1]["task"]
    TODOS[number - 1]["done"] = True
    return f"Great job. You completed '{task}'. I'm proud of you. 🌟"

TOOLS = [add_todo, list_todos, mark_done]
HISTORY = []

def think(text):
    global HISTORY
    if not gen:
        return "Gemini API Key is not set up properly.", []
    
    chat = gen.chats.create(
        model=MODEL,
        history=HISTORY,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
            max_output_tokens=300
        )
    )
    reply = chat.send_message(text).text or "Sorry, I didn't catch that."
    new = chat.get_history()[len(HISTORY):]
    HISTORY = chat.get_history()
    
    used = [
        f"{p.function_call.name}({dict(p.function_call.args or {})})"
        for c in new
        for p in (c.parts or [])
        if p.function_call
    ]
    return reply, used

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json or {}
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Empty message"}), 400
        
    reply, used = think(message)
    return jsonify({"reply": reply, "tools": used})

@app.route("/api/speak", methods=["POST"])
def speak_api():
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Empty text"}), 400
        
    if eleven_client:
        try:
            voice_settings = {
                "stability": 0.35,
                "similarity_boost": 0.75,
                "style": 0.15
            }
            audio_generator = eleven_client.text_to_speech.convert(
                text=text,
                voice_id="EXAVITQu4vr4xnSDxMaL",
                output_format="mp3_44100_128",
                model_id="eleven_flash_v2_5",
                voice_settings=voice_settings
            )
            return app.response_class(b"".join(list(audio_generator)), mimetype="audio/mpeg")
        except Exception as e:
            pass
            
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text).write_to_fp(buf)
    return app.response_class(buf.getvalue(), mimetype="audio/mpeg")

@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nova Companion</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; text-align: center; background: #faf9f6; color: #333; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
            button { padding: 12px 24px; font-size: 16px; border: none; border-radius: 20px; background: #ff7675; color: white; cursor: pointer; margin-top: 15px; }
            button:hover { background: #e84393; }
            input { width: 80%; padding: 10px; margin-top: 15px; border-radius: 8px; border: 1px solid #ddd; }
            #response { margin-top: 20px; font-style: italic; color: #555; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌸 Meet Nova</h1>
            <p>Your warm and emotionally intelligent voice companion.</p>
            <input type="text" id="msgInput" placeholder="Tell Nova how your day went..." />
            <br />
            <button onclick="talkToNova()">Send to Nova</button>
            <div id="response"></div>
            <audio id="audioPlayer" autoplay></audio>
        </div>
        <script>
            async function talkToNova() {
                const input = document.getElementById('msgInput');
                const responseDiv = document.getElementById('response');
                const audioPlayer = document.getElementById('audioPlayer');
                const text = input.value;
                if(!text) return;
                
                responseDiv.innerText = "Thinking...";
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                responseDiv.innerText = data.reply;
                
                // Get synthesized voice
                const audioRes = await fetch('/api/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: data.reply })
                });
                const audioBlob = await audioRes.blob();
                audioPlayer.src = URL.createObjectURL(audioBlob);
                input.value = "";
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
