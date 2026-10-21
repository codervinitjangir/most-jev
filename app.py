import os
import io
import streamlit as st
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from gtts import gTTS

# Configure Streamlit page
st.set_page_config(
    page_title="🌸 Nova - AI Companion",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stChatInput { margin-bottom: 20px; }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background: #f1f2f6;
        font-size: 0.85rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌸 Nova Companion")
st.caption("Your warm, emotionally intelligent AI voice companion & todo assistant.")

# Fetch API keys from Streamlit secrets or OS environment
gemini_key = (
    st.secrets.get("GEMINI_API_KEY") 
    or st.secrets.get("Gemini") 
    or os.getenv("GEMINI_API_KEY") 
    or os.getenv("Gemini")
)
eleven_key = (
    st.secrets.get("ELEVEN_API_KEY") 
    or st.secrets.get("Eleven") 
    or os.getenv("ELEVEN_API_KEY") 
    or os.getenv("Eleven")
)

# Sidebar for configuration and To-Do overview
with st.sidebar:
    st.header("⚙️ Configuration")
    if not gemini_key:
        gemini_key = st.text_input("Gemini API Key", type="password")
    else:
        st.success("✅ Gemini Connected")

    if not eleven_key:
        eleven_key = st.text_input("ElevenLabs API Key (Optional)", type="password")
    else:
        st.success("✅ ElevenLabs Connected")
        
    st.markdown("---")
    st.header("📋 Your To-Do List")
    if "todos" not in st.session_state:
        st.session_state.todos = []

    if not st.session_state.todos:
        st.write("No tasks yet! Ask Nova to add tasks for you.")
    else:
        for idx, item in enumerate(st.session_state.todos, 1):
            check = "✅" if item["done"] else "⬜"
            st.write(f"{check} **{idx}.** {item['task']}")

# Initialize Clients
gen_client = genai.Client(api_key=gemini_key) if gemini_key else None
eleven_client = ElevenLabs(api_key=eleven_key) if eleven_key else None

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

# Tool functions for Gemini
def add_todo(task: str) -> str:
    """Add a new task to the user's todo list."""
    st.session_state.todos.append({"task": task, "done": False})
    return f"I've added '{task}'. One small step at a time. ❤️"

def list_todos() -> str:
    """List all current tasks and their status."""
    if not st.session_state.todos:
        return "Your to-do list is empty! 😊"
    res = []
    for idx, item in enumerate(st.session_state.todos, 1):
        status = "✅" if item["done"] else "❌"
        res.append(f"{idx}. {item['task']} [{status}]")
    return "\n".join(res)

def mark_done(number: int) -> str:
    """Mark a task as done by its task number (1-based index)."""
    todos = st.session_state.todos
    if not (1 <= number <= len(todos)):
        return "I couldn't find that task number."
    task = todos[number - 1]["task"]
    todos[number - 1]["done"] = True
    return f"Great job! You completed '{task}'. I'm proud of you. 🌟"

TOOLS = [add_todo, list_todos, mark_done]

# Initialize message state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_history" not in st.session_state:
    st.session_state.gemini_history = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# Text-to-speech generator helper
def generate_audio(text: str) -> bytes:
    if eleven_client:
        try:
            audio_generator = eleven_client.text_to_speech.convert(
                text=text,
                voice_id="EXAVITQu4vr4xnSDxMaL",
                output_format="mp3_44100_128",
                model_id="eleven_flash_v2_5",
                voice_settings={"stability": 0.35, "similarity_boost": 0.75, "style": 0.15}
            )
            return b"".join(list(audio_generator))
        except Exception:
            pass

    # Fallback to gTTS
    buf = io.BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(buf)
    return buf.getvalue()

# Handle input
user_prompt = st.chat_input("Tell Nova how your day went or manage your todos...")

if user_prompt:
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Process response
    if not gen_client:
        with st.chat_message("assistant"):
            warning_msg = "Please provide your Gemini API key in the sidebar or via Streamlit secrets to chat with Nova."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Nova is thinking..."):
                chat = gen_client.chats.create(
                    model="gemini-2.5-flash",
                    history=st.session_state.gemini_history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=TOOLS,
                        max_output_tokens=400
                    )
                )
                response = chat.send_message(user_prompt)
                reply_text = response.text or "I'm here with you."
                st.session_state.gemini_history = chat.get_history()

                st.write(reply_text)

                # Generate speech
                try:
                    audio_bytes = generate_audio(reply_text)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply_text,
                        "audio": audio_bytes
                    })
                except Exception:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply_text
                    })
                
                # Rerun to update sidebar todo changes if any tool modified state
                st.rerun()
