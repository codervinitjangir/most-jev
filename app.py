import os
import io
import streamlit as st
from google import genai
from gtts import gTTS

st.set_page_config(page_title="🌸 Nova AI Voice", page_icon="🌸")

st.title("🌸 Nova AI Companion")

# Get API key from secrets, env or sidebar input
default_key = (
    st.secrets.get("GEMINI_API_KEY")
    or st.secrets.get("Gemini")
    or os.getenv("GEMINI_API_KEY")
    or ""
)

api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

if not api_key:
    st.info("👈 Please enter your Gemini API Key in the sidebar to start chatting.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# The model name required by your API Key / Account
MODEL_NAME = "gemini-3.8-flash"

# Maintain simple chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# Input options: Voice recorder or Text
st.write("🎙️ **Record Voice:**")
audio_record = st.audio_input("Speak to Nova", key="audio_in")
text_record = st.chat_input("Or type here...")

user_text = None
if audio_record:
    audio_id = f"{audio_record.name}_{audio_record.size}"
    if st.session_state.get("last_audio_id") != audio_id:
        st.session_state["last_audio_id"] = audio_id
        with st.spinner("Transcribing voice..."):
            try:
                audio_bytes = audio_record.getvalue()
                mime = audio_record.type or "audio/wav"
                res = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        genai.types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                        "Transcribe this spoken audio verbatim. Output only the spoken words, nothing else."
                    ]
                )
                user_text = res.text.strip() if res.text else None
            except Exception as e:
                st.error(f"Error reading voice: {e}")

elif text_record:
    user_text = text_record

if user_text:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Get response from Nova
    with st.chat_message("assistant"):
        with st.spinner("Nova is thinking..."):
            try:
                prompt = (
                    "You are Nova, a friendly, warm, caring companion. "
                    "Keep your replies concise, supportive, and natural.\n\n"
                    f"User: {user_text}"
                )
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                bot_reply = response.text or "I'm here with you."
                st.write(bot_reply)

                # Generate speech with gTTS
                try:
                    tts_buf = io.BytesIO()
                    tts = gTTS(text=bot_reply, lang="en")
                    tts.write_to_fp(tts_buf)
                    audio_data = tts_buf.getvalue()

                    st.audio(audio_data, format="audio/mp3", autoplay=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_reply,
                        "audio": audio_data
                    })
                except Exception:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_reply
                    })

            except Exception as err:
                st.error(f"Gemini API Error: {err}")
