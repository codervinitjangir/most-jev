import os
import io
import json
import urllib.request
import streamlit as st
from google import genai
from gtts import gTTS

st.set_page_config(page_title="🌸 Nova AI Voice", page_icon="🌸")

st.title("🌸 Nova AI Companion")

# Get API keys securely from secrets or environment
default_gemini = (
    st.secrets.get("GEMINI_API_KEY")
    or st.secrets.get("Gemini")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("Gemini")
    or ""
)
default_eleven = (
    st.secrets.get("ELEVEN_API_KEY")
    or st.secrets.get("Eleven")
    or os.getenv("ELEVEN_API_KEY")
    or os.getenv("Eleven")
    or ""
)

with st.sidebar:
    st.header("⚙️ Keys & Settings")
    api_key = st.text_input("Gemini API Key", value=default_gemini, type="password")
    eleven_key = st.text_input("ElevenLabs API Key", value=default_eleven, type="password")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("👈 Please enter your Gemini API Key in the sidebar or under Streamlit Cloud Secrets.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# Working models verified for Gemini 3.5 / 3.8
PREFERRED_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest"
]

# Audio synthesis function (ElevenLabs high quality with gTTS fallback)
def generate_speech(text: str) -> bytes:
    if eleven_key:
        try:
            req_data = json.dumps({"text": text}).encode("utf-8")
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL",
                data=req_data,
                headers={
                    "xi-api-key": eleven_key,
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read()
        except Exception:
            pass

    # Fallback to gTTS
    buf = io.BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(buf)
    return buf.getvalue()

# Maintain message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# Voice & Text inputs
st.write("🎙️ **Record Voice:**")
audio_record = st.audio_input("Speak to Nova", key="audio_in")
text_record = st.chat_input("Or type here...")

user_text = None
if audio_record:
    audio_id = f"{audio_record.name}_{audio_record.size}"
    if st.session_state.get("last_audio_id") != audio_id:
        st.session_state["last_audio_id"] = audio_id
        with st.spinner("🎧 Transcribing voice..."):
            audio_bytes = audio_record.getvalue()
            mime = audio_record.type or "audio/wav"
            for m in PREFERRED_MODELS:
                try:
                    res = client.models.generate_content(
                        model=m,
                        contents=[
                            genai.types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                            "Transcribe this spoken audio verbatim. Output only the spoken words, nothing else."
                        ]
                    )
                    if res.text:
                        user_text = res.text.strip()
                        break
                except Exception:
                    continue

elif text_record:
    user_text = text_record

if user_text:
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Nova responds
    with st.chat_message("assistant"):
        response_box = st.empty()
        response_box.markdown("*(thinking...)*")

        prompt = (
            "You are Nova, a friendly, warm, emotionally intelligent AI companion. "
            "Keep your reply natural, supportive, and concise (1-3 sentences).\n\n"
            f"User: {user_text}"
        )

        bot_reply = None
        last_error = None

        for m in PREFERRED_MODELS:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=prompt
                )
                if resp.text:
                    bot_reply = resp.text.strip()
                    break
            except Exception as e:
                last_error = e
                continue

        if bot_reply:
            response_box.write(bot_reply)

            # Generate and play realistic voice
            audio_bytes = None
            try:
                audio_bytes = generate_speech(bot_reply)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception:
                pass

            msg_payload = {"role": "assistant", "content": bot_reply}
            if audio_bytes:
                msg_payload["audio"] = audio_bytes
            st.session_state.messages.append(msg_payload)
        else:
            response_box.error(f"Error from Gemini: {last_error}")
