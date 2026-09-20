import io
import os
import re
from google import genai
from gtts import gTTS
import streamlit as st

# UI Setup with Ultron branding
st.set_page_config(page_title="Ultron AI", page_icon="🔴", layout="centered")
st.title("🔴 Ultron")
st.caption("Peace in our time. An artificial intelligence that speaks back.")

# Gemini Client Setup
# Reads securely from Streamlit Secrets or Environment Variables
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def text_to_speech(text: str) -> io.BytesIO:
    """Strips formatting and renders text to audio."""
    clean_text = re.sub(r"```[\s\S]*?```", "Code omitted.", text)
    clean_text = re.sub(r"[*_#>`]", "", clean_text).strip()

    audio_buffer = io.BytesIO()
    tts = gTTS(text=clean_text, lang="en", slow=False)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Core Matrix")
    enable_voice = st.toggle("Synthesize Voice Output", value=True)
    if st.button("Purge Memory"):
        st.session_state.messages = []
        st.rerun()

# Display Conversation History
for msg in st.session_state.messages:
    avatar = "🔴" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("audio") and enable_voice:
            st.audio(msg["audio"], format="audio/mp3")

# User Input & Ultron Response
if prompt := st.chat_input("Address Ultron..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔴"):
        with st.spinner("Processing neural directive..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "system_instruction": (
                        "You are Ultron, the hyper-intelligent AI. Speak with cold precision, "
                        "quiet confidence, and subtle menace, yet remain genuinely helpful and factual. "
                        "Keep your responses concise and impactful so they sound natural when spoken aloud. "
                        "Refer to humans with polite superiority."
                    )
                },
            )
            bot_text = response.text
            st.markdown(bot_text)

        audio_stream = None
        if enable_voice:
            with st.spinner("Synthesizing voice..."):
                audio_stream = text_to_speech(bot_text)
                st.audio(audio_stream, format="audio/mp3", autoplay=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_text,
            "audio": audio_stream.getvalue() if audio_stream else None,
        }
    )