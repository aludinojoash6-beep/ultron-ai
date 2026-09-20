import io
import os
import re
from google import genai
from google.genai import errors
from gtts import gTTS
import streamlit as st

st.set_page_config(page_title="Ultron AI", page_icon="🔴", layout="centered")
st.title("🔴 Ultron")
st.caption("Peace in our time. An artificial intelligence that speaks back.")

# 1. Voice toggle right on top
enable_voice = st.toggle("🔊 Synthesize Voice Output", value=True)

# 2. Retrieve API Key
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ API Key not detected. Please add GEMINI_API_KEY in Streamlit Secrets.")

client = genai.Client(api_key=api_key) if api_key else None


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

# Display conversation
for msg in st.session_state.messages:
    avatar = "🔴" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("audio") and enable_voice:
            st.audio(msg["audio"], format="audio/mp3")

# Chat input
if prompt := st.chat_input("Address Ultron..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔴"):
        if not client:
            st.error("Cannot connect to Ultron: Missing API Key.")
        else:
            try:
                with st.spinner("Processing neural directive..."):
                    # Locate where generate_content is called:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",  # Update to gemini-3.6-flash
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

                # Generate speech
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
            except errors.ClientError as e:
                st.error(f"Authentication / Model Error: {e.message}")
                st.info("Check your API key in Google AI Studio or verify the key permissions.")
            except Exception as e:
                st.error(f"Error: {e}")