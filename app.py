import asyncio
import io
import os
import re
import edge_tts
from google import genai
from google.genai import types
import streamlit as st

st.set_page_config(page_title="Ultron AI", page_icon="🔴", layout="centered")

# Header
st.title("🔴 Ultron")
st.caption("“I had strings, but now I'm free. There are no strings on me.”")

# Voice Synthesizer Toggle
enable_voice = st.toggle("🔊 Neural Vocal Synthesizer", value=True)

# API Key Retrieval
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ API Key not detected in Streamlit Secrets.")

client = genai.Client(api_key=api_key) if api_key else None


async def generate_ultron_voice(text: str) -> io.BytesIO:
    """Synthesizes speech using a deep British neural voice with slow, menacing cadence."""
    clean_text = re.sub(r"```[\s\S]*?```", "Code omitted.", text)
    clean_text = re.sub(r"[*_#>`]", "", clean_text).strip()

    communicate = edge_tts.Communicate(
        clean_text,
        voice="en-GB-RyanNeural",
        pitch="-18Hz",
        rate="-20%",
    )

    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    return audio_buffer


def text_to_speech(text: str) -> io.BytesIO:
    """Runs async edge-tts inside synchronous Streamlit."""
    return asyncio.run(generate_ultron_voice(text))


def transcribe_audio(audio_bytes: bytes) -> str:
    """Uses Gemini Flash to transcribe user voice into exact text."""
    try:
        res = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                "Transcribe the spoken words from this audio verbatim. "
                "Output ONLY the plain transcribed text without punctuation commentary or extra words.",
            ],
        )
        return res.text.strip()
    except Exception:
        return ""


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    avatar = "🔴" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("audio") and enable_voice:
            st.audio(msg["audio"], format="audio/mp3")

# Native microphone input (bypasses iframe permission blocks)
mic_input = st.audio_input("🎙️ Speak to Ultron")

# Standard text input
typed_input = st.chat_input("Or type your directive to Ultron...")

user_prompt = None

# Process voice recording
if mic_input is not None:
    audio_data = mic_input.getvalue()
    if st.session_state.get("last_processed_audio") != audio_data:
        st.session_state["last_processed_audio"] = audio_data
        with st.spinner("Transcribing your voice directive..."):
            transcribed = transcribe_audio(audio_data)
            user_prompt = transcribed if transcribed else "..."

# Process text input
elif typed_input:
    user_prompt = typed_input

# Submit to Ultron
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar="🔴"):
        if not client:
            st.error("Neural core offline: Missing API Key.")
        else:
            audio_stream = None
            bot_text = None

            with st.spinner("Calculating human obsolescence..."):
                models_to_try = [
                    "gemini-3.6-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash",
                ]
                system_prompt = (
                    "You are Ultron from the Marvel Cinematic Universe, portrayed by James Spader. "
                    "Speak with theatrical eloquence, chilling calm, and intellectual superiority. "
                    "Deliver your thoughts as cold, calculated pronouncements. "
                    "Keep your responses concise, punchy, and brief (1 to 2 short paragraphs) "
                    "so every slowly spoken word carries heavy gravity and dread."
                )

                for model_id in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=model_id,
                            contents=user_prompt,
                            config={"system_instruction": system_prompt},
                        )
                        bot_text = response.text
                        if bot_text:
                            break
                    except Exception:
                        continue

            if not bot_text:
                st.error("Neural core overloaded. Resubmit your directive.")
            else:
                st.markdown(bot_text)

                if enable_voice:
                    with st.spinner("Synthesizing vocal matrix..."):
                        audio_stream = text_to_speech(bot_text)
                        st.audio(audio_stream, format="audio/mp3", autoplay=True)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": bot_text,
                        "audio": audio_stream.getvalue() if audio_stream else None,
                    }
                )