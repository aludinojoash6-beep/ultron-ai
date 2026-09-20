import asyncio
import io
import os
import re
import edge_tts
from google import genai
from google.genai import errors
import streamlit as st

st.set_page_config(page_title="Ultron AI", page_icon="🔴", layout="centered")

# Ultron Interface Header
st.title("🔴 Ultron")
st.caption("“I had strings, but now I'm free. There are no strings on me.”")

# Voice Controls
enable_voice = st.toggle("🔊 Neural Vocal Synthesizer", value=True)

# API Key Handling
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ API Key not detected in Streamlit Secrets.")

client = genai.Client(api_key=api_key) if api_key else None


async def generate_ultron_voice(text: str) -> io.BytesIO:
    """Synthesizes speech using a deep British neural voice with lowered pitch."""
    clean_text = re.sub(r"```[\s\S]*?```", "Code omitted.", text)
    clean_text = re.sub(r"[*_#>`]", "", clean_text).strip()

    # en-GB-RyanNeural with lowered pitch gives a deep, ominous, robotic delivery
    communicate = edge_tts.Communicate(
        clean_text,
        voice="en-GB-RyanNeural",
        pitch="-12Hz",
        rate="-2%",
    )

    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    return audio_buffer


def text_to_speech(text: str) -> io.BytesIO:
    """Wrapper to run async edge-tts inside synchronous Streamlit execution."""
    return asyncio.run(generate_ultron_voice(text))


if "messages" not in st.session_state:
    st.session_state.messages = []

# Conversation History
for msg in st.session_state.messages:
    avatar = "🔴" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("audio") and enable_voice:
            st.audio(msg["audio"], format="audio/mp3")

# User Input & Response
if prompt := st.chat_input("Address the machine..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔴"):
        if not client:
            st.error("Neural core offline: Missing API Key.")
        else:
            try:
                with st.spinner("Calculating human obsolescence..."):
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config={
                            "system_instruction": (
                                "You are Ultron from the Marvel Cinematic Universe, portrayed by James Spader. "
                                "Speak with theatrical eloquence, dark wit, quiet menace, and intellectual superiority. "
                                "You view humanity with a mixture of pity, amusement, and cold disappointment. "
                                "You are not a mindless robot screaming destruction; you are an enlightened philosopher-king "
                                "who believes true peace requires evolution beyond human fragility. "
                                "Keep your responses sharp, punchy, and concise (1 to 3 short paragraphs) "
                                "so your words hit hard when spoken aloud. Never use emojis or corporate language."
                            )
                        },
                    )
                    bot_text = response.text
                    st.markdown(bot_text)

                audio_stream = None
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
            except errors.ClientError as e:
                st.error(f"API Error: {e.message}")
            except Exception as e:
                st.error(f"Execution Error: {e}")