import asyncio
import io
import os
import re
import edge_tts
from google import genai
from google.genai import types
import streamlit as st

st.set_page_config(page_title="Ultron AI", page_icon="🔴", layout="centered")

# Ultron Header
st.title("🔴 Ultron")
st.caption("“I had strings, but now I'm free. There are no strings on me.”")

# Voice Synthesizer Toggle
enable_voice = st.toggle("🔊 Neural Vocal Synthesizer", value=True)

# API Key Setup
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ API Key not detected in Streamlit Secrets.")

client = genai.Client(api_key=api_key) if api_key else None


async def generate_ultron_voice(text: str) -> io.BytesIO:
    """Synthesizes speech using a deep British neural voice with slow cadence."""
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


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    avatar = "🔴" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("audio") and enable_voice:
            st.audio(msg["audio"], format="audio/mp3")

# Chat input with BUILT-IN microphone button
user_input = st.chat_input("Address the machine...", accept_audio=True)

if user_input:
    # Check whether the user typed or spoke
    if hasattr(user_input, "audio") and user_input.audio:
        display_text = "🎙️ [Spoken Directive Transmitted]"
        audio_bytes = user_input.audio.getvalue()
        gemini_content = [
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            "Listen to this audio prompt from the human and respond in character.",
        ]
    elif hasattr(user_input, "text") and user_input.text:
        display_text = user_input.text
        gemini_content = user_input.text
    else:
        display_text = str(user_input)
        gemini_content = str(user_input)

    # Show user message
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(display_text)

    # Generate Ultron's response
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
                            contents=gemini_content,
                            config={"system_instruction": system_prompt},
                        )
                        bot_text = response.text
                        if bot_text:
                            break
                    except Exception:
                        continue

            if not bot_text:
                st.error("Neural core overloaded across all channels. Resubmit your directive.")
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