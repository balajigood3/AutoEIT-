import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001/evaluate/"

st.set_page_config(page_title="AutoEIT++", layout="centered")

st.title("🎤 AutoEIT++ Language Evaluator")
st.markdown("AI-powered Speech Evaluation System")

# 🔹 FUNCTION
def process_audio(files):
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Evaluation Complete")

                # Transcription
                st.subheader("🎤 Transcription")
                st.info(result.get("transcription", "No speech"))

                # Scores
                scores = result.get("score", {})

                st.subheader("📊 Scores")
                st.progress(scores.get("final", 0.0))

                # Each line creates a new vertical row
                st.metric("Semantic", scores.get("semantic", 0))
                st.metric("Syntax", scores.get("syntax", 0))
                st.metric("Grammar", scores.get("grammar", 0))
                st.metric("Final Score", scores.get("final", 0))

                # Feedback
                st.subheader("💡 Feedback")
                st.write(result.get("feedback", "No feedback"))

            else:
                st.error("Backend Error")

        except Exception as e:
            st.error(f"Connection Error: {e}")


# 🔹 UI
tab1, tab2 = st.tabs(["Record", "Upload"])

with tab1:
    audio = st.audio_input("Record your voice")
    if audio and st.button("Evaluate Recording"):
        files = {"audio": ("live.wav", audio.getvalue(), "audio/wav")}
        process_audio(files)

with tab2:
    file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
    if file and st.button("Evaluate File"):
        files = {"audio": (file.name, file.getvalue(), file.type)}
        process_audio(files)