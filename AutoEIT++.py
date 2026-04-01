import streamlit as st
import io
import numpy as np
import librosa
import whisper
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIG & MODELS ---
st.set_page_config(page_title="AutoEIT++", layout="centered")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return whisper_model, st_model

with st.spinner("Loading AI Results... please wait."):
    whisper_model, st_model = load_models()

# --- 2. LOGIC FUNCTIONS ---
def identify_sound_type(y, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    if np.max(np.abs(y)) < 0.001: return "Silence", "Audio too quiet."
    if chroma > 0.4 and zcr < 0.12: return "Music", "Detected music."
    if 0.12 < zcr < 0.3 or (1500 < centroid < 3000): return "Speech", "Speech detected."
    return "Noise", "Non-speech sound."

def get_scores(text, ref="she goes to school"):
    if not text: return 0.0, 0.0, 0.0
    blob = TextBlob(text)
    gram = round(1 - (len(str(blob.correct())) - len(text)) / max(len(text), 1), 2)
    syn = min(1.0, len(text.split()) / 10)
    ref_emb = st_model.encode(ref, convert_to_tensor=True)
    hyp_emb = st_model.encode(text, convert_to_tensor=True)
    sem = max(0, round(float(util.pytorch_cos_sim(ref_emb, hyp_emb)), 2))
    return sem, syn, gram

def generate_feedback(sem, syn, gram, text):
    feedback = []
    if len(text.split()) < 3: feedback.append("Speak a full sentence.")
    if sem < 0.4: feedback.append("Meaning is unclear.")
    if syn < 0.6: feedback.append("Structure can be improved.")
    if gram < 0.7: feedback.append("Grammar needs correction.")
    return " ".join(feedback) if feedback else "Excellent! Clear speech."

# --- 3. UI LAYOUT ---
st.title("🎤 AutoEIT++ Language Evaluator")
st.markdown("AI-powered Speech Evaluation System")

audio_input = st.audio_input("Record your voice")
file_input = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a"])

audio_data = audio_input if audio_input else file_input

if audio_data:
    try:
        with st.spinner("Analyzing..."):
            # Load Audio
            y, sr = librosa.load(io.BytesIO(audio_data.getvalue()), sr=16000)
            label, msg = identify_sound_type(y, sr)
            
            if label == "Speech":
                # Process Transcription
                result = whisper_model.transcribe(y, fp16=False)
                hyp = result.get("text", "").strip()
                
                # Get Scores
                sem, syn, gram = get_scores(hyp)
                final = round((sem * 0.5 + syn * 0.25 + gram * 0.25), 2)
                feedback = generate_feedback(sem, syn, gram, hyp)

                # --- DISPLAY RESULTS ---
                st.success("✅ Evaluation Complete")
                st.subheader("🎤 Transcription")
                st.info(hyp)

                st.subheader("📊 Scores")
                st.progress(final)
                st.metric("Semantic", sem)
                st.metric("Syntax", syn)
                st.metric("Grammar", gram)
                st.metric("Final Score", final)

                st.subheader("💡 Feedback")
                st.write(feedback)
            else:
                st.warning(f"Detection Result: {label} - {msg}")
                
    except Exception as e:
        st.error(f"Error processing audio: {e}")
