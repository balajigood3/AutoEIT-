import io
import numpy as np
import librosa
import whisper
from fastapi import FastAPI, UploadFile, File, Form
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="AutoEIT++ Speech Evaluation API")

# Load Models (only once)
whisper_model = whisper.load_model("base")
st_model = SentenceTransformer('all-MiniLM-L6-v2')


# SOUND DETECTION
def identify_sound_type(y, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    if np.max(np.abs(y)) < 0.001:
        return "Silence", "Audio too quiet."

    if chroma > 0.4 and zcr < 0.12:
        return "Music", "Detected music, not speech."

    if 0.12 < zcr < 0.3 or (1500 < centroid < 3000):
        return "Speech", "Speech detected."

    return "Noise", "Non-speech sound detected."


# 🔹 GRAMMAR SCORE
def grammar_score(text):
    if not text:
        return 0.0
    blob = TextBlob(text)
    corrected = str(blob.correct())

    # similarity between original & corrected
    return round(1 - (len(corrected) - len(text)) / max(len(text), 1), 2)


# 🔹 SYNTAX SCORE
def syntax_score(text):
    words = text.split()
    if len(words) < 3:
        return 0.5  # penalize short sentences

    # simple heuristic
    return min(1.0, len(words) / 10)


# 🔹 SEMANTIC SCORE
def semantic_score(ref, hyp):
    if not hyp:
        return 0.0

    ref_emb = st_model.encode(ref, convert_to_tensor=True)
    hyp_emb = st_model.encode(hyp, convert_to_tensor=True)

    score = float(util.pytorch_cos_sim(ref_emb, hyp_emb))
    return max(0, round(score, 2))


# 🔹 FEEDBACK ENGINE (IMPORTANT)
def generate_feedback(sem, syn, gram, text):
    feedback = []

    if len(text.split()) < 3:
        feedback.append("Speak a full sentence for better evaluation.")

    if sem < 0.4:
        feedback.append("Meaning is unclear or different from reference.")

    if syn < 0.6:
        feedback.append("Sentence structure can be improved.")

    if gram < 0.7:
        feedback.append("Grammar needs correction.")

    if not feedback:
        return "Excellent! Clear speech with good structure."

    return " ".join(feedback)


# 🔹 MAIN API
@app.post("/evaluate/")
async def evaluate(
    audio: UploadFile = File(...),
    reference_text: str = Form("she goes to school")
):
    try:
        audio_bytes = await audio.read()

        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # Detect sound
        sound_label, sound_feedback = identify_sound_type(y, sr)

        # Transcribe
        result = whisper_model.transcribe(y, fp16=False)
        hyp = result.get("text", "").strip()

        # If not speech → return early
        if sound_label != "Speech":
            return {
                "transcription": hyp,
                "identified_sound": sound_label,
                "score": {
                    "semantic": 0.0,
                    "syntax": 0.0,
                    "grammar": 0.0,
                    "final": 0.0
                },
                "feedback": sound_feedback
            }

        # Scores
        sem = semantic_score(reference_text, hyp)
        syn = syntax_score(hyp)
        gram = grammar_score(hyp)

        final = round((sem * 0.5 + syn * 0.25 + gram * 0.25), 2)

        feedback = generate_feedback(sem, syn, gram, hyp)

        return {
            "transcription": hyp,
            "identified_sound": sound_label,
            "score": {
                "semantic": sem,
                "syntax": syn,
                "grammar": gram,
                "final": final
            },
            "feedback": feedback
        }

    except Exception as e:
        return {"error": str(e)}