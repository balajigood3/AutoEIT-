import librosa
import numpy as np

def pronunciation_score(audio_path):
    y, sr = librosa.load(audio_path)

    duration = librosa.get_duration(y=y, sr=sr)
    energy = np.mean(librosa.feature.rms(y=y))

    score = min(1.0, energy * 10)

    return {
        "duration": duration,
        "energy": float(energy),
        "pronunciation_score": score
    }