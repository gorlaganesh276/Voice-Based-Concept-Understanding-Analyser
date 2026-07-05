import librosa, soundfile as sf

def preprocess_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    sf.write("normalized.wav", audio, target_sr)
    return "normalized.wav"
