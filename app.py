import streamlit as st
import whisper
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Voice-Based Concept Analyzer", layout="centered")
st.title("🎙️ Voice-Based Concept Understanding Analyzer")

# 1. Cache Models to avoid reloading on every user action
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return whisper_model, sbert_model

whisper_model, sbert_model = load_models()

# 2. UI Layout for Inputs
st.header("1. Upload and Configure")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
reference_text = st.text_input("Enter reference text/concept to compare similarity against:", "Hello world")

if uploaded_file is not None:
    # Save uploaded file temporarily for libraries to read
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(uploaded_file, format="audio/wav")

    # Trigger Analysis
    if st.button("Run Full Analysis"):
        with st.spinner("Processing audio, please wait..."):
            
            # --- SECTION 1: Audio Feature Extraction (Librosa) ---
            st.header("📊 Audio Feature Extraction")
            y, sr = librosa.load("temp_audio.wav")
            zcr = librosa.feature.zero_crossing_rate(y)
            mean_zcr = np.mean(zcr)
            st.metric(label="Mean Zero Crossing Rate (ZCR)", value=f"{mean_zcr:.4f}")
            
            # --- SECTION 2: Speech-to-Text (Whisper) ---
            st.header("📝 Audio Transcription")
            result = whisper_model.transcribe("temp_audio.wav")
            transcribed_text = result["text"]
            st.text_area("Transcribed Text:", transcribed_text, height=100)
            
            # --- SECTION 3: Semantic Similarity (Sentence-BERT) ---
            st.header("🧠 Semantic Similarity Analysis")
            # Generate embeddings
            embeddings1 = sbert_model.encode(transcribed_text, convert_to_tensor=True)
            embeddings2 = sbert_model.encode(reference_text, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity_score = util.cos_sim(embeddings1, embeddings2).item()
            
            st.success("✅ Semantic embeddings generated successfully!")
            st.metric(label="Similarity Score to Reference Text", value=f"{similarity_score * 100:.2f}%")