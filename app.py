import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile
import requests
import wave

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# URL to the raw model file on GitHub
github_model_url = "https://raw.githubusercontent.com/username/repository-name/main/path/to/your_model_file.keras"

# Function to download file from GitHub
def download_file_from_github(url, destination):
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)
        return destination
    else:
        raise Exception("Failed to retrieve file from GitHub")

# Function to load the model
def load_model_from_github(url):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        downloaded_file = download_file_from_github(url, tmp.name)
        return load_model(downloaded_file)

# Load the model (without caching)
trained_model = load_model_from_github(github_model_url)

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame)
        return frame

def save_temp_wav(frames, sample_rate=16000):
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    samples = b"".join([f.to_ndarray().tobytes() for f in frames])
    with wave.open(wav_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return wav_file.name

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    predictions = trained_model.predict(mfcc)
    predicted_class = np.argmax(predictions, axis=1)
    return emotion_labels[predicted_class[0]]

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition (Streamlit Cloud-Compatible)")
st.write("Speak into the mic below. Your emotion will be predicted!")

ctx = webrtc_streamer(
    key="mic",
    mode="sendonly",
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"video": False, "audio": True},
)

if ctx.audio_processor and ctx.audio_processor.frames:
    st.success("Recording complete! Click button below to analyze.")
    if st.button("Predict Emotion"):
        audio_path = save_temp_wav(ctx.audio_processor.frames)
        emotion = process_audio(audio_path)
        st.subheader("🎯 Predicted Emotion:")
        st.write(f"**{emotion}**")



