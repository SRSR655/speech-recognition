import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile
import wave
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Load model directly from file path
MODEL_PATH = "trainedmodel.keras"  # Make sure this file is in your project folder
trained_model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Audio processor class
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame)
        return frame

# Save to temporary WAV file
def save_temp_wav(frames, sample_rate=16000):
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    samples = b"".join([f.to_ndarray().tobytes() for f in frames])
    with wave.open(wav_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return wav_file.name

# Predict emotion
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    predictions = trained_model.predict(mfcc)
    predicted_class = np.argmax(predictions, axis=1)
    return emotion_labels[predicted_class[0]]

# Streamlit UI
st.title("🎤 Real-Time Speech Emotion Recognition")
st.write("Speak into the mic and click the button to predict your emotion.")

# Start the microphone streamer
ctx = webrtc_streamer(
    key="audio",
    mode="sendonly",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

if ctx.audio_processor and ctx.audio_processor.frames:
    st.info("Recording... Speak now.")

    if st.button("🔍 Predict Emotion"):
        audio_path = save_temp_wav(ctx.audio_processor.frames)
        emotion = process_audio(audio_path)
        st.success(f"🎯 Predicted Emotion: **{emotion}**")


