import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile
import wave
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Custom audio processor to collect frames
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame)
        return frame

# Save recorded audio as a WAV file
def save_temp_wav(frames, sample_rate=16000):
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    samples = b"".join([f.to_ndarray().tobytes() for f in frames])
    with wave.open(wav_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return wav_file.name

# Preprocess and predict emotion
def process_audio(file_path, model):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(mfcc)
    predicted_class = np.argmax(predictions, axis=1)
    return emotion_labels[predicted_class[0]]

# Streamlit UI
st.title("🎤 Live Speech Emotion Recognition")
st.write("Speak into the mic. Get your predicted emotion.")

uploaded_model_file = st.file_uploader("Upload your trained Keras model (.keras)", type="keras")

model = None
if uploaded_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(uploaded_model_file.read())
        model = load_model(tmp.name)
        st.success("✅ Model loaded successfully!")

if model:
    # Start recording
    ctx = webrtc_streamer(
        key="audio",
        mode="sendonly",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
    )

    if ctx.audio_processor and ctx.audio_processor.frames:
        st.info("Recording... speak now!")

        if st.button("🔍 Analyze Emotion"):
            audio_path = save_temp_wav(ctx.audio_processor.frames)
            emotion = process_audio(audio_path, model)
            st.success(f"🎯 Predicted Emotion: **{emotion}**")

