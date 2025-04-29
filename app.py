import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile
import wave
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Function to save audio frames as a temporary WAV file
def save_temp_wav(frames, sample_rate=16000):
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    samples = b"".join([f.to_ndarray().tobytes() for f in frames])
    with wave.open(wav_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return wav_file.name

# Function to process the audio and make predictions
def process_audio(file_path, model):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(mfcc)
    predicted_class = np.argmax(predictions, axis=1)
    return emotion_labels[predicted_class[0]]

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition (Streamlit Cloud-Compatible)")
st.write("Speak into the mic below. Your emotion will be predicted!")

# Upload model file
uploaded_model = st.file_uploader("trainedmodel.keras", type="keras")

# Load model once uploaded
if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(uploaded_model.read())  # Save the uploaded file to a temporary location
        model = load_model(tmp.name)  # Load the model

    st.success("Model successfully uploaded and loaded!")

    # Start WebRTC for audio recording
    ctx = webrtc_streamer(
        key="mic",
        mode="sendonly",
        audio_processor_factory=AudioProcessorBase,
        media_stream_constraints={"video": False, "audio": True},
    )

    if ctx.audio_processor and ctx.audio_processor.frames:
        st.success("Recording complete! Click button below to analyze.")
        if st.button("Predict Emotion"):
            audio_path = save_temp_wav(ctx.audio_processor.frames)
            emotion = process_audio(audio_path, model)
            st.subheader("🎯 Predicted Emotion:")
            st.write(f"**{emotion}**")
