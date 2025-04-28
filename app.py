import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile
from streamlit_audio_recorder import audio_recorder

# Load the trained and testing models
trained_model = load_model('path_to_your_trained_model.keras')
#testing_model = load_model('path_to_your_testing_model.keras')

# Emotion labels (ensure these match with the model's class labels)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Function to save recorded bytes into a WAV file
def save_audio(audio_bytes):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_wav.write(audio_bytes)
    temp_wav.flush()
    return temp_wav.name

# Function to preprocess the audio and make predictions from both models
def process_audio(file_path):
    # Load the audio using librosa
    y, sr = librosa.load(file_path, sr=16000)

    # Extract MFCC features from the audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension

    # Reshape to match model's expected input shape
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension

    # Make predictions using both models
    trained_predictions = trained_model.predict(mfcc)
    #testing_predictions = testing_model.predict(mfcc)

    # Get predicted class from both models
    trained_class = np.argmax(trained_predictions, axis=1)
    #testing_class = np.argmax(testing_predictions, axis=1)

    # Get the emotion labels
    trained_emotion = emotion_labels[trained_class[0]]
    #testing_emotion = emotion_labels[testing_class[0]]

    return trained_emotion

# Streamlit UI
def main():
    # Streamlit app title
    st.title("Speech Emotion Recognition")
    st.write("This app records your voice and predicts your emotion based on speech.")

    # Record audio using browser
    audio_bytes = audio_recorder()

    # When user records
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.write("Processing your voice...")

        # Save bytes to a temp WAV file
        audio_file = save_audio(audio_bytes)

        # Make predictions from both models
        trained_emotion = process_audio(audio_file)

        # Show the predictions
        st.subheader("Emotion Predictions:")
        st.write(f"Trained Model Prediction: {trained_emotion}")
        #st.write(f"Testing Model Prediction: {testing_emotion}")

if __name__ == "__main__":
    main()

