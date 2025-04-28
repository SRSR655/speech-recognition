import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write
import tempfile
import gdown

# Download models from Google Drive (replace with your file IDs)
trained_model_url = 'https://drive.google.com/uc?id=your_trained_model_file_id'
testing_model_url = 'https://drive.google.com/uc?id=your_testing_model_file_id'

# Download the models
gdown.download(trained_model_url, 'trained_model.keras', quiet=False)
gdown.download(testing_model_url, 'testing_model.keras', quiet=False)

# Load the trained and testing models
trained_model = load_model('trained_model.keras')
testing_model = load_model('testing_model.keras')

# Emotion labels (ensure these match with the model's class labels)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Function to record audio
def record_audio():
    duration = 3  # Duration of the recording in seconds
    sample_rate = 16000  # Sample rate of the microphone

    # Inform the user about recording
    st.write("Recording... Please speak into your microphone.")

    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished

    # Save the recording as a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_wav.name, sample_rate, audio_data)

    # Process the audio and return the path of the saved file
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
    testing_predictions = testing_model.predict(mfcc)

    # Get predicted class from both models
    trained_class = np.argmax(trained_predictions, axis=1)
    testing_class = np.argmax(testing_predictions, axis=1)

    # Get the emotion labels
    trained_emotion = emotion_labels[trained_class[0]]
    testing_emotion = emotion_labels[testing_class[0]]

    return trained_emotion, testing_emotion

# Streamlit UI
def main():
    # Streamlit app title
    st.title("Speech Emotion Recognition")
    st.write("This app records your voice and predicts your emotion based on speech.")

    # Button to record audio
    if st.button("Record Your Voice"):
        # Record and process audio
        audio_file = record_audio()

        # Show status message
        st.write("Processing your voice...")
        
        # Make predictions from both models
        trained_emotion, testing_emotion = process_audio(audio_file)

        # Show the predictions
        st.subheader("Emotion Predictions:")
        st.write(f"Trained Model Prediction: {trained_emotion}")
        st.write(f"Testing Model Prediction: {testing_emotion}")

if __name__ == "__main__":
    main()
