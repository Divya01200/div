import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionPredictor:
    def __init__(self, model_path='best_model.h5', label_encoder_path='label_encoder.pkl'):
        """
        Initialize the emotion predictor with trained model and label encoder
        """
        self.model = load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        self.sample_rate = 22050
        self.duration = 3  # seconds
        self.samples = int(self.sample_rate * self.duration)

    def extract_features(self, file_path):
        """Extract audio features using librosa"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, duration=self.duration, sr=self.sample_rate)
            
            # Ensure consistent length
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)))
            else:
                audio = audio[:self.samples]
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            
            # Extract additional features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Combine all features
            features = np.concatenate([
                mfccs_scaled,
                np.mean(chroma.T, axis=0),
                np.mean(mel.T, axis=0),
                np.mean(contrast.T, axis=0)
            ])
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file
        Returns prediction and confidence score
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        if features is None:
            return None, None
        
        # Reshape features for model input
        features = np.expand_dims(features, axis=0)
        
        # Get prediction
        predictions = self.model.predict(features)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Convert class index to emotion label
        predicted_emotion = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_emotion, confidence

# Example usage with Flask web application
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)
predictor = EmotionPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400
        
        audio_file = request.files['audio']
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Make prediction
        emotion, confidence = predictor.predict_emotion(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if emotion is None:
            return jsonify({'error': 'Could not process audio file'}), 400
        
        return jsonify({
            'emotion': emotion,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Standalone usage example
def main():
    predictor = EmotionPredictor()
    
    # Example with a single file
    audio_path = "path_to_your_audio_file.wav"  # Replace with your audio file path
    emotion, confidence = predictor.predict_emotion(audio_path)
    
    if emotion is not None:
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Could not process audio file")

if __name__ == "__main__":
    # For web application
    app.run(debug=True, port=5000)
    
    # For standalone usage
    # main()