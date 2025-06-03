import librosa
import numpy as np
from tensorflow.keras.models import load_model
import soundfile as sf
import os

class EmotionRecognizer:
    def __init__(self):
        """Initialize the emotion recognizer with pre-trained model"""

        self.model = load_model('path_to_model.h5')  
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def extract_features(self, audio_path):
        """Extract audio features using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=3, offset=0.5)
            
            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            
            # Extract additional features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Combine features
            features = np.concatenate([
                mfcc_scaled,
                np.mean(chroma.T, axis=0),
                np.mean(mel.T, axis=0),
                np.mean(contrast.T, axis=0)
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def predict_emotion(self, audio_path):
        """Predict emotion from audio file"""
        try:
            # Extract features
            features = self.extract_features(audio_path)
            if features is None:
                return None
            
            # Reshape features for model input
            features = np.expand_dims(features, axis=0)
            
            # Make prediction
            predictions = self.model.predict(features)
            predicted_emotion = self.emotions[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            
            return {
                'emotion': predicted_emotion,
                'confidence': float(confidence),
                'all_probabilities': {
                    emotion: float(prob) 
                    for emotion, prob in zip(self.emotions, predictions[0])
                }
            }
            
        except Exception as e:
            print(f"Error predicting emotion: {str(e)}")
            return None

    def process_audio_file(self, audio_path):
        """Process audio file and return emotion prediction"""
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None
            
        result = self.predict_emotion(audio_path)
        return result

def main():
    """Example usage of EmotionRecognizer"""
    recognizer = EmotionRecognizer()
    
    # Example audio file path
    audio_path = "path_to_audio_file.wav"
    
    # Get prediction
    result = recognizer.process_audio_file(audio_path)
    
    if result:
        print(f"Predicted Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nAll Probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"{emotion}: {prob:.2f}")

if __name__ == "__main__":
    main()