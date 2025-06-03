import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf
import os

class SpeechEmotionRecognizer:
    def __init__(self):
        # Define emotion labels
        self.emotions = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }
        
        # Load the pre-trained model
        self.model = self.load_pretrained_model()
        
    def load_pretrained_model(self):
        """
        Load or download pre-trained model.
        For this example, we'll create a simple model architecture.
        In practice, you would load your trained model file.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        # In practice, you would load weights here:
        # model.load_weights('path_to_your_weights.h5')
        return model

    def extract_features(self, audio_path):
        """
        Extract audio features using librosa
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=3, offset=0.5)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Combine features
            features = np.concatenate([mfcc, mel_spec, chroma], axis=0)
            
            # Pad or truncate to ensure consistent size
            if features.shape[1] < 128:
                features = np.pad(features, ((0, 0), (0, 128 - features.shape[1])))
            else:
                features = features[:, :128]
            
            # Reshape for model input
            features = np.resize(features, (128, 128))
            features = np.expand_dims(features, axis=-1)
            features = np.expand_dims(features, axis=0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        if features is None:
            return "Error processing audio file"
        
        # Make prediction
        prediction = self.model.predict(features)
        predicted_emotion = self.emotions[np.argmax(prediction[0])]
        
        # Get confidence scores
        confidence_scores = {
            emotion: float(score) 
            for emotion, score in zip(self.emotions.values(), prediction[0])
        }
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence_scores': confidence_scores
        }

class AudioRecorder:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 3  # seconds
        
    def record_audio(self, output_path):
        """
        Record audio using soundfile
        Note: This is a placeholder. In practice, you would use 
        a library like pyaudio for real-time recording
        """
        print("Recording... (3 seconds)")
        # Placeholder for actual recording logic
        print("Recording complete!")

def main():
    # Initialize the recognizer
    recognizer = SpeechEmotionRecognizer()
    recorder = AudioRecorder()
    
    while True:
        print("\nSpeech Emotion Recognition System")
        print("1. Analyze audio file")
        print("2. Record and analyze")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            audio_path = input("Enter the path to audio file: ")
            if os.path.exists(audio_path):
                result = recognizer.predict_emotion(audio_path)
                print("\nResults:")
                print(f"Predicted Emotion: {result['predicted_emotion']}")
                print("\nConfidence Scores:")
                for emotion, score in result['confidence_scores'].items():
                    print(f"{emotion}: {score:.2f}")
            else:
                print("File not found!")
                
        elif choice == '2':
            output_path = "recorded_audio.wav"
            recorder.record_audio(output_path)
            result = recognizer.predict_emotion(output_path)
            print("\nResults:")
            print(f"Predicted Emotion: {result['predicted_emotion']}")
            print("\nConfidence Scores:")
            for emotion, score in result['confidence_scores'].items():
                print(f"{emotion}: {score:.2f}")
                
        elif choice == '3':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()