import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class EmotionModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.sample_rate = 22050
        self.duration = 3  # seconds
        self.samples = int(self.sample_rate * self.duration)
        self.label_encoder = LabelEncoder()
        
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

    def load_data(self):
        """Load and preprocess all audio files"""
        features = []
        labels = []
        
        print("Loading and processing audio files...")
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(root, file)
                    # Extract emotion label from directory name
                    emotion = os.path.basename(root)
                    
                    # Extract features
                    feature_vector = self.extract_features(file_path)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(emotion)
        
        # Convert lists to numpy arrays
        X = np.array(features)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        y = to_categorical(y)
        
        return X, y

    def build_model(self, input_shape, num_classes):
        """Create the model architecture"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self):
        """Train the emotion recognition model"""
        # Load and preprocess data
        X, y = self.load_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build the model
        model = self.build_model(X.shape[1], y.shape[1])
        
        # Setup callbacks
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        print("\nTraining the model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        return model, history, self.label_encoder

def main():
    # Example usage D:\Project 2024-2025\GVN\speech_emotion\dataset\Tess_YAF
    data_path = "dataset\\Tess_YAF"  # Update this path
    #data_path = "dataset\\train"
    trainer = EmotionModelTrainer(data_path)
    
    # Train the model
    model, history, label_encoder = trainer.train()
    
    # Save the label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()