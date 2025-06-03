
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['EMOTION_IMAGES_FOLDER'] = 'static/emotion'  


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class EmotionPredictor:
    def __init__(self, model_path='best_model.h5', label_encoder_path='label_encoder.pkl'):
        self.model = load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        self.sample_rate = 22050
        self.duration = 3
        self.samples = int(self.sample_rate * self.duration)

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, duration=self.duration, sr=self.sample_rate)
            
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)))
            else:
                audio = audio[:self.samples]
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
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
        features = self.extract_features(audio_path)
        
        if features is None:
            return None, None
        
        features = np.expand_dims(features, axis=0)
        predictions = self.model.predict(features)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_emotion = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_emotion, float(confidence)


predictor = EmotionPredictor()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login',methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        print(user, pwd)

        if user == "admin" and pwd == "admin":
            print("inside if")
            msg = 'Logged in successfully !'
            return render_template('upload_folder.html', msg=msg)
        elif user == "user" and pwd == "user":
            print("inside if")
            msg = 'Logged in successfully !'
            return render_template('dashboard.html', msg=msg)

        else:
            msg = 'Incorrect email / password !'
    else:
        print("form invalid")
    return render_template('login.html', msg=msg)

@app.route('/upload_folder', methods=['POST','GET'])
def upload_folder():
    return render_template('upload_folder.html', msg="")

@app.route('/preprocessing', methods=['POST','GET'])
def preprocessing():
    return render_template('preprocess.html', msg="")    

@app.route('/training', methods=['POST','GET'])
def training():
    return render_template('training.html', msg="")   

@app.route('/performance', methods=['POST','GET'])
def performance():
    return render_template('performance.html', msg="")   

@app.route('/predict_audio', methods=['POST','GET'])
def predict_audio():
    return render_template('predictaudio.html', msg="")   

@app.route('/live_predict', methods=['POST','GET'])
def live_predict():
    return render_template('live_predict.html', msg="")  

@app.route('/audio_live')
def audio_live():
    return render_template('audio_live.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if audio_file:
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        try:
            emotion, confidence = predictor.predict_emotion(filepath)
            
            
            os.remove(filepath)
            
            if emotion is None:
                return jsonify({'error': 'Could not process audio file'}), 400
             
            cleaned_emotion = emotion.replace('YAF_', '').lower()

            
            image_filename = f"{cleaned_emotion}.jpg"
            image_path = os.path.join(app.config['EMOTION_IMAGES_FOLDER'], image_filename)

            if not os.path.exists(image_path):
                image_filename = "default.jpg"  
            
            return jsonify({
                'emotion': emotion.replace('YAF_', ' '),
                'confidence': confidence,
                'image_url': f"/static/emotion/{image_filename}"
            })
            
        except Exception as e:
            
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)