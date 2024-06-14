import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import librosa

# Directory containing preprocessed audio files and tokenized descriptions
external_drive_path = '/Volumes/Encrypt/soundpen'
audio_dir = os.path.join(external_drive_path,'sounds/processed')
tokenized_descriptions_file = os.path.join(external_drive_path,'sounds/tokenized_descriptions.json')

# Load tokenized descriptions
with open(tokenized_descriptions_file, 'r') as f:
    tokenized_descriptions = json.load(f)

# Function to load and preprocess audio
def load_audio(file_path, target_sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=target_sample_rate)
    return audio

# Load data
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
data = []
for file in audio_files:
    audio = load_audio(os.path.join(audio_dir, file))
    tokenized_description = tokenized_descriptions[os.path.splitext(file)[0]]
    data.append((audio, tokenized_description))

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare data function
def prepare_data(data, max_audio_len, max_desc_len):
    X_audio = np.zeros((len(data), max_audio_len, 1))
    X_desc = np.zeros((len(data), max_desc_len))
    Y = np.zeros((len(data), max_audio_len, 1))
    
    for i, (audio, tokenized_description) in enumerate(data):
        X_audio[i, :len(audio), 0] = audio
        X_desc[i, :len(tokenized_description['input_ids'])] = tokenized_description['input_ids']
        Y[i, :len(audio), 0] = audio
    
    return X_audio, X_desc, Y

# Determine max lengths
max_audio_len = max([len(audio) for audio, _ in data])
max_desc_len = max([len(desc['input_ids']) for _, desc in data])

# Prepare training and validation data
X_audio_train, X_desc_train, Y_train = prepare_data(train_data, max_audio_len, max_desc_len)
X_audio_val, X_desc_val, Y_val = prepare_data(val_data, max_audio_len, max_desc_len)

# Load the trained model
wavenet_model = tf.keras.models.load_model('soundpen_model.h5')

# Evaluate the model on the validation set
loss = wavenet_model.evaluate(X_audio_val, Y_val)

print(f'Validation Loss: {loss}')
