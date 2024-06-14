import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Add, Activation, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import librosa
import psutil
from tqdm import tqdm
import gc

# Update these paths to point to your external hard drive
external_drive_path = '/Volumes/Encrypt/soundpen'
audio_dir = os.path.join(external_drive_path, 'sounds/processed')
tokenized_descriptions_file = os.path.join(external_drive_path, 'sounds/tokenized_descriptions.json')

# Load tokenized descriptions
print("Loading tokenized descriptions...")
with open(tokenized_descriptions_file, 'r') as f:
    tokenized_descriptions = json.load(f)

# Function to load and preprocess audio
def load_audio(file_path, target_sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=target_sample_rate)
    return audio

# Define the WaveNet model
def residual_block(x, dilation_rate, filters, kernel_size):
    original_x = x
    conv_x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    tanh_out = Activation('tanh')(conv_x)
    sigmoid_out = Activation('sigmoid')(conv_x)
    merged = Multiply()([tanh_out, sigmoid_out])
    skip_out = Conv1D(filters, 1)(merged)
    res_out = Add()([skip_out, original_x])
    return res_out, skip_out

def build_wavenet_model(input_shape, num_blocks=1, num_layers=2, filters=8, kernel_size=2):
    inputs = Input(shape=input_shape)
    x = inputs
    skip_connections = []

    for b in range(num_blocks):
        for i in range(num_layers):
            x, skip = residual_block(x, dilation_rate=2**i, filters=filters, kernel_size=kernel_size)
            skip_connections.append(skip)
    x = Add()(skip_connections)
    x = Activation('relu')(x)
    x = Conv1D(filters, 1, activation='relu')(x)
    outputs = Conv1D(1, 1)(x)
    model = Model(inputs, outputs)
    return model

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

# Load data
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

# Use a smaller subset of the data for testing
audio_files = audio_files[:1000]


# Total samples for full dataset
total_samples = len(audio_files)
# Batch size
batch_size = 1
# Steps per epoch for full dataset
steps_per_epoch = total_samples / batch_size

print(f'Total samples: {total_samples}')
print(f'Steps per epoch: {steps_per_epoch}')

# Determine the maximum lengths
max_audio_len = 0
for file in tqdm(audio_files, desc="Calculating audio lengths"):
    audio_path = os.path.join(audio_dir, file)
    audio_len = librosa.get_duration(path=audio_path) * 22050
    max_audio_len = max(max_audio_len, audio_len)

max_audio_len = int(max_audio_len)
max_desc_len = max(len(desc['input_ids']) for desc in tokenized_descriptions.values())

# Data loading function
def load_data(file):
    file_id = os.path.splitext(file)[0]
    audio_path = os.path.join(audio_dir, file)
    audio, _ = librosa.load(audio_path, sr=22050)
    tokenized_description = tokenized_descriptions[file_id]['input_ids']
    audio = pad_sequences([audio], maxlen=max_audio_len, dtype='float32', padding='post', truncating='post')[0]
    tokenized_description = pad_sequences([tokenized_description], maxlen=max_desc_len, padding='post', truncating='post')[0]
    return np.expand_dims(audio, axis=-1), audio

# Create TensorFlow dataset
def data_generator():
    for file in audio_files:
        yield load_data(file)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(max_audio_len, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(max_audio_len,), dtype=tf.float32)
    )
).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

print(f'Building and compiling the model...')
print_memory_usage()

# Build and compile the model
input_shape = (max_audio_len, 1)
wavenet_model = build_wavenet_model(input_shape)
wavenet_model.compile(optimizer='adam', loss='mse')

print(f'Training the model...')
print_memory_usage()

# Train the model
wavenet_model.fit(dataset, epochs=10, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print_memory_usage())])

# Save the model to the external drive
model_save_path = os.path.join(external_drive_path, 'soundpen_model.h5')
wavenet_model.save(model_save_path)

print("Training complete and model saved.")
print_memory_usage()