import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Directory containing the sound files
external_drive_path = '/Volumes/Encrypt/soundpen'
input_dir = os.path.join(external_drive_path,'sounds/effects')
output_dir = os.path.join(external_drive_path,'sounds/processed')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def normalize_audio(audio, target_db=-20.0):
    """
    Normalize the audio to a target dB.
    """
    # Compute the mean loudness in dB
    loudness = librosa.core.amplitude_to_db(np.abs(audio), ref=np.max)
    mean_loudness = np.mean(loudness)
    # Compute the required gain to achieve the target loudness
    gain = target_db - mean_loudness
    # Apply gain to audio
    audio_normalized = librosa.effects.preemphasis(audio, coef=gain)
    return audio_normalized

def preprocess_audio(file_path, target_sample_rate=22050):
    """
    Preprocess an audio file: load, normalize, trim silence, and resample.
    """
    # Convert MP3 to WAV if necessary
    if file_path.endswith('.mp3'):
        audio_segment = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace('.mp3', '.wav')
        audio_segment.export(wav_path, format='wav')
        file_path = wav_path
    
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Normalize the audio
    audio = normalize_audio(audio)
    
    # Trim silence from the beginning and end
    audio, _ = librosa.effects.trim(audio)
    
    # Resample to the target sample rate
    if sr != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
    
    return audio, target_sample_rate, file_path

def process_files(input_dir, output_dir, target_sample_rate=22050):
    """
    Process all audio files in the input directory and save them to the output directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp3'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.wav')
                
                # Check if the output file already exists
                if os.path.exists(output_file_path):
                    print(f'Skipping already processed file: {output_file_path}')
                    continue
                
                print(f'Processing: {input_file_path}')

                try:
                    # Preprocess the audio file
                    audio, sr, processed_file_path = preprocess_audio(input_file_path, target_sample_rate)
                    
                    # Save the processed audio
                    sf.write(output_file_path, audio, sr)
                    
                    print(f'Processed and saved: {output_file_path}')
                    
                    # Delete the processed file in the input directory if it was converted from mp3 to wav
                    if processed_file_path.endswith('.wav') and processed_file_path != input_file_path:
                        os.remove(processed_file_path)
                        print(f'Deleted temporary WAV file: {processed_file_path}')
                except Exception as e:
                    print(f'Error processing {input_file_path}: {e}')

# Run the preprocessing on the sound effects directory
process_files(input_dir, output_dir)
