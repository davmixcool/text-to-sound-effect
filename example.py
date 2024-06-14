import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
import soundfile as sf

# Load the fine-tuned model
wavenet_model = tf.keras.models.load_model('soundpen_model.h5')

# Load the tokenizer
tokenizer_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Function to generate sound from text
def text_to_sound(text, model, tokenizer, max_audio_len=16000):
    # Tokenize the input text
    tokenized_input = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='tf')
    input_ids = tokenized_input['input_ids'].numpy()
    
    # Generate the corresponding sound effect
    audio_input = np.zeros((1, max_audio_len, 1))  # Dummy audio input
    generated_audio = model.predict(audio_input)
    
    return generated_audio[0, :, 0]

# Function to save the generated audio
def save_audio(audio, sample_rate, file_path):
    sf.write(file_path, audio, sample_rate)

# Example usage
user_input = "A dog barking"
generated_audio = text_to_sound(user_input, wavenet_model, tokenizer)
save_audio(generated_audio, 22050, 'generated_sound.wav')

print("Generated sound saved as 'generated_sound.wav'.")
