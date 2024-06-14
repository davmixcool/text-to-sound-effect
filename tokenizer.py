import os
import json
from transformers import AutoTokenizer

# Directory containing the text descriptions
external_drive_path = '/Volumes/Encrypt/soundpen'
descriptions_file =  os.path.join(external_drive_path,'sounds/descriptions.json')  # JSON file containing descriptions
tokenized_output_file = os.path.join(external_drive_path,'sounds/tokenized_descriptions.json')  # Output file for tokenized descriptions

# Load the tokenizer
tokenizer_name = 'bert-base-uncased'  # You can use any tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Load text descriptions
with open(descriptions_file, 'r') as f:
    descriptions = json.load(f)

# Tokenize descriptions
tokenized_descriptions = {}
for sound_id, description in descriptions.items():
    print(f'Tokenizing sound effect: {sound_id}:{description}')
    tokenized_description = tokenizer(description, truncation=True, padding='max_length', max_length=128)
    # Convert to dictionary
    tokenized_descriptions[sound_id] = dict(tokenized_description)

# Save tokenized descriptions to a JSON file
with open(tokenized_output_file, 'w') as f:
    json.dump(tokenized_descriptions, f)

print(f'Tokenized descriptions saved to: {tokenized_output_file}')