import csv
import json

external_drive_path = '/Volumes/Encrypt/soundpen'
# Path to your CSV file
csv_file_path = os.path.join(external_drive_path,'sounds/descriptions.csv')
# Path to your JSON file
json_file_path = os.path.join(external_drive_path,'sounds/descriptions.json')

# Initialize an empty dictionary
data_dict = {}

# Read the CSV file and populate the dictionary
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        uuid = row['uuid']
        name = row['name']
        data_dict[uuid] = name

# Save the dictionary to a JSON file
with open(json_file_path, mode='w', encoding='utf-8') as json_file:
    json.dump(data_dict, json_file, indent=4)

# Print a confirmation message
print(f"Data has been saved to {json_file_path}")
