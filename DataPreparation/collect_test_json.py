import os
import json
import csv

# Path to the directory containing JSON files
json_dir = 'src/DAC/quality_captions'

# Output CSV file path
output_csv = 'test_1000.csv'

def process_json_files(json_dir, output_csv):
    # Collect JSON file names in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # Initialize a list to store rows for the CSV
    csv_data = []

    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        with open(json_path, 'r') as file:
            try:
                content = json.load(file)
                # Extract positive captions from the JSON
                positive_caption = ', '.join(content.get('positive_caption', []))  # Combine captions into a single string

                # Remove the `.json` extension from the file name to get the base image name
                image_file = os.path.splitext(json_file)[0]

                # Append to CSV data
                csv_data.append([image_file, positive_caption])

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing file {json_file}: {e}")

    # Write to the CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(['Image File', 'Positive Captions'])
        # Write rows
        writer.writerows(csv_data)

    print(f"CSV file has been created at {output_csv}.")

# Execute the function
process_json_files(json_dir, output_csv)