import os
import json
import random
import shutil
from collections import defaultdict

# Define paths for attribute and relation
categories = {
    "attribute": {
        "subfolders": ["data_json/attribute/vaw", "data_json/attribute/vg"],  # Subfolders inside "attribute"
        "json_files": ["action.json", "color.json", "material.json", "size.json", "state.json"],  # JSON files
    },
    "relation": {
        "json_folder": "data_json/relation",  # Folder for relation JSON files
        "json_files": ["hake_action.json", "swig_action.json", "action.json", "spatial.json"],  # JSON files
    }
}

# Root directory for images
image_root_dir = "vl_datasets"  # Directory containing all images

# Output directories
output_json_folder = "final_sampled_json"  # Folder to store sampled JSON files
output_image_folder = "final_sampled_images"  # Folder to store sampled images
os.makedirs(output_json_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# Sampling limits
max_samples_per_category = 500  # Limit to 500 samples per category

# Track copied images to avoid duplicates
copied_images = set()

# Function to locate an image file under the root directory
def find_image(image_path, root_dir, specific_folder=None):
    """
    Finds an image file. If a specific folder is provided, it looks in that folder first.
    """
    if specific_folder:
        specific_path = os.path.join(root_dir, specific_folder, image_path)
        if os.path.exists(specific_path):
            return specific_path

    # Default: Search directly in the root directory
    full_path = os.path.join(root_dir, image_path)
    if os.path.exists(full_path):
        return full_path

    return None

# Function to sample entries with valid images
def sample_with_image_check(data, num_samples, root_dir, specific_folder=None):
    sampled_data = []
    remaining_data = data.copy()
    while len(sampled_data) < num_samples:
        entry = random.choice(remaining_data)
        image_path = entry[0]  # Assume the first element is the image path

        # Check if the image has already been copied
        if image_path in copied_images:
            remaining_data.remove(entry)
            continue

        # Check if the image exists
        full_image_path = find_image(image_path, root_dir, specific_folder)
        if full_image_path:
            # Copy the image and add to the set
            shutil.copy(full_image_path, os.path.join(output_image_folder, os.path.basename(full_image_path)))
            copied_images.add(image_path)
            sampled_data.append(entry)
            remaining_data.remove(entry)
        else:
            print(f"Skipping entry with missing image: {image_path}")

        # Stop if there are no more entries to sample
        if not remaining_data:
            print(f"Not enough valid images to sample {num_samples} entries. Collected {len(sampled_data)}.")
            break
    return sampled_data

# Process each category
category_counts = defaultdict(int)  # Track counts for each category
for category, details in categories.items():
    json_files = details.get("json_files", [])
    num_files = len(json_files)
    if num_files == 0:
        print(f"No JSON files found for category {category}. Skipping.")
        continue

    # Calculate samples per file dynamically
    samples_per_file = max_samples_per_category // num_files
    total_samples = 0  # Track total samples for the category

    if "subfolders" in details:  # Attribute category
        for subfolder in details["subfolders"]:
            for json_file in json_files:
                if total_samples >= max_samples_per_category:
                    break

                json_path = os.path.join(subfolder, json_file)
                if not os.path.exists(json_path):
                    print(f"File not found: {json_path}. Skipping.")
                    continue

                with open(json_path, "r") as f:
                    data = json.load(f)

                # Determine the number of samples to draw
                samples_to_draw = min(samples_per_file, max_samples_per_category - total_samples)

                sampled_data = sample_with_image_check(data, samples_to_draw, image_root_dir)

                # Save sampled JSON data
                subfolder_name = os.path.basename(subfolder)  # e.g., "vaw" or "vg"
                sampled_json_path = os.path.join(output_json_folder, f"{subfolder_name}_sampled_{json_file}")
                with open(sampled_json_path, "w") as f:
                    json.dump(sampled_data, f, indent=4)

                total_samples += len(sampled_data)
                category_counts[category] += len(sampled_data)
                print(f"Processed {subfolder}/{json_file}: Selected {len(sampled_data)} entries and copied images.")

    elif "json_folder" in details:  # Relation category
        for json_file in json_files:
            if total_samples >= max_samples_per_category:
                break

            json_path = os.path.join(details["json_folder"], json_file)
            if not os.path.exists(json_path):
                print(f"File not found: {json_path}. Skipping.")
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            # Determine the number of samples to draw
            samples_to_draw = min(samples_per_file, max_samples_per_category - total_samples)

            # Specific folder for relation subcategories
            specific_folder = None
            if "swig_action.json" in json_file:
                specific_folder = "swig"

            sampled_data = sample_with_image_check(data, samples_to_draw, image_root_dir, specific_folder)

            # Save sampled JSON data
            sampled_json_path = os.path.join(output_json_folder, f"relation_sampled_{json_file}")
            with open(sampled_json_path, "w") as f:
                json.dump(sampled_data, f, indent=4)

            total_samples += len(sampled_data)
            category_counts[category] += len(sampled_data)
            print(f"Processed relation/{json_file}: Selected {len(sampled_data)} entries and copied images.")

# Print summary of results
print("All categories processed successfully!")
print(f"Sampled JSON files are saved in: {output_json_folder}")
print(f"Sampled images are saved in: {output_image_folder}")
print(f"Summary of sampled entries:")
for category, count in category_counts.items():
    print(f"  - {category}: {count} entries")
print(f"Total unique images: {len(copied_images)}")


# Paths to sampled JSON files
json_folder = "final_sampled_json"
output_combined_json = "final_dataset.json"

# Combine all JSON files into one list
combined_data = []

for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        with open(os.path.join(json_folder, json_file), "r") as f:
            data = json.load(f)
            combined_data.extend(data)

# Save the combined JSON file
with open(output_combined_json, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"Combined JSON file saved as {output_combined_json}")