import pandas as pd
import shutil
import os
import re

# File paths
csv_file = r"E:\ProcessedV3\6-images_tags_reach.csv" # CSV containing selected images
source_folder = r"E:\Dataset\images"  # Folder containing all images
destination_folder = r"E:\ProcessedV3\selected_images"  # Folder for selected images

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Load CSV file
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Clean column names

# Helper function to extract numeric part from image filenames in the images folder
def extract_numeric_filename(filename):
    match = re.search(r"(\d+)\.jpg", filename)  # Extract numeric part before ".jpg"
    return match.group(1) if match else None

# Build a mapping of numeric IDs to actual filenames in the images folder
image_filenames_dict = {}
for img_filename in os.listdir(source_folder):
    numeric_part = extract_numeric_filename(img_filename)
    if numeric_part:
        image_filenames_dict[numeric_part] = img_filename  # Store full filename

# Copy images
missing_files = []
for index, row in df.iterrows():
    image_filename = row["Image_file_name"].strip()  # Image name from CSV (e.g., "1931181234042823537.jpg")
    numeric_id = extract_numeric_filename(image_filename)  # Extract numeric ID

    if numeric_id and numeric_id in image_filenames_dict:
        actual_filename = image_filenames_dict[numeric_id]  # Get correct image filename from folder
        source_path = os.path.join(source_folder, actual_filename)
        destination_path = os.path.join(destination_folder, image_filename)  # Save with just image name

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            print(f"✅ Copied: {image_filename}")
        else:
            missing_files.append(image_filename)
            print(f"⚠️ File not found: {image_filename}")
    else:
        missing_files.append(image_filename)
        print(f"⚠️ Image not found for ID: {numeric_id}")

# Save missing files list
if missing_files:
    missing_file_path = os.path.join(destination_folder, "missing_images.txt")
    with open(missing_file_path, "w") as f:
        for file in missing_files:
            f.write(file + "\n")

print(f"\n✅ All available images copied to {destination_folder}.")
print(f"⚠️ Missing images logged in 'missing_images.txt'.")
