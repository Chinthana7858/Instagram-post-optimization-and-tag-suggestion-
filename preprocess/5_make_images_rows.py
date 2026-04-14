import pandas as pd
import ast

# File Paths
input_csv = r"E:\ProcessedV3\4-cleaned_selected_posts.csv"  # input file
output_csv = r"E:\ProcessedV3\5-images_with_tags.csv"        # output file

# Load Data
df = pd.read_csv(input_csv)

# Strip whitespace from column headers
df.columns = df.columns.str.strip()

# ✅ Include Category column here
columns_to_keep = ["Username", "Category", "#Followers", "#Followees", "Likes", "Comments", "Hashtags"]

# Expand DataFrame by splitting Image_file_name into multiple rows
expanded_rows = []

for _, row in df.iterrows():
    try:
        image_filenames = ast.literal_eval(row["Image_file_name"])  # Convert image list from string
        hashtags = ast.literal_eval(row["Hashtags"])                # Convert hashtag string to list

        for img in image_filenames:
            expanded_rows.append([
                img,
                row["Username"],
                row["Category"],             # ✅ Include category value
                row["#Followers"],
                row["#Followees"],
                row["Likes"],
                row["Comments"],
                ", ".join(hashtags)          # Turn list into CSV-friendly string
            ])
    except (ValueError, SyntaxError):
        print(f"⚠️ Skipping invalid format in row: {row}")

# Create final DataFrame
df_transformed = pd.DataFrame(expanded_rows, columns=["Image_file_name"] + columns_to_keep)

# Save to CSV
df_transformed.to_csv(output_csv, index=False)

print(f"✅ Transformed dataset with Category saved at: {output_csv}")
