import pandas as pd
import json
import os

# File Paths
csv_file = r"E:\ProcessedV3\2-selected_posts_mapping.csv"
json_folder = r"E:\Dataset\posts\posts"
output_file = r"E:\ProcessedV3\3-selected_posts_enriched.csv"

# Load CSV file
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Clean column names

# Lists to store extracted data
likes_list = []
comments_list = []
hashtags_list = []

# Extract Data from JSON Files
for index, row in df.iterrows():
    influencer_name = row["Username"]
    post_id = row["JSON_PostMetadata_file_name"].replace(".info", "")  # Extract Post ID

    # Construct correct JSON filename format (Username-PostID.info)
    json_filename = f"{influencer_name}-{post_id}.info"
    json_path = os.path.join(json_folder, json_filename)

    # Ensure file exists and is not empty
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Extract likes count
            likes = data.get("edge_media_preview_like", {}).get("count", 0)

            # Extract comments count
            comments = data.get("edge_media_preview_comment", {}).get("count", 0)

            # Extract hashtags from caption
            caption = data.get("edge_media_to_caption", {}).get("edges", [{}])
            hashtags = []
            if caption and "node" in caption[0] and "text" in caption[0]["node"]:
                text = caption[0]["node"]["text"]
                hashtags = [word for word in text.split() if word.startswith("#")]  # Extract hashtags

            likes_list.append(likes)
            comments_list.append(comments)
            hashtags_list.append(", ".join(hashtags))  # Store as comma-separated hashtags

        except json.JSONDecodeError:
            print(f"⚠️ Corrupt JSON: {json_filename}, skipping...")
            likes_list.append(0)
            comments_list.append(0)
            hashtags_list.append("")
    else:
        print(f"⚠️ File not found or empty: {json_filename}")
        likes_list.append(0)
        comments_list.append(0)
        hashtags_list.append("")

# Add extracted data to DataFrame
df["Likes"] = likes_list
df["Comments"] = comments_list
df["Hashtags"] = hashtags_list

# Save updated CSV
df.to_csv(output_file, index=False)

print(f"✅ Enriched CSV file saved at: {output_file}")
