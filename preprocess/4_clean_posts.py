import pandas as pd
import re

# File Paths
input_file = r"E:\ProcessedV3\3-selected_posts_enriched.csv"
output_file = r"E:\ProcessedV3\4-cleaned_selected_posts.csv"

# Load CSV file
df = pd.read_csv(input_file, encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip()  # Clean column names

# Function to process hashtags
def process_hashtags(text):
    """Convert string to list of valid English hashtags, remove non-English hashtags."""
    if pd.isna(text) or text.strip() == "":
        return None  # Remove empty hashtags

    hashtags = text.split(",")  # Assuming hashtags are comma-separated
    hashtags = [ht.strip() for ht in hashtags if re.match(r"^#[A-Za-z]+$", ht.strip())]  # Keep only English hashtags

    return hashtags if hashtags else None  # Return None if no valid hashtags

# Apply cleaning function
df["Hashtags"] = df["Hashtags"].astype(str).apply(process_hashtags)

# Remove rows with no valid hashtags
df_filtered = df.dropna(subset=["Hashtags"])

# Save cleaned data
df_filtered.to_csv(output_file, index=False, encoding="utf-8")

# Summary
print(f"✅ Cleaned CSV saved: {output_file}")
print(f"🔹 Total posts before: {len(df)}")
print(f"🔹 Total posts after cleanup: {len(df_filtered)}")
print(f"⚠️ Removed {len(df) - len(df_filtered)} posts with no valid English hashtags.")
