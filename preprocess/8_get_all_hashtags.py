import pandas as pd
import ast
from collections import Counter

# File Paths
input_file = r"E:\ProcessedV3\6-images_tags_reach.csv"
output_file = r"E:\ProcessedV3\7-hashtags_occurrences_100_plus.xlsx"

# Load the CSV file
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

# Ensure Hashtags column is treated correctly
df["Hashtags"] = df["Hashtags"].astype(str)

# Count hashtag occurrences
hashtag_counter = Counter()

for hashtags in df["Hashtags"]:
    try:
        hashtag_list = ast.literal_eval(hashtags) if hashtags.startswith("[") else hashtags.split(", ")
        for tag in hashtag_list:
            tag = tag.strip()
            if tag:
                hashtag_counter[tag] += 1
    except (SyntaxError, ValueError):
        print(f"⚠️ Skipping invalid format: {hashtags}")

# ✅ Filter hashtags that occurred more than 100 times
filtered_hashtags = {tag: count for tag, count in hashtag_counter.items() if count > 100}

# Convert to DataFrame
result_df = pd.DataFrame(list(filtered_hashtags.items()), columns=["Hashtag", "Occurrence"])
result_df = result_df.sort_values(by="Occurrence", ascending=False).reset_index(drop=True)

# Save to Excel
result_df.to_excel(output_file, index=False)

# Summary
print(f"✅ Hashtags with more than 100 occurrences saved to: {output_file}")
print(f"🔹 Total matching hashtags: {len(result_df)}")

