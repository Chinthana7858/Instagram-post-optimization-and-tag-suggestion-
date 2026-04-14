import pandas as pd

# File Paths
input_file = r"E:\ProcessedV3\5-images_with_tags.csv"    # Use the transformed file
output_file = r"E:\ProcessedV3\6-images_tags_reach.csv"  # Final output

# Load CSV file
df = pd.read_csv(input_file, encoding="utf-8")
df.columns = df.columns.str.strip()

# Define weight for comments (w_C)
w_C = 2

# Compute IHC_h (Reach Contribution)
df["IHC_h"] = ((df["Likes"] + w_C * df["Comments"]) / df["#Followers"]) * 100

# Save updated CSV with IHC_h column
df.to_csv(output_file, index=False, encoding="utf-8")

# Summary
print(f"Final CSV saved at: {output_file}")
print(f"Total images processed: {len(df)}")
print(f"IHC_h calculation added successfully.")
