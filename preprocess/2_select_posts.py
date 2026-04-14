import pandas as pd
import os

# File paths
mapping_file = r"E:\Dataset\JSON-Image_files_mapping.txt"
selected_users_file = r"E:\ProcessedV3\1-selected_users.csv"
output_file = r"E:\ProcessedV3\2-selected_posts_mapping.csv"

# Ensure the output file is not open elsewhere
if os.path.exists(output_file):
    try:
        os.remove(output_file)
    except PermissionError:
        print(f"ERROR: Cannot delete {output_file}. Close the file and try again.")
        exit(1)

# Load selected users
df_users = pd.read_csv(selected_users_file)

# Load post mappings
column_names = ["influencer_name", "JSON_PostMetadata_file_name", "Image_file_name"]
df_posts = pd.read_csv(mapping_file, sep="\t", names=column_names, skiprows=1)

# Clean column names
df_posts.columns = df_posts.columns.str.strip()
df_users.columns = df_users.columns.str.strip()

# Merge selected users with post mappings
df_merged = df_posts.merge(df_users, left_on="influencer_name", right_on="Username", how="inner")

# Users with posts found
found_users = df_merged["influencer_name"].unique().tolist()

# Users with no posts found
all_selected_users = df_users["Username"].unique().tolist()
not_found_users = list(set(all_selected_users) - set(found_users))

# Select up to 10 posts per user (Fixed Warning)
df_selected = (
    df_merged.groupby("influencer_name", group_keys=False)
    .apply(lambda x: x.sample(n=min(30, len(x)), random_state=42), include_groups=False)
    .reset_index(drop=True)
)


# Save to CSV (Ensure it's not open elsewhere)
df_selected.to_csv(output_file, index=False)

# Print results
print(f"Total users in selected list: {len(all_selected_users)}")
print(f"Users with posts found: {len(found_users)}")
print(f"Users with NO posts found: {len(not_found_users)}")

print(f"\nCSV file saved with {len(df_selected)} selected posts.")
