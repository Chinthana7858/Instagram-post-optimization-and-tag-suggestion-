import pandas as pd

file_path = r"E:\Dataset\influencers.csv"
output_path = r"E:\ProcessedV3\1-selected_users.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Columns before filtering:", df.columns.tolist())

# Sample max 1000 users per category (and preserve 'Category')
df_selected = (
    df.groupby("Category", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 1000), random_state=42))
    .reset_index(drop=True)
)

print("Columns after filtering:", df_selected.columns.tolist())

# Save
df_selected.to_csv(output_path, index=False)
print(f"✅ Done! Saved {len(df_selected)} users to {output_path}")
